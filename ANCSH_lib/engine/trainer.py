from torch.optim import optimizer
from ANCSH_lib.model import ANCSH
from ANCSH_lib.data import ANCSHDataset
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import h5py
import logging

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class ANCSHTrainer:
    def __init__(self, cfg, data_path, network_type, num_parts, device=None):
        self.cfg = cfg
        # data_path is a dictionary {'train', 'test'}
        if device == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.writer = SummaryWriter(cfg.paths.train.output_dir)

        self.network_type = network_type
        self.num_parts = num_parts
        self.max_epochs = cfg.network.max_epochs
        self.model = self.build_model()
        self.model.to(device)
        self.log = logging.getLogger('Network')

        self.log.info(f"Below is the network structure:\n {self.model}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.network.lr, betas=(0.9, 0.99))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)

        self.data_path = data_path
        self.train_loader = torch.utils.data.DataLoader(
            ANCSHDataset(self.data_path["train"], num_points=self.cfg.network.num_points),
            batch_size=cfg.network.batch_size,
            shuffle=True,
            num_workers=cfg.network.num_workers,
        )

    def __del__(self):
        self.writer.close()

    def build_model(self):
        model = ANCSH(self.network_type, self.num_parts)
        return model

    def train(self):
        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = None
            step_num = 0
            for camcs_per_point, gt_dict, id in self.train_loader:
                # Move the tensors to the device
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict.items():
                    gt[k] = v.to(self.device)
                # Get the loss
                pred = self.model(camcs_per_point)
                loss_dict = self.model.losses(pred, gt)

                loss = torch.tensor(0.0, device=self.device)
                loss_weight = self.cfg.network.loss_weight
                # use different loss weight to calculate the final loss
                for k, v in loss_dict.items():
                    if k not in loss_weight:
                        raise ValueError(f"No loss weight for {k}")
                    loss += loss_weight[k] * v

                # Used to calcuate the avg loss  
                if epoch_loss == None:
                    epoch_loss = loss_dict
                    epoch_loss["total_loss"] = loss
                else:
                    for k, v in loss_dict.items():
                        epoch_loss[k] += v
                    epoch_loss["total_loss"] += loss
                step_num += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            # Add the loss values into the tensorboard
            for k,v  in epoch_loss.items():
                epoch_loss[k] = v/step_num
                self.writer.add_scalar(f"loss/{k}", epoch_loss[k], epoch)

            if not epoch == 0 and epoch % self.cfg.log_frequency == 0:
                output_string = f"Epoch: {epoch}  "
                for k, v in epoch_loss.items():
                    output_string += f"{k}: {round(float(v), 5)}  "
                self.log.info(output_string)
            
            if not epoch == 0 and epoch % self.cfg.model_frequency == 0 or epoch == self.max_epochs - 1:
                # Save the model 
                existDir(f"{self.cfg.paths.train.output_dir}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f"{self.cfg.paths.train.output_dir}/model_{epoch}.pth")


    def test(self, inference_model):
        test_loader = torch.utils.data.DataLoader(
            ANCSHDataset(
                self.data_path["test"], batch_size=16, shuffle=False, num_workers=4
            )
        )
        # Load the model
        self.log.info(f"Load model from {inference_model}")
        checkpoint = torch.load(inference_model, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for camcs_per_point, gt_dict, id in test_loader:
                # Move the tensors to the device
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict.items():
                    gt[k] = v.to(self.device)
                    
                pred = self.model(camcs_per_point)
                self.save_results(pred, camcs_per_point, gt, id)
    
    def save_results(self, pred, camcs_per_point, gt, id):
        # Save the results and gt into hdf5 for further optimization
        batch_size = pred["seg_per_point"].shape[0]
        f = h5py.File(f"{self.cfg.paths.test.output_dir}/pred_gt.h5", 'w')
        f.attrs["network_type"] = self.network_type
        for b in batch_size:
            group = f.create_group(f"{id[b]}")
            group.create_dataset("camcs_per_point", data=camcs_per_point[b].detach().cpu().numpy(), compression="gzip")
            for k, v in pred.items():
                # Save the pred
                group.create_dataset(f"pred_{k}", v[b].detach().cpu().numpy(), compression="gzip")
            for k, v in gt.items(): 
                # Save the gt
                group.create_dataset(f"gt_{k}", gt[k][b].detach().cpu().numpy(), compression="gzip")
            
    def resume_train(self, model):
        self.log.info(f"Load model from {model}")
        # Load the model
        checkpoint = torch.load(model, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)

        self.train()