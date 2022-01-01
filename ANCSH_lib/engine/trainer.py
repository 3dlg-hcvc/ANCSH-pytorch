from torch.optim import optimizer
from ANCSH_lib.model import ANCSH
from ANCSH_lib.data import ANCSHDataset
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import h5py

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

        print(f"Below is the network structure:\n {self.model}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.network.lr, betas=(0.9, 0.99))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)

        self.data_path = data_path
        self.train_loader = torch.utils.data.DataLoader(
            ANCSHDataset(self.data_path["train"]),
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
            for camcs_per_point, gt_dict, joint_type in self.train_loader:
                # Move the tensors to the device
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict:
                    gt[k] = v.to(self.device)
                # Move the joint type to the device
                joint_type = joint_type.to(self.device)
                # Get the loss
                pred = self.model(camcs_per_point)
                loss_dict = self.model.losses(pred, gt, joint_type)

                loss = torch.tensor(0.0, device=self.device)
                loss_weight = self.cfg.network.loss_weight
                # use different loss weight to calculate the final loss
                for k, v in loss_dict:
                    if k not in loss_weight:
                        raise ValueError(f"No loss weight for {k}")
                    loss += loss_weight[k] * v

                # Used to calcuate the avg loss  
                if epoch_loss == None:
                    epoch_loss = loss_dict
                    epoch_loss["total_loss"] = loss
                else:
                    for k, v in loss_dict:
                        epoch_loss[k] += v
                    epoch_loss["total_loss"] += loss
                step_num += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            # Add the loss values into the tensorboard
            for k,v  in epoch_loss:
                self.writer.add_scalar(f"loss/{k}", v/step_num, epoch)
            
            if epoch % self.cfg.model_frequncy == 0 or epoch == self.max_epochs - 1:
                # Save the model 
                existDir(f"{self.cfg.paths.project_paths}")
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
        print(f"Load model from {inference_model}")
        checkpoint = torch.load(inference_model, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for camcs_per_point, gt_dict, joint_type in test_loader:
                # Move the tensors to the device
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict:
                    gt[k] = v.to(self.device)
                # Move the joint type to the device
                joint_type = joint_type.to(self.device)
                    
                pred = self.model(camcs_per_point)
                self.save_results(pred, camcs_per_point, gt, joint_type)
    
    def save_results(self, pred, camcs_per_point, gt, joint_type):
        # Save the results and gt into hdf5 for further optimization
        batch_size = pred["seg_per_point"].shape[0]
        f = h5py.File(f"{self.cfg.paths.test.output_dir}/pred_gt.h5", 'w')
        f.attrs["network_type"] = self.network_type
        for b in batch_size:
            group = f.create_group(f"{gt['filename'][b]}")
            group.attrs["filename"] = gt["filename"][b]
            group.create_dataset("camcs_per_point", data=camcs_per_point[b].detach().cpu().numpy(), compression="gzip")
            group.create_dataset("joint_type", data=joint_type[b].detach().cpu().numpy(), compression="gzip")
            for k, v in pred:
                # Save the pred
                group.create_dataset(f"pred_{k}", v[b].detach().cpu().numpy(), compression="gzip")
                # Save the gt
                group.create_dataset(f"gt_{k}", gt[k][b].detach().cpu().numpy(), compression="gzip")
            
    def resume_train(self, model):
        print(f"Load model from {model}")
        # Load the model
        checkpoint = torch.load(model, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)

        self.train()