from ANCSH_lib.model import ANCSH
from ANCSH_lib.data import ANCSHDataset
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ANCSHTrainer:
    def __init__(self, data_path, network_type, num_parts, max_epochs, lr=0.001, device=None):
        # data_path is a dictionary {'train', 'test'}
        if device == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.writer = SummaryWriter()

        self.network_type = network_type
        self.num_parts = num_parts
        self.max_epochs = max_epochs
        self.model = self.build_model()
        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)

        self.data_path = data_path
        self.train_loader = torch.utils.data.DataLoader(
            ANCSHDataset(self.data_path["train"]),
            batch_size=16,
            shuffle=True,
            num_workers=4,
        )

    def build_model(self):
        model = ANCSH(self.network_type, self.num_parts)
        return model

    def train(self):
        self.model.train()
        for epoch in range(self.max_epochs):
            avg_loss = 0.0
            for camera_per_point, gt_dict in self.train_loader:
                # Move the tensors to the device
                camera_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict:
                    gt[k] = v.to(self.device)
                # Get the loss
                pred = self.model(camera_per_point)
                loss_dict = self.model.losses(pred, gt)

                loss = torch.tensor(0.0, device=self.device)
                # todo: add loss weights to the config
                for k, v in loss_dict:
                    if k == "npcs_loss":
                        loss += 10 * v
                    else:
                        loss += v

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

    def test(self):
        test_loader = torch.utils.data.DataLoader(
            ANCSHDataset(
                self.data_path["test"], batch_size=16, shuffle=False, num_workers=4
            )
        )
        self.model.eval()
        with torch.no_grad():
            for camera_per_point, gt_dict in test_loader:
                # Move the tensors to the device
                camera_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict:
                    gt[k] = v.to(self.device)
                # Get the loss
                pred = self.model(camera_per_point)
                # todo: Save the results
                pass
