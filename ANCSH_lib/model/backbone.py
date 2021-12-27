import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.nn.modules import dropout
from torch.nn.modules.batchnorm import BatchNorm1d


class PointNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the shared PN++
        self.sa_module_1 = PointnetSAModule(
            npoint=512,
            radius=0.2,
            nsample=64,
            mlp=[0, 64, 64, 128],
            bn=True,
            use_xyz=True,
        )

        self.sa_module_2 = PointnetSAModule(
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            bn=True,
            use_xyz=True,
        )

        self.sa_module_3 = PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 256, 512, 1024],
            bn=True,
            use_xyz=True,
        )

        self.fp_module_1 = PointnetFPModule(mlp=[256+1024, 256, 256])
        self.fp_module_2 = PointnetFPModule(mlp=[128+256, 256, 128])
        self.fp_module_3 = PointnetFPModule(mlp=[3+128, 128, 128, 128])

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.dropout(0.5)
        )


    def forward(self, input):
        l0_xyz = input[:, :, :3]
        # Here l0_features should be blank
        l0_features = input[:, :, 3:]

        l1_xyz, l1_features = self.sa_module_1(l0_xyz, l0_features)
        l2_xyz, l2_features = self.sa_module_2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa_module_3(l2_xyz, l2_features)

        l2_features = self.fp_module_1(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp_module_2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp_module_3(l0_xyz, l1_xyz, torch.cat((l0_xyz, l0_features), -1), l1_features)

        return self.fc_layer(l0_features)
