import torch
import torch.nn as nn
from backbone import PointNet2

class ANCSH(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the shared PN++
        self.backbone = PointNet2()
