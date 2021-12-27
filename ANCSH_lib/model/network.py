import torch
import torch.nn as nn
from backbone import PointNet2

class ANCSH(nn.Module):
    def __init__(self, num_parts):
        super().__init__()
        # Define the shared PN++
        self.backbone = PointNet2()
        # segmentation branch
        self.seg_layer = nn.Conv1d(128, 3, kernel_size=1, padding='valid')
        # NPCS branch 
        self.NPCS_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding='valid'),
            nn.Conv1d(128, 3*num_parts, kernel_size=1, padding='valid')
        )
        # NAOCS scale and translation
        self.scale_layer = nn.Conv1d(128, 1*num_parts, kernel_size=1, padding='valid')
        self.trans_layer = nn.Conv2d(128, 3*num_parts, kernel_size=1, padding='valid')
        # Confidence
        self.conf_layer = nn.Conv1d(128, 1, kernel_size=1, padding='valid')
        # Joint parameters
        self.joint_feature_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding='valid', bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.dropout(0.5),
            nn.Conv1d(128, 128, kernel_size=1, padding='valid', bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.dropout(0.5)
        )
        # Joint UNitVec, heatmap, joint_cls
        self.unitvec_layer = nn.Conv1d(128, 3, kernel_size=1, padding='valid')
        self.heatmap_layer = nn.Conv1d(128, 1, kernel_size=1, padding='valid')
        self.joint_cls_layer = nn.Conv1d(128, num_parts, kernel_size=1, padding='valid')


    def forward(self, input):
        features = self.backbone(input)
        pred_seg = self.seg_layer(features)
        pred_NPCS= self.NPCS_layer(features)
        pred_scale = self.scale_layer(features)
        pred_trans = self.trans_layer(features)
        pred_conf = self.conf_layer(features)

        joint_features = self.joint_feature_layer(features)
        pred_unitvec = self.unitvec_layer(joint_features)
        pred_heatmap = self.heatmap_layer(joint_features)
        pred_joint_cls = self.joint_cls_layer(joint_features)

        

        
