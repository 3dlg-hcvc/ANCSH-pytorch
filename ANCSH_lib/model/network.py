import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.axis_layer = nn.Conv1d(128, 3, kernel_size=1, padding='valid')
        self.unitvec_layer = nn.Conv1d(128, 3, kernel_size=1, padding='valid')
        self.heatmap_layer = nn.Conv1d(128, 1, kernel_size=1, padding='valid')
        self.joint_cls_layer = nn.Conv1d(128, num_parts, kernel_size=1, padding='valid')


    def forward(self, input):
        features = self.backbone(input)
        pred_seg_per_point = self.seg_layer(features)
        pred_NPCS_per_point = self.NPCS_layer(features)
        pred_scale_per_point = self.scale_layer(features)
        pred_trans_per_point = self.trans_layer(features)
        pred_conf_per_point = self.conf_layer(features)

        joint_features = self.joint_feature_layer(features)
        pred_axis_per_point = self.axis_layer(joint_features)
        pred_unitvec_per_point = self.unitvec_layer(joint_features)
        pred_heatmap_per_point = self.heatmap_layer(joint_features)
        pred_joint_cls_per_point = self.joint_cls_layer(joint_features)

        # Process the predicted things
        pred_scale_per_point = F.sigmoid(pred_scale_per_point)
        pred_trans_per_point = F.tanh(pred_trans_per_point)
        
        pred_seg_per_point = F.softmax(pred_seg_per_point, dim=2)
        pred_conf_per_point = F.sigmoid(pred_conf_per_point)
        pred_NPCS_per_point = F.sigmoid(pred_NPCS_per_point)

        pred_heatmap_per_point = F.sigmoid(pred_heatmap_per_point)
        pred_unitvec_per_point = F.tanh(pred_unitvec_per_point)
        pred_axis_per_point = F.tanh(pred_axis_per_point)
        pred_joint_cls_per_point = F.softmax(pred_joint_cls_per_point, dim=2)
        
        # Calculate the NAOCS per point
        pred_scale_per_point_repeat = pred_scale_per_point.repeat(1, 1, 3)
        pred_NAOCS_per_point = pred_NPCS_per_point * pred_scale_per_point_repeat + pred_trans_per_point

        pred = {
            'seg_per_point': pred_seg_per_point,
            'NPCS_per_point': pred_NPCS_per_point,
            'conf_per_point': pred_conf_per_point,
            'heatmap_per_point': pred_heatmap_per_point,
            'unit_vec_per_point': pred_unitvec_per_point,
            'axis_per_point': pred_axis_per_point,
            'joint_cls_per_point': pred_joint_cls_per_point,
            'scale_per_point': pred_scale_per_point,
            'trans_per_point': pred_trans_per_point,
            'NAOCS_per_point': pred_NAOCS_per_point
        }


        
