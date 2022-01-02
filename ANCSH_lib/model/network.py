import torch
import torch.nn as nn
import torch.nn.functional as F
from ANCSH_lib.model.backbone import PointNet2
import ANCSH_lib.model.loss as loss


class ANCSH(nn.Module):
    def __init__(self, network_type, num_parts):
        super().__init__()
        self.network_type = network_type

        # Define the shared PN++
        self.backbone = PointNet2()

        if self.network_type == "ancsh":
            # segmentation branch
            self.seg_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
            # NPCS branch
            self.npcs_layer = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.Conv1d(128, 3 * num_parts, kernel_size=1, padding=0),
            )
            # NAOCS scale and translation
            self.scale_layer = nn.Conv1d(
                128, 1 * num_parts, kernel_size=1, padding=0
            )
            self.trans_layer = nn.Conv1d(
                128, 3 * num_parts, kernel_size=1, padding=0
            )
            # Joint parameters
            self.joint_feature_layer = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
            # Joint UNitVec, heatmap, joint_cls
            self.axis_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
            self.unitvec_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
            self.heatmap_layer = nn.Conv1d(128, 1, kernel_size=1, padding=0)
            self.joint_cls_layer = nn.Conv1d(
                128, num_parts, kernel_size=1, padding=0
            )
        elif self.network_type == "npcs":
            # segmentation branch
            self.seg_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
            # NPCS branch
            self.npcs_layer = nn.Conv1d(
                128, 3 * num_parts, kernel_size=1, padding=0
            )
        else:
            raise ValueError("Not Implemented for the network type")

    def forward(self, input):
        features = self.backbone(input)
        pred_seg_per_point = self.seg_layer(features).transpose(1, 2)
        pred_npcs_per_point = self.npcs_layer(features).transpose(1, 2)
        if self.network_type == "ancsh":
            pred_scale_per_point = self.scale_layer(features).transpose(1, 2)
            pred_trans_per_point = self.trans_layer(features).transpose(1, 2)

            joint_features = self.joint_feature_layer(features)
            pred_axis_per_point = self.axis_layer(joint_features).transpose(1, 2)
            pred_unitvec_per_point = self.unitvec_layer(joint_features).transpose(1, 2)
            pred_heatmap_per_point = self.heatmap_layer(joint_features).transpose(1, 2)
            pred_joint_cls_per_point = self.joint_cls_layer(joint_features).transpose(1, 2)

        # Process the predicted things
        pred_seg_per_point = F.softmax(pred_seg_per_point, dim=2)
        pred_npcs_per_point = F.sigmoid(pred_npcs_per_point)

        if self.network_type == "ancsh":
            pred_scale_per_point = F.sigmoid(pred_scale_per_point)
            pred_trans_per_point = F.tanh(pred_trans_per_point)

            pred_heatmap_per_point = F.sigmoid(pred_heatmap_per_point)
            pred_unitvec_per_point = F.tanh(pred_unitvec_per_point)
            pred_axis_per_point = F.tanh(pred_axis_per_point)
            pred_joint_cls_per_point = F.softmax(pred_joint_cls_per_point, dim=2)

            # Calculate the NAOCS per point
            pred_scale_per_point_repeat = pred_scale_per_point.repeat(1, 1, 3)
            pred_naocs_per_point = (
                pred_npcs_per_point * pred_scale_per_point_repeat + pred_trans_per_point
            )

        pred = {
            "seg_per_point": pred_seg_per_point,
            "npcs_per_point": pred_npcs_per_point,
        }

        if self.network_type == "ancsh":
            pred.update(
                {
                    "heatmap_per_point": pred_heatmap_per_point,
                    "unitvec_per_point": pred_unitvec_per_point,
                    "axis_per_point": pred_axis_per_point,
                    "joint_cls_per_point": pred_joint_cls_per_point,
                    "scale_per_point": pred_scale_per_point,
                    "trans_per_point": pred_trans_per_point,
                    "naocs_per_point": pred_naocs_per_point,
                }
            )

        return pred

    def losses(self, pred, gt):
        # The returned loss is a value
        num_parts = pred["seg_per_point"].shape[2]
        # Convert the gt['seg_per_point'] into gt_seg_onehot B*N*K
        gt_seg_onehot = F.one_hot(gt["seg_per_point"], num_class=num_parts)
        # pred['seg_per_point']: B*N*K, gt_seg_onehot: B*N*K
        seg_loss = loss.compute_miou_loss(pred["seg_per_point"], gt_seg_onehot)
        # pred['npcs_per_point']: B*N*3K, gt['npcs_per_point']: B*N*3, gt_seg_onehot: B*N*K
        npcs_loss = loss.compute_coorindate_loss(
            pred["npcs_per_point"],
            gt["npcs_per_point"],
            num_parts=num_parts,
            gt_seg_onehot=gt_seg_onehot,
        )
        if self.network_type == "ancsh":
            # pred['naocs_per_point']: B*N*3K, gt['naocs_per_point']: B*N*3, gt_seg_onehot: B*N*K
            naocs_loss = loss.compute_coorindate_loss(
                pred["naocs_per_point"],
                gt["naocs_per_point"],
                num_parts=num_parts,
                gt_seg_onehot=gt_seg_onehot,
            )

            # Get the useful joint mask, gt['joint_cls_per_point'] == 0 means that that point doesn't have a corresponding joint
            # B*N 
            gt_joint_mask = (gt["joint_cls_per_point"] > 0).float()
            # Get the heatmap and unitvec map, the loss should only be calculated for revolute joint
            gt_revolute_mask = None
            revlote_index = torch.where(gt["joint_type"] == 1)[0]
            assert gt["joint_type"][0] == -1
            for i in revlote_index:
                if gt_revolute_mask == None:
                    gt_revolute_mask = (gt["joint_cls_per_point"] == i)
                else:
                    gt_revolute_mask = torch.logical_or(gt_revolute_mask, (gt["joint_cls_per_point"] == i))
            gt_revolute_mask = gt_revolute_mask.float()
            # pred['heatmap_per_point']: B*N*1, gt['heatmap_per_point']: B*N, gt_revolute_mask: B*N
            heatmap_loss = loss.compute_vect_loss(
                pred["heatmap_per_point"], gt["heatmap_per_point"], mask=gt_revolute_mask
            )
            # pred['unitvec_per_point']: B*N*3, gt['unitvec_per_point']: B*N*3, gt_revolute_mask: B*N
            unitvec_loss = loss.compute_vect_loss(
                pred["unitvec_per_point"], gt["unitvec_per_point"], mask=gt_revolute_mask
            )
            # pred['axis_per_point]: B*N*3, gt['axis_per_point']: B*N*3, gt_joint_mask: B*N
            axis_loss = loss.compute_vect_loss(
                pred["axis_per_point"], gt["axis_per_point"], mask=gt_joint_mask
            )

            # Conver the gt['joint_cls_per_point'] into gt_joint_cls_onehot B*N*K
            gt_joint_cls_onehot = F.one_hot(
                gt["joint_cls_per_point"], num_class=num_parts
            )
            joint_loss = loss.compute_miou_loss(
                pred["joint_cls_per_point"], gt_joint_cls_onehot
            )

        loss_dict = {
            "seg_loss": seg_loss,
            "npcs_loss": npcs_loss,
        }

        if self.network_type == "ancsh":
            loss_dict.update(
                {
                    "naocs_loss": naocs_loss,
                    "heatmap_loss": heatmap_loss,
                    "unitvec_loss": unitvec_loss,
                    "axis_loss": axis_loss,
                    "joint_loss": joint_loss,
                }
            )

        return loss_dict
