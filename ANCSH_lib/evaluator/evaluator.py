import os
import h5py
import itertools

import numpy as np
import logging

from tools.utils import io


def get_3d_bbox(scale, shift=np.zeros(3)):
    """
    Input:
        scale: [3]
        shift: [3]
    Return
        bbox_3d: [3, N]

    """

    bbox_3d = (
        np.array(
            [
                [scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
            ]
        )
        + shift
    )
    return bbox_3d


def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1 > 0, p1 < np.dot(u1, u1))
    p2 = np.logical_and(p2 > 0, p2 < np.dot(u2, u2))
    p3 = np.logical_and(p3 > 0, p3 < np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)


def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union == 0:
        return 1
    else:
        return intersect / float(union)


def axis_diff_degree(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    r_diff = (
        np.arccos(np.clip(np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), a_min=-1.0, a_max=1.0))
        * 180
        / np.pi
    )
    return min(r_diff, 180 - r_diff)


def dist_between_3d_lines(p1, e1, p2, e2):
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    p = p1 - p2

    if np.linalg.norm(orth_vect) == 0:
        dist = np.linalg.norm(np.cross(p, e1)) / np.linalg.norm(e1)
    else:
        dist = np.linalg.norm(np.dot(orth_vect, p)) / np.linalg.norm(orth_vect)

    return dist


class ANCSHEvaluator:
    def __init__(self, cfg, combined_results_path, num_parts):
        self.cfg = cfg
        self.f_combined = h5py.File(combined_results_path, "r+")
        self.instances = sorted(self.f_combined.keys())
        self.num_parts = num_parts
        self.log = logging.getLogger('evaluator')
        self.results = {}

    def process_ANCSH(self):
        self.results = []
        for instance in self.instances:
            ins_combined = self.f_combined[instance]
            # Get the useful information from the combined_results
            pred_seg_per_point = ins_combined["pred_seg_per_point"][:]
            pred_npcs_per_point = ins_combined["pred_npcs_per_point"][:]
            pred_naocs_per_point = ins_combined["pred_naocs_per_point"][:]
            # pred_seg_per_point = ins_combined["gt_seg_per_point"][:]
            # pred_npcs_per_point = ins_combined["gt_npcs_per_point"][:]
            # pred_naocs_per_point = ins_combined["gt_naocs_per_point"][:]
            gt_naocs_per_point = ins_combined["gt_naocs_per_point"][:]

            pred_unitvec_per_point = ins_combined["pred_unitvec_per_point"][:]
            # pred_unitvec_per_point = ins_combined["gt_unitvec_per_point"][:]
            gt_unitvec_per_point = ins_combined["gt_unitvec_per_point"][:]
            pred_heatmap_per_point = ins_combined["pred_heatmap_per_point"][:]
            # pred_heatmap_per_point = ins_combined["gt_heatmap_per_point"][:]
            gt_heatmap_per_point = ins_combined["gt_heatmap_per_point"][:]
            pred_axis_per_point = ins_combined["pred_axis_per_point"][:]
            # pred_axis_per_point = ins_combined["gt_axis_per_point"][:]
            gt_axis_per_point = ins_combined["gt_axis_per_point"][:]
            pred_joint_cls_per_point = ins_combined["pred_joint_cls_per_point"][:]
            # pred_joint_cls_per_point = ins_combined["gt_joint_cls_per_point"][:]
            gt_joint_cls_per_point = ins_combined["gt_joint_cls_per_point"][:]

            gt_npcs_scale = ins_combined["gt_npcs2cam_scale"][:]
            gt_npcs_rt = ins_combined["gt_npcs2cam_rt"][:]
            gt_naocs_scale = ins_combined["gt_naocs2cam_scale"][:]
            gt_naocs_rt = ins_combined["gt_naocs2cam_rt"][:]
            pred_npcs_scale = ins_combined["pred_npcs2cam_scale"][:]
            pred_npcs_rt = ins_combined["pred_npcs2cam_rt"][:]
            # pred_npcs_scale = ins_combined["gt_npcs2cam_scale"][:]
            # pred_npcs_rt = ins_combined["gt_npcs2cam_rt"][:]

            gt_jointIndex_per_point = gt_joint_cls_per_point

            # Get the norm factors and corners used to calculate NPCS to calculate the 3dbbx
            gt_norm_factors = ins_combined["gt_norm_factors"]
            gt_corners = ins_combined["gt_norm_corners"]

            result = {
                "err_pose_scale": [],
                "err_pose_volume": [],
                "iou_cam_3dbbx": [],
                "gt_cam_3dbbx": [],
                "pred_cam_3dbbx": [],
                "pred_joint_axis_naocs": [],
                "pred_joint_pt_naocs": [],
                "gt_joint_axis_naocs": [],
                "gt_joint_pt_naocs": [],
                "pred_joint_axis_cam": [],
                "pred_joint_pt_cam": [],
                "gt_joint_axis_cam": [],
                "gt_joint_pt_cam": [],
                "err_joint_axis": [],
                "err_joint_line": [],
            }
            for partIndex in range(self.num_parts):
                norm_factor = gt_norm_factors[partIndex]
                corner = gt_corners[partIndex]
                npcs_corner = np.zeros_like(corner)
                # Calculatet the corners in npcs
                npcs_corner[0] = (
                    np.array([0.5, 0.5, 0.5])
                    - 0.5 * (corner[1] - corner[0]) * norm_factor
                )
                npcs_corner[1] = (
                    np.array([0.5, 0.5, 0.5])
                    + 0.5 * (corner[1] - corner[0]) * norm_factor
                )
                # Calculate the gt bbx
                gt_scale = npcs_corner[1] - npcs_corner[0]
                gt_3dbbx = get_3d_bbox(gt_scale, shift=np.array([0.5, 0.5, 0.5]))
                # Calculate the pred bbx
                pred_part_points_index = np.where(
                    pred_seg_per_point == partIndex
                )[0]
                centered_npcs = (
                    pred_npcs_per_point[
                        pred_part_points_index
                    ]
                    - 0.5
                )
                
                pred_scale = 2 * np.max(abs(centered_npcs), axis=0)
                pred_3dbbx = get_3d_bbox(pred_scale, np.array([0.5, 0.5, 0.5]))

                # Record the pose scale and volume error
                result["err_pose_scale"].append(
                    np.linalg.norm(
                        pred_scale * pred_npcs_scale[partIndex]
                        - gt_scale * gt_npcs_scale[partIndex]
                    )
                )
                # todo: whethere to take if it's smaller than 1, then it needs to consider the ratio
                ratio_pose_volume = pred_scale[0] * pred_scale[1] * pred_scale[2] * pred_npcs_scale[partIndex]**3 / (
                        gt_scale[0] * gt_scale[1] * gt_scale[2] * gt_npcs_scale[partIndex]**3
                    )
                if ratio_pose_volume > 1:
                    result["err_pose_volume"].append(
                        ratio_pose_volume - 1
                    )
                else:
                    result["err_pose_volume"].append(
                        1 / ratio_pose_volume - 1
                    )

                # Calcualte the mean relative error for the parts
                # This evaluation metric seems wierd, don't code it
                # https://github.com/dragonlong/articulated-pose/blob/master/evaluation/eval_pose_err.py#L263
                # https://github.com/dragonlong/articulated-pose/blob/master/evaluation/eval_pose_err.py#L320

                # Calculatet the 3diou for each part
                gt_scaled_3dbbx = gt_3dbbx * gt_npcs_scale[partIndex]
                pred_scaled_3dbbx = pred_3dbbx * pred_npcs_scale[partIndex]
                gt_cam_3dbbx = (
                    np.dot(gt_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, :3], gt_scaled_3dbbx.T).T
                    + gt_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, 3].T
                )
                pred_cam_3dbbx = (
                    np.dot(pred_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, :3], pred_scaled_3dbbx.T).T
                    + pred_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, 3].T
                )
                iou_cam_3dbbx = iou_3d(gt_cam_3dbbx, pred_cam_3dbbx)
                result["gt_cam_3dbbx"].append(gt_cam_3dbbx)
                result["pred_cam_3dbbx"].append(pred_cam_3dbbx)
                result["iou_cam_3dbbx"].append(iou_cam_3dbbx)

                # Calculate the evaluation metric for the joints
                # Calculate the scale and translation from naocs to npcs
                pred_npcs = pred_npcs_per_point[
                    pred_part_points_index
                ]
                pred_naocs = pred_naocs_per_point[
                    pred_part_points_index
                ]

                if partIndex == 0:
                    self.naocs_npcs_scale = np.std(np.mean(pred_npcs, axis=1)) / np.std(
                        np.mean(pred_naocs, axis=1)
                    )
                    self.naocs_npcs_translation = np.mean(
                        pred_npcs - self.naocs_npcs_scale * pred_naocs, axis=0
                    )

                if partIndex >= 1:
                    # joint 0 is meaningless, the joint index starts from 1
                    thres_r = self.cfg.evaluation.thres_r
                    # Calculate the predicted joint info
                    pred_offset = (
                        pred_unitvec_per_point
                        * (1 - pred_heatmap_per_point.reshape(-1, 1))
                        * thres_r
                    )   
                    pred_joint_pts = pred_naocs_per_point + pred_offset
                    pred_joint_points_index = np.where(
                        pred_joint_cls_per_point == partIndex
                    )[0]
                    pred_joint_axis = np.median(
                        pred_axis_per_point[pred_joint_points_index], axis=0
                    )
                    pred_joint_pt = np.median(
                        pred_joint_pts[pred_joint_points_index], axis=0
                    )
                    result["pred_joint_axis_naocs"].append(pred_joint_axis)
                    result["pred_joint_pt_naocs"].append(pred_joint_pt)

                    # Convert the pred joint into camera coordinate from naocs -> npcs -> camera
                    temp_joint_pt_npcs = (
                        pred_joint_pt * self.naocs_npcs_scale
                        + self.naocs_npcs_translation
                    )
                    pred_joint_pt_cam = (
                        np.dot(
                            pred_npcs_rt[0].reshape((4, 4), order='F')[:3, :3], pred_npcs_scale[0] * temp_joint_pt_npcs.T
                        ).T
                        + pred_npcs_rt[0].reshape((4, 4), order='F')[:3, 3]
                    )
                    pred_joint_axis_cam = np.dot(
                        pred_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, :3], pred_joint_axis.T
                    ).T
                    result["pred_joint_axis_cam"].append(pred_joint_axis_cam)
                    result["pred_joint_pt_cam"].append(pred_joint_pt_cam)
                    # Calculate the gt joint info
                    gt_offset = (
                        gt_unitvec_per_point
                        * (1 - gt_heatmap_per_point.reshape(-1, 1))
                        * thres_r
                    )
                    gt_joint_pts = gt_naocs_per_point + gt_offset
                    gt_joint_points_index = np.where(
                        gt_jointIndex_per_point == partIndex
                    )[0]
                    gt_joint_axis = np.median(
                        gt_axis_per_point[gt_joint_points_index], axis=0
                    )
                    gt_joint_pt = np.median(gt_joint_pts[gt_joint_points_index], axis=0)
                    result["gt_joint_axis_naocs"].append(gt_joint_axis)
                    result["gt_joint_pt_naocs"].append(gt_joint_pt)
                    # Conver the gt joint into camera coordinate using the naocs pose, naocs -> camera
                    gt_joint_pt_cam = (
                        np.dot(gt_naocs_rt.reshape((4, 4), order='F')[:3, :3], gt_naocs_scale * gt_joint_pt.T).T
                        + gt_naocs_rt.reshape((4, 4), order='F')[:3, 3]
                    )
                    gt_joint_axis_cam = np.dot(gt_naocs_rt.reshape((4, 4), order='F')[:3, :3], gt_joint_axis.T).T
                    result["gt_joint_axis_cam"].append(gt_joint_axis_cam)
                    result["gt_joint_pt_cam"].append(gt_joint_pt_cam)

                    # Calculate the error between the gt joints and pred joints in the camera coordinate
                    err_joint_axis = axis_diff_degree(
                        gt_joint_axis_cam, pred_joint_axis_cam
                    )
                    err_joint_line = dist_between_3d_lines(
                        gt_joint_pt_cam,
                        gt_joint_axis_cam,
                        pred_joint_pt_cam,
                        pred_joint_axis_cam,
                    )
                    result["err_joint_axis"].append(err_joint_axis)
                    result["err_joint_line"].append(err_joint_line)

            self.results.append(result)
        self.print_and_save()

    def print_and_save(self):
        # Print the mean errors for scale, volume
        err_pose_scale = [result["err_pose_scale"] for result in self.results]
        err_pose_volume = [result["err_pose_volume"] for result in self.results]
        mean_err_pose_scale = np.mean(err_pose_scale, axis=0)
        mean_err_pose_volume = np.mean(err_pose_volume, axis=0)
        self.log.info(f"Mean Error for pose scale: {mean_err_pose_scale}")
        self.log.info(f"Mean Error for pose volume: {mean_err_pose_volume}")

        # Print the mean iou for different parts
        iou_cam_3dbbx = [result["iou_cam_3dbbx"] for result in self.results]
        mean_iou_cam_3dbbx = np.mean(iou_cam_3dbbx, axis=0)
        self.log.info(f"Mean iou for different parts is: {mean_iou_cam_3dbbx}")

        # Print the mean error for joints in the camera coordinate
        err_joint_axis = [result["err_joint_axis"] for result in self.results]
        err_joint_line = [result["err_joint_line"] for result in self.results]
        mean_err_joint_axis = np.mean(err_joint_axis, axis=0)
        mean_err_joint_line = np.mean(err_joint_line, axis=0)
        self.log.info(f"Mean joint axis error in camera coordinate (degree): {mean_err_joint_axis}")
        self.log.info(f"Mean joint axis line distance in camera coordinate (m): {mean_err_joint_line}")

        io.ensure_dir_exists(self.cfg.paths.evaluation.output_dir)
        f = h5py.File(
            os.path.join(self.cfg.paths.evaluation.output_dir, self.cfg.paths.evaluation.prediction_filename),
            "w"
        )
        for k, v in self.f_combined.attrs.items():
            f.attrs[k] = v
        f.attrs["err_pose_scale"] = mean_err_pose_scale
        f.attrs["err_pose_volume"] = mean_err_pose_volume
        f.attrs["iou_cam_3dbbx"] = mean_iou_cam_3dbbx
        f.attrs["err_joint_axis"] = mean_err_joint_axis
        f.attrs["err_joint_line"] = mean_err_joint_line

        for i, ins in enumerate(self.instances):
            result = self.results[i]
            group = f.create_group(ins)
            for k, v in self.f_combined[ins].items():
                group.create_dataset(k, data=v, compression="gzip")
            for k, v in result.items():
                group.create_dataset(k, data=v, compression="gzip")
