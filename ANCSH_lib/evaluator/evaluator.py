import h5py
import numpy as np


def get_3d_bbox(scale, shift=0):
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


class ANCSHEvaluator:
    def __init__(self, cfg, combined_results_path):
        self.cfg = cfg
        self.f_combined = h5py.File(combined_results_path, "r+")
        self.instances = sorted(self.f_combined.keys())
        self.results = {}

    def process_ANCSH(self):
        num_parts = self.cfg.evaluation.num_parts
        results = []
        for instance in self.instances:
            ins_combined = self.f_combined[instance]
            # Get the useful information from the combined_results
            pred_seg_per_point = ins_combined["pred_seg_per_point"]
            pred_npcs_per_point = ins_combined["pred_npcs_per_point"]
            gt_npcs_scale = ins_combined["gt_npcs_scale"]
            gt_npcs_rt = ins_combined["gt_npcs_rt"]
            pred_npcs_scale = ins_combined["pred_npcs_scale"]
            pred_npcs_rt = ins_combined["pred_npcs_rt"]

            pred_partIndex_per_point = np.argmax(pred_seg_per_point, axis=1)

            # Get the norm factors and corners used to calculate NPCS to calculate the 3dbbx
            gt_norm_factors = ins_combined["gt_norm_factors"]
            gt_corners = ins_combined["gt_corners"]

            result = {"pose_err_scale": [], "pose_err_volume": []}
            for partIndex in range(num_parts):
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
                pred_partIndex = np.where(pred_partIndex_per_point == partIndex)[0]
                centered_npcs = pred_npcs_per_point[
                    pred_partIndex, 3 * partIndex : 3 * (partIndex + 1)
                ]
                ## todo: Here their scale is not that proper
                pred_scale = 2 * np.max(abs(centered_npcs), axis=0)
                pred_3dbbx = get_3d_bbox(pred_scale, np.array([0.5, 0.5, 0.5]))

                # Record the pose scale and volume error
                result["pose_err_scale"].append(
                    np.linalg.norm(
                        pred_scale * pred_npcs_scale[partIndex]
                        - gt_scale * gt_npcs_scale[partIndex]
                    )
                )
                result["pose_err_volume"].append(
                    pred_scale[0]
                    * pred_scale[1]
                    * pred_scale[2]
                    * pred_npcs_scale[partIndex]
                    / (
                        gt_scale[0]
                        * gt_scale[1]
                        * gt_scale[2]
                        * gt_npcs_scale[partIndex]
                    )
                    - 1
                )

                # Calcualte the mean relative error for the parts
                # This evaluation metric seems wierd, don't code it 
                # https://github.com/dragonlong/articulated-pose/blob/master/evaluation/eval_pose_err.py#L263
                # https://github.com/dragonlong/articulated-pose/blob/master/evaluation/eval_pose_err.py#L320

                # Calculatet the 3diou for each part
                

            results.append(result)

    def print_and_save(self):
        pass
