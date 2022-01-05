import os
import logging
import h5py
import numpy as np
from matplotlib import cm

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from tools.utils import io
from tools.visualization.viewer import Viewer
from ANCSH_lib.utils import NetworkType

log = logging.getLogger('ANCSH visualizer')


class ANCSHVisualizer:
    def __init__(self, npcs_data, ancsh_data, export=None, gt=False):
        self.npcs_data = npcs_data
        self.ancsh_data = ancsh_data
        self.gt = gt
        self.export_dir = export
        if self.export_dir is not None:
            io.ensure_dir_exists(self.export_dir)
        self.npcs_items = []
        self.ancsh_items = []
        self.show_flag = False
        self.render_flag = False
        self.export_flag = False

        self.width = 1024
        self.height = 768
        self.fig_ext = '.png'
        self.mesh_ext = '.ply'

    def parse_items(self):
        visit_groups = lambda name, node: self.npcs_items.append(name) if isinstance(node, h5py.Group) else None
        self.npcs_data.visititems(visit_groups)
        visit_groups = lambda name, node: self.ancsh_items.append(name) if isinstance(node, h5py.Group) else None
        self.ancsh_data.visititems(visit_groups)

    def render_options(self, viewer, name):
        if self.show_flag:
            viewer.show(window_size=[self.width, self.height], window_name=name)
        if self.render_flag:
            viewer.render(fig_path=os.path.join(self.export_dir, name + self.fig_ext),
                          fig_size=[self.width, self.height])
        if self.export_flag:
            viewer.export(mesh_path=os.path.join(self.export_dir, name + self.mesh_ext))

    def viz_segmentation(self, data_group, data_name):
        suffix = '_segmentation'
        if self.gt:
            segmentations = data_group['gt_seg_per_point'][:]
        else:
            raw_segmentations = data_group['pred_seg_per_point'][:]
            segmentations = np.argmax(raw_segmentations, axis=1)
        points_camera = data_group['camcs_per_point'][:]
        viewer = Viewer(points_camera, mask=segmentations)
        self.render_options(viewer, data_name + suffix)
        del viewer

    @staticmethod
    def get_prediction_vertices(pred_segmentation, pred_coordinates):
        segmentations = np.argmax(pred_segmentation, axis=1)
        x = pred_coordinates[np.arange(pred_coordinates.shape[0]), segmentations]
        y = pred_coordinates[np.arange(pred_coordinates.shape[0]), segmentations + 1]
        z = pred_coordinates[np.arange(pred_coordinates.shape[0]), segmentations + 2]
        return segmentations, np.column_stack((x, y, z))

    def viz_npcs(self, data_group, data_name):
        suffix = '_npcs'
        if self.gt:
            segmentations = data_group['gt_seg_per_point'][:]
            npcs_points = data_group['gt_npcs_per_point'][:]
        else:
            raw_segmentations = data_group['pred_seg_per_point'][:]
            raw_npcs_points = data_group['pred_npcs_per_point'][:]
            segmentations, npcs_points = ANCSHVisualizer.get_prediction_vertices(raw_segmentations, raw_npcs_points)
        viewer = Viewer(npcs_points, mask=segmentations)
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_naocs(self, data_group, data_name):
        suffix = '_naocs'
        if self.gt:
            segmentations = data_group['gt_seg_per_point'][:]
            naocs_points = data_group['gt_naocs_per_point'][:]
        else:
            raw_segmentations = data_group['pred_seg_per_point'][:]
            raw_naocs_points = data_group['pred_naocs_per_point'][:]
            segmentations, naocs_points = ANCSHVisualizer.get_prediction_vertices(raw_segmentations, raw_naocs_points)
        viewer = Viewer(naocs_points, mask=segmentations)
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_joint_association(self, data_group, data_name):
        suffix = '_joint_association'
        if self.gt:
            joint_associations = data_group['gt_joint_cls_per_point'][:]
            joint_axes = data_group['gt_axis_per_point'][:]
            naocs_points = data_group['gt_naocs_per_point'][:]
        else:
            raw_joint_associations = data_group['pred_joint_cls_per_point'][:]
            joint_axes = data_group['pred_axis_per_point'][:]
            raw_naocs_points = data_group['pred_naocs_per_point'][:]
            joint_associations, naocs_points = ANCSHVisualizer.get_prediction_vertices(raw_joint_associations,
                                                                                       raw_naocs_points)
        colors = Viewer.colors_from_mask(joint_associations, empty_first=True)
        viewer = Viewer(naocs_points, colors=colors)
        arrow_sample_indices = joint_associations != 0
        arrow_sample_indices[::2] = False
        viewer.add_arrows(naocs_points[arrow_sample_indices], joint_axes[arrow_sample_indices],
                          color=[0, 0, 0, 0.6], radius=0.002, length=0.02)
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_point2joint_offset(self, data_group, data_name):
        suffix = '_point2joint_offset'
        if self.gt:
            point_heatmaps = data_group['gt_heatmap_per_point'][:]
            unit_vectors = data_group['gt_unitvec_per_point'][:]
            naocs_points = data_group['gt_naocs_per_point'][:]
            invalid_heatmap_mask = point_heatmaps == 0
        else:
            raw_joint_associations = data_group['pred_joint_cls_per_point'][:]
            point_heatmaps = data_group['pred_heatmap_per_point'][:].flatten()
            unit_vectors = data_group['pred_unitvec_per_point'][:]
            raw_naocs_points = data_group['pred_naocs_per_point'][:]
            joint_associations, naocs_points = ANCSHVisualizer.get_prediction_vertices(raw_joint_associations,
                                                                                       raw_naocs_points)

            invalid_heatmap_mask = joint_associations == 0
        max_val = np.amax(point_heatmaps)
        cmap = cm.get_cmap('jet')
        colors = cmap(point_heatmaps / max_val)
        colors[invalid_heatmap_mask] = np.array([0.5, 0.5, 0.5, 0.5])
        viewer = Viewer(naocs_points, colors=colors)
        arrow_sample_indices = ~invalid_heatmap_mask
        arrow_sample_indices[::2] = False
        arrow_length = (1 - point_heatmaps) * 0.2 + 10e-8
        viewer.add_trimesh_arrows(naocs_points[arrow_sample_indices], unit_vectors[arrow_sample_indices],
                          color=[0, 0, 0, 0.6], radius=0.002, length=arrow_length)
        self.render_options(viewer, data_name + suffix)
        del viewer

    def render(self, show=False, export=False):
        self.show_flag = show
        if self.show_flag:
            self.render_flag = False
        else:
            self.render_flag = True
        self.export_flag = export

        self.parse_items()
        for i, item_name in enumerate(self.ancsh_items):
            npcs_data_group = self.npcs_data[item_name]
            ancsh_data_group = self.ancsh_data[item_name]
            print(ancsh_data_group.keys())
            self.viz_segmentation(npcs_data_group, item_name)
            self.viz_npcs(npcs_data_group, item_name)
            self.viz_naocs(ancsh_data_group, item_name)
            self.viz_joint_association(ancsh_data_group, item_name)
            self.viz_point2joint_offset(ancsh_data_group, item_name)


@hydra.main(config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig):
    prediction = True
    ancsh_path = f'/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/{NetworkType.ANCSH.value}_pred_gt.h5'
    npcs_path = f'/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/{NetworkType.NPCS.value}_pred_gt.h5'
    if prediction:
        export_dir = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/visualization/pred'
    else:
        export_dir = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/visualization/gt'
    ancsh_input_h5 = h5py.File(ancsh_path, 'r')
    npcs_input_h5 = h5py.File(npcs_path, 'r')

    visualizer = ANCSHVisualizer(npcs_input_h5, ancsh_input_h5, export=export_dir, gt=not prediction)
    visualizer.render(show=True)


if __name__ == "__main__":
    main()
