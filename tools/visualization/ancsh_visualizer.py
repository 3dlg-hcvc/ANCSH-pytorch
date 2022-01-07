import os
import logging
import h5py
import numpy as np
import trimesh.base
from matplotlib import cm

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from tools.utils import io
from tools.visualization.viewer import Viewer
from ANCSH_lib.utils import NetworkType, get_prediction_vertices

log = logging.getLogger('ANCSH visualizer')


class ANCSHVisualizer:
    def __init__(self, data: h5py.File, network_type, gt=False, sampling=1):
        self.data = data
        self.gt = gt
        self.prefix = 'gt_' if gt else 'pred_'
        self.export_dir = None
        if self.export_dir is not None:
            io.ensure_dir_exists(self.export_dir)
        self.items = []
        self.show_flag = False
        self.render_flag = False
        self.export_flag = False
        self.network_type = network_type
        self.sampling = sampling

        self.width = 1024
        self.height = 768
        self.fig_ext = '.png'
        self.mesh_ext = '.ply'

        self.additional_mesh = None

    def parse_items(self):
        visit_groups = lambda name, node: self.items.append(name) if isinstance(node, h5py.Group) else None
        self.data.visititems(visit_groups)

    def add_trimesh(self, mesh: trimesh.base.Trimesh):
        self.additional_mesh = mesh

    def render_options(self, viewer, name):
        if self.additional_mesh is not None:
            viewer.add_trimesh(self.additional_mesh)
        if self.show_flag:
            viewer.show(window_size=[self.width, self.height], window_name=name)
        if self.render_flag and self.export_dir:
            io.ensure_dir_exists(self.export_dir)
            viewer.render(fig_path=os.path.join(self.export_dir, name + '_' + self.network_type.value + self.fig_ext),
                          fig_size=[self.width, self.height])
        if self.export_flag and self.export_dir:
            io.ensure_dir_exists(self.export_dir)
            viewer.export(mesh_path=os.path.join(self.export_dir, name + '_' + self.network_type.value + self.mesh_ext))

    def viz_segmentation(self, data_group, data_name):
        suffix = '_segmentation'
        segmentations = data_group[f'{self.prefix}seg_per_point'][:]
        points_camera = data_group['camcs_per_point'][:]
        viewer = Viewer(points_camera, mask=segmentations)
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_npcs(self, data_group, data_name):
        suffix = '_npcs'
        segmentations = data_group[f'{self.prefix}seg_per_point'][:]
        npcs_points = data_group[f'{self.prefix}npcs_per_point'][:]

        viewer = Viewer(npcs_points[::self.sampling], mask=segmentations[::self.sampling])
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_naocs(self, data_group, data_name):
        suffix = '_naocs'
        segmentations = data_group[f'{self.prefix}seg_per_point'][:]
        naocs_points = data_group[f'{self.prefix}naocs_per_point'][:]
        viewer = Viewer(naocs_points[::self.sampling], mask=segmentations[::self.sampling])
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_joint_association(self, data_group, data_name):
        suffix = '_joint_association'
        joint_associations = data_group[f'{self.prefix}joint_cls_per_point'][:]
        joint_axes = data_group[f'{self.prefix}axis_per_point'][:]
        naocs_points = data_group[f'{self.prefix}naocs_per_point'][:]
        colors = Viewer.colors_from_mask(joint_associations, empty_first=True)
        viewer = Viewer(naocs_points, colors=colors)
        arrow_sample_indices = joint_associations != 0
        arrow_sample_indices[::2] = False
        viewer.add_arrows(naocs_points[arrow_sample_indices][::self.sampling],
                          joint_axes[arrow_sample_indices][::self.sampling],
                          color=[0, 0, 0, 0.6], radius=0.002, length=0.04)
        self.render_options(viewer, data_name + suffix)
        del viewer

    def viz_point2joint_offset(self, data_group, data_name):
        suffix = '_point2joint_offset'
        joint_associations = data_group[f'{self.prefix}joint_cls_per_point'][:]
        point_heatmaps = data_group[f'{self.prefix}heatmap_per_point'][:]
        unit_vectors = data_group[f'{self.prefix}unitvec_per_point'][:]
        naocs_points = data_group[f'{self.prefix}naocs_per_point'][:]

        invalid_heatmap_mask = joint_associations == 0
        max_val = np.amax(point_heatmaps)
        cmap = cm.get_cmap('jet')
        colors = cmap(point_heatmaps / max_val)
        colors[invalid_heatmap_mask] = np.array([0.5, 0.5, 0.5, 0.5])
        viewer = Viewer(naocs_points, colors=colors)
        arrow_sample_indices = ~invalid_heatmap_mask
        arrow_sample_indices[::2] = False
        arrow_length = (1 - point_heatmaps) * 0.2 + 10e-8
        viewer.add_trimesh_arrows(naocs_points[arrow_sample_indices][::self.sampling],
                                  unit_vectors[arrow_sample_indices][::self.sampling],
                                  color=[0, 0, 0, 0.6], radius=0.002,
                                  length=arrow_length[arrow_sample_indices][::self.sampling])
        self.render_options(viewer, data_name + suffix)
        del viewer

    def render(self, show=False, export=None):
        self.show_flag = show
        if self.show_flag:
            self.render_flag = False
        else:
            self.render_flag = True
        self.export_flag = False if export is None else True
        self.export_dir = export

        self.parse_items()
        log.info(f'Rendering instances {self.items}')
        for i, item_name in enumerate(self.items):
            data_group = self.data[item_name]
            print(data_group.keys())
            self.viz_segmentation(data_group, item_name)
            self.viz_npcs(data_group, item_name)
            if self.network_type == NetworkType.ANCSH:
                self.viz_naocs(data_group, item_name)
                self.viz_joint_association(data_group, item_name)
                self.viz_point2joint_offset(data_group, item_name)


@hydra.main(config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig):
    prediction = False
    ancsh_path = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/network/test/ANCSH_2022-01-05_23-36-49/ANCSH_inference_result.h5'
    npcs_path = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/network/test/NPCS_2022-01-05_23-30-33/NPCS_inference_result.h5'
    if prediction:
        export_dir = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/visualization/pred'
    else:
        export_dir = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/visualization/gt'
    ancsh_input_h5 = h5py.File(ancsh_path, 'r')
    # npcs_input_h5 = h5py.File(npcs_path, 'r')

    visualizer = ANCSHVisualizer(ancsh_input_h5, NetworkType.ANCSH, gt=not prediction)
    # visualizer = ANCSHVisualizer(npcs_input_h5, NetworkType.NPCS, gt=not prediction)
    visualizer.render(show=True, export=export_dir)


if __name__ == "__main__":
    main()
