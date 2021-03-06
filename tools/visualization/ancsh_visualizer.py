import os
import logging
import h5py
import numpy as np
import trimesh.base
from matplotlib import cm
from progress.bar import Bar

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
        self.items = []
        self.show_flag = False
        self.render_flag = False
        self.export_flag = False
        self.network_type = network_type
        self.sampling = sampling
        self.arrow_sampling = 4

        self.width = 1024
        self.height = 768
        self.point_size = 10
        self.fig_ext = '.jpg'
        self.mesh_ext = '.ply'

        self.draw_offset_arrows = True
        self.additional_mesh = None

    def parse_items(self):
        visit_groups = lambda name, node: self.items.append(name) if isinstance(node, h5py.Group) else None
        self.data.visititems(visit_groups)

    def add_trimesh(self, mesh: trimesh.base.Trimesh):
        self.additional_mesh = mesh

    def render_options(self, viewer, name, suffix=''):
        if self.additional_mesh is not None:
            viewer.add_trimesh(self.additional_mesh)
        filename = name + suffix + '_' + self.network_type.value
        folder_names = name.split('_')
        viz_output_dir = os.path.join(self.export_dir, folder_names[0], folder_names[1], folder_names[2])
        viewer.point_size = self.point_size
        if self.show_flag:
            viewer.show(window_size=[self.width, self.height], window_name=name)
        if self.render_flag and self.export_dir:
            viewer.render(fig_path=os.path.join(viz_output_dir, filename + self.fig_ext),
                          fig_size=[self.width, self.height])
        if self.export_flag and self.export_dir:
            viewer.export(mesh_path=os.path.join(viz_output_dir, filename + self.mesh_ext))

    def viz_segmentation(self, data_group, data_name):
        suffix = '_segmentation'
        segmentations = data_group[f'{self.prefix}seg_per_point'][:][::self.sampling]
        points_camera = data_group['camcs_per_point'][:][::self.sampling]
        viewer = Viewer(points_camera, mask=segmentations)
        self.render_options(viewer, data_name, suffix)
        del viewer

    def viz_npcs(self, data_group, data_name):
        suffix = '_npcs'
        segmentations = data_group[f'{self.prefix}seg_per_point'][:][::self.sampling]
        npcs_points = data_group[f'{self.prefix}npcs_per_point'][:][::self.sampling]

        viewer = Viewer(npcs_points, mask=segmentations)
        self.render_options(viewer, data_name, suffix)
        del viewer

    def viz_naocs(self, data_group, data_name):
        suffix = '_naocs'
        segmentations = data_group[f'{self.prefix}seg_per_point'][:][::self.sampling]
        naocs_points = data_group[f'{self.prefix}naocs_per_point'][:][::self.sampling]
        viewer = Viewer(naocs_points, mask=segmentations)
        self.render_options(viewer, data_name, suffix)
        del viewer

    def viz_joint_association(self, data_group, data_name):
        suffix = '_joint_association'
        joint_associations = data_group[f'{self.prefix}joint_cls_per_point'][:][::self.sampling]
        joint_axes = data_group[f'{self.prefix}axis_per_point'][:][::self.sampling]
        naocs_points = data_group[f'{self.prefix}naocs_per_point'][:][::self.sampling]
        colors = Viewer.colors_from_mask(joint_associations, empty_first=True)
        viewer = Viewer(naocs_points, colors=colors)
        arrow_sample_indices = joint_associations != 0
        viewer.add_arrows(naocs_points[arrow_sample_indices][::self.arrow_sampling],
                          joint_axes[arrow_sample_indices][::self.arrow_sampling],
                          color=[0, 0, 0, 0.6], radius=0.002, length=0.04)
        self.render_options(viewer, data_name, suffix)
        del viewer

    def viz_point2joint_offset(self, data_group, data_name):
        suffix = '_point2joint_offset'
        joint_associations = data_group[f'{self.prefix}joint_cls_per_point'][:][::self.sampling]
        point_heatmaps = data_group[f'{self.prefix}heatmap_per_point'][:][::self.sampling]
        unit_vectors = data_group[f'{self.prefix}unitvec_per_point'][:][::self.sampling]
        naocs_points = data_group[f'{self.prefix}naocs_per_point'][:][::self.sampling]

        invalid_heatmap_mask = joint_associations == 0
        max_val = np.amax(point_heatmaps)
        cmap = cm.get_cmap('jet')
        colors = cmap(point_heatmaps / max_val)
        colors[invalid_heatmap_mask] = np.array([0.5, 0.5, 0.5, 0.5])
        viewer = Viewer(naocs_points, colors=colors)
        arrow_sample_indices = ~invalid_heatmap_mask
        arrow_length = (1 - point_heatmaps) * 0.2 + 10e-8
        if self.draw_offset_arrows:
            viewer.add_trimesh_arrows(naocs_points[arrow_sample_indices][::self.arrow_sampling],
                                      unit_vectors[arrow_sample_indices][::self.arrow_sampling],
                                      color=[0, 0, 0, 0.6], radius=0.002,
                                      length=arrow_length[arrow_sample_indices][::self.arrow_sampling])
        else:
            viewer.add_arrows(naocs_points[arrow_sample_indices][::self.arrow_sampling],
                              unit_vectors[arrow_sample_indices][::self.arrow_sampling],
                              color=[0, 0, 0, 0.6], radius=0.002, length=0.04)
            self.render_options(viewer, data_name, suffix)
        self.render_options(viewer, data_name, suffix)
        del viewer

    def render(self, show=False, export=None, export_mesh=False):
        self.show_flag = show
        if self.show_flag:
            self.render_flag = False
        else:
            self.render_flag = True
        self.export_flag = export_mesh
        self.export_dir = export

        self.parse_items()
        bar = Bar(f'Rendering {len(self.items)} instances', max=len(self.items))
        for i, item_name in enumerate(self.items):
            data_group = self.data[item_name]
            log.debug(f'Render {item_name}')
            log.debug(data_group.keys())
            log.debug(data_group.attrs)
            self.viz_segmentation(data_group, item_name)
            self.viz_npcs(data_group, item_name)
            if self.network_type == NetworkType.ANCSH:
                self.viz_naocs(data_group, item_name)
                self.viz_joint_association(data_group, item_name)
                self.viz_point2joint_offset(data_group, item_name)
            bar.next()
        bar.finish()


class OptimizerVisualizer:
    def __init__(self, data: h5py.File):
        self.data = data
        self.export_dir = None
        self.items = []
        self.show_flag = False
        self.render_flag = False
        self.export_flag = False

        self.width = 1024
        self.height = 768
        self.fig_ext = '.jpg'
        self.mesh_ext = '.ply'

    def parse_items(self):
        visit_groups = lambda name, node: self.items.append(name) if isinstance(node, h5py.Group) else None
        self.data.visititems(visit_groups)

    def render_options(self, viewer, name, suffix):
        filename = name + suffix + '_optimization'
        folder_names = name.split('_')
        viz_output_dir = os.path.join(self.export_dir, folder_names[0], folder_names[1], folder_names[2])
        if self.show_flag:
            viewer.show(window_size=[self.width, self.height], window_name=name)
        if self.render_flag and self.export_dir:
            viewer.render(fig_path=os.path.join(viz_output_dir, filename + self.fig_ext),
                          fig_size=[self.width, self.height])
        if self.export_flag and self.export_dir:
            viewer.export(mesh_path=os.path.join(viz_output_dir, filename + self.mesh_ext))

    def viz_npcs2cam(self, data_group, data_name):
        segmentations = data_group['pred_seg_per_point'][:]
        npcs_points = data_group['pred_npcs_per_point'][:]
        npcs2cam_rt = data_group['pred_npcs2cam_rt'][:]
        npcs2cam_scale = data_group['pred_npcs2cam_scale'][:]
        camera_points = data_group['camcs_per_point'][:]
        npcs2cam_points = np.empty_like(npcs_points)
        for k in range(npcs2cam_rt.shape[0]):
            rt = npcs2cam_rt[k].reshape((4, 4), order='F')
            scale = npcs2cam_scale[k]
            npcs2cam_points_part = npcs_points[segmentations == k] * scale
            npcs2cam_points_part_p4 = np.hstack((npcs2cam_points_part, np.ones((npcs2cam_points_part.shape[0], 1))))
            npcs2cam_points_part = np.dot(npcs2cam_points_part_p4, rt.T)[:, :3]
            npcs2cam_points[segmentations == k] = npcs2cam_points_part

        distance = np.linalg.norm(npcs2cam_points - camera_points, axis=1)
        max_val = np.amax(distance)
        cmap = cm.get_cmap('jet')
        colors = cmap(distance / max_val)
        viewer = Viewer(npcs2cam_points, mask=segmentations)
        self.render_options(viewer, data_name, '_npcs2cam')
        viewer.reset()
        viewer.add_geometry(npcs2cam_points, colors=colors)
        self.render_options(viewer, data_name, '_difference')
        del viewer

    def render(self, show=False, export=None, export_mesh=False):
        self.show_flag = show
        if self.show_flag:
            self.render_flag = False
        else:
            self.render_flag = True
        self.export_flag = export_mesh
        self.export_dir = export

        self.parse_items()
        bar = Bar(f'Rendering {len(self.items)} instances', max=len(self.items))
        for i, item_name in enumerate(self.items):
            data_group = self.data[item_name]
            log.debug(f'Render {item_name}')
            log.debug(data_group.keys())
            log.debug(data_group.attrs)
            self.viz_npcs2cam(data_group, item_name)
            bar.next()
        bar.finish()


@hydra.main(config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig):
    prediction = True
    ancsh_path = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/network/test/ANCSH_2022-01-07_00-33-40/ANCSH_inference_result.h5'
    npcs_path = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/network/test/NPCS_2022-01-07_00-34-24/NPCS_inference_result.h5'
    optimize_path = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/optimization_result.h5'
    if prediction:
        export_dir = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/visualization/pred'
    else:
        export_dir = '/home/sam/Development/Research/ancsh/ANCSH-pytorch/results/visualization/gt'
    ancsh_input_h5 = h5py.File(ancsh_path, 'r')
    npcs_input_h5 = h5py.File(npcs_path, 'r')
    optimize_input_h5 = h5py.File(optimize_path, 'r')

    visualizer = OptimizerVisualizer(optimize_input_h5)
    visualizer.render(show=False, export=export_dir)

    # visualizer = ANCSHVisualizer(npcs_input_h5, NetworkType.NPCS, gt=not prediction)
    # visualizer.render(show=True, export=export_dir)
    # visualizer = ANCSHVisualizer(ancsh_input_h5, NetworkType.ANCSH, gt=not prediction)
    # visualizer.render(show=True, export=export_dir)


if __name__ == "__main__":
    main()
