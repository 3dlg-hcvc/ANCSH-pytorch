import os
import trimesh
import numpy as np
import pandas as pd
from enum import Enum
from matplotlib import cm
from urdfpy import URDF, JointLimit

from tools.utils import io
# from tools.visualization import Viewer

# override attributes to make effort, velocity optional
JointLimit._ATTRIBS = {
    'effort': (float, False),
    'velocity': (float, False),
    'lower': (float, False),
    'upper': (float, False),
}
# set default values
JointLimit.effort = 1.0
JointLimit.velocity = 1000


class DatasetName(Enum):
    SAPIEN = 0
    SHAPE2MOTION = 1
    MULTISCAN = 2


class JointType(Enum):
    prismatic = 0
    revolute = 1
    fixed = -1
    continuous = -1
    floating = -1
    planar = -1

def get_mesh_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    assert isinstance(mesh, trimesh.base.Trimesh)
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]
    center = np.mean(mesh.bounds, axis=0)
    scale = mesh.scale
    mesh_info = {
        'min_bound': min_bound.tolist(),
        'max_bound': max_bound.tolist(),
        'center': center.tolist(),
        'scale': scale
    }
    return mesh_info


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        dataset_name = self.cfg.dataset.name
        self.dataset_name = DatasetName[dataset_name] if isinstance(dataset_name, str) else dataset_name
        self.dataset_dir = self.cfg.paths.preprocess.input_dir
        self.stage1_input = self.cfg.paths.preprocess.stage1.input
        self.render_dir = os.path.join(self.dataset_dir, self.stage1_input.render.folder_name)
        self.motion_dir = os.path.join(self.dataset_dir, self.stage1_input.motion.folder_name)
        self.data_info = pd.DataFrame()

    def parse_input(self):
        render_data_info = self.parse_render_input()
        motion_data_info = self.parse_motion_input()
        self.data_info = render_data_info.merge(motion_data_info, how='inner', on=['objectCat', 'objectId'])

        selected_categories = self.data_info['objectCat'].isin(self.cfg.settings.categories) \
            if len(self.cfg.settings.categories) > 0 else self.data_info['objectCat'].astype(bool)
        selected_object_ids = self.data_info['objectId'].isin(self.cfg.settings.object_ids) \
            if len(self.cfg.settings.object_ids) > 0 else self.data_info['objectId'].astype(bool)
        selected_articulation_ids = self.data_info['articulationId'].isin(self.cfg.settings.articulation_ids) \
            if len(self.cfg.settings.articulation_ids) > 0 else self.data_info['articulationId'].astype(bool)
        self.data_info = self.data_info[selected_categories & selected_object_ids & selected_articulation_ids]
        self.data_info = self.data_info.reset_index(drop=True)

    def parse_render_input(self):
        df_list = []
        object_cats = os.listdir(self.render_dir)
        # object categories
        for object_cat in object_cats:
            object_cat_path = os.path.join(self.render_dir, object_cat)
            object_ids = io.alphanum_ordered_folder_list(object_cat_path)
            # object instance ids
            for object_id in object_ids:
                object_id_path = os.path.join(object_cat_path, object_id)
                articulation_ids = io.alphanum_ordered_folder_list(object_id_path)
                # object with different articulations instance ids
                for articulation_id in articulation_ids:
                    articulation_id_path = os.path.join(object_id_path, articulation_id)
                    depth_dir = os.path.join(articulation_id_path, self.stage1_input.render.depth_folder)
                    depth_frames = io.alphanum_ordered_file_list(depth_dir, ext=self.stage1_input.render.depth_ext)
                    mask_dir = os.path.join(articulation_id_path, self.stage1_input.render.mask_folder)
                    mask_frames = io.alphanum_ordered_file_list(mask_dir, ext=self.stage1_input.render.mask_ext)
                    metadata_file = self.stage1_input.render.metadata_file
                    num_renders = len(depth_frames)
                    df_row = pd.concat([pd.DataFrame(
                        [[object_cat, object_id, articulation_id, depth_frames[i], mask_frames[i],
                          metadata_file]],
                        columns=['objectCat', 'objectId', 'articulationId',
                                 'depthFrame', 'maskFrame', 'metadata']) for i in range(num_renders)],
                        ignore_index=True)
                    df_list.append(df_row)
        return pd.concat(df_list, ignore_index=True)

    def parse_motion_input(self):
        df_list = []
        object_cats = os.listdir(self.motion_dir)
        # object categories
        for object_cat in object_cats:
            object_cat_path = os.path.join(self.motion_dir, object_cat)
            object_ids = io.alphanum_ordered_folder_list(object_cat_path)
            # object instance ids
            for object_id in object_ids:
                object_id_path = os.path.join(object_cat_path, object_id)
                motion_file = self.stage1_input.motion.motion_file
                motion_file_path = os.path.join(object_id_path, motion_file)
                if io.file_exist(motion_file_path):
                    df_row = pd.DataFrame(
                        [[object_cat, object_id, motion_file]],
                        columns=['objectCat', 'objectId', 'motion'])
                    df_list.append(df_row)
        return pd.concat(df_list, ignore_index=True)


class URDFReader:
    def __init__(self, urdf_file_path=None):
        self.urdf_file_path = urdf_file_path
        self.urdf_data = None
        if self.urdf_file_path:
            self.load(self.urdf_file_path)
        self.debug = False

    def set_debug(self, debug=True):
        self.debug = debug

    def load(self, urdf_file_path):
        self.urdf_file_path = urdf_file_path
        self.urdf_data = URDF.load(self.urdf_file_path)

    def parse_urdf(self):
        assert self.urdf_data, "URDF data is empty!"

        link_infos = []
        link_idx = 0
        link_meshes = []
        for link in self.urdf_data.links:
            fk_link, link_abs_pose = list(self.urdf_data.link_fk(links=[link]).items())[0]
            link_info = {'name': link.name}
            fk_visual = self.urdf_data.visual_trimesh_fk(links=[link])
            is_virtual = not bool(fk_visual)
            link_info['virtual'] = is_virtual
            link_info['abs_pose'] = link_abs_pose.flatten(order='F').tolist()
            if not is_virtual:
                link_mesh = trimesh.base.Trimesh()
                for mesh, mesh_abs_pose in fk_visual.items():
                    mesh.apply_transform(mesh_abs_pose)
                    # remove texture visual
                    mesh.visual = trimesh.visual.create_visual()
                    link_mesh += mesh
                # part mesh visualization
                # color = Viewer.rgba_by_index(link_idx)
                color = list(cm.get_cmap('Set1')(link_idx))
                color[-1] = 0.8
                link_mesh.visual.vertex_colors = color
                if self.debug:
                    link_mesh.show()
                link_info['part_index'] = link_idx
                link_meshes.append(link_mesh)
                link_idx += 1
            else:
                link_info['part_index'] = -1
            link_infos.append(link_info)

        joint_infos = []
        for link_info in link_infos:
            joint_info = {}
            for joint in self.urdf_data.joints:
                if joint.child == link_info['name']:
                    joint_info = {
                        'name': joint.name,
                        'type': joint.joint_type,
                        'parent': joint.parent,
                        'child': joint.child,
                        'axis': joint.axis.tolist(),
                        'pose2link': joint.origin.flatten(order='F').tolist()
                    }
                    break
            joint_infos.append(joint_info)
        return link_infos, joint_infos, link_meshes

    def export(self, result_data_path, rest_state_data_filename='rest_state.json',
               rest_state_mesh_filename='rest_state.ply'):
        io.ensure_dir_exists(result_data_path)

        link_infos, joint_infos, link_meshes = self.parse_urdf()
        object_mesh = trimesh.base.Trimesh()
        for link_info in link_infos:
            if not link_info['virtual']:
                part_mesh = link_meshes[link_info['part_index']]
                object_mesh += part_mesh
                part_mesh_filename = f'{link_info["name"]}_{rest_state_mesh_filename}'
                part_mesh.export(os.path.join(result_data_path, part_mesh_filename))
        object_mesh.export(os.path.join(result_data_path, rest_state_mesh_filename))

        rest_state_data = {
            'links': link_infos,
            'joints': joint_infos
        }
        io.write_json(rest_state_data, os.path.join(result_data_path, rest_state_data_filename))
