import os
import re
import trimesh
import numpy as np
import pandas as pd
from enum import Enum
from urdfpy import URDF, JointLimit

from tools.utils import io
from tools.visualization import Viewer

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
    MOTIONNET = 3


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


def get_files_in_format(folder_path, format, ext=None, one=False):
    files = io.alphanum_ordered_file_list(folder_path)
    valid_files = []
    for f in files:
        pattern = re.compile(format)
        filename, extension = os.path.splitext(f)
        if pattern.fullmatch(filename):
            if ext is None or ext == extension:
                if one:
                    return f
                valid_files.append(f)
    return valid_files


def get_one_door_articulations(root_path, filename):
    data = io.read_json(os.path.join(root_path, filename))
    motions = data['motions']
    if len(motions) > 1:
        return None
    if motions[0]['label'] == 'door' and motions[0]['type'] == 'rotation':
        return os.path.splitext(filename)[0]


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
        if io.file_exist(self.cfg.settings.object_ids_path, '.csv'):
            setting_object_ids = pd.read_csv(self.cfg.settings.object_ids_path, dtype=str)['objectId'].to_list()
        else:
            setting_object_ids = self.cfg.settings.object_ids
        selected_object_ids = self.data_info['objectId'].isin(setting_object_ids) \
            if len(setting_object_ids) > 0 else self.data_info['objectId'].astype(bool)
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
            if self.dataset_name == DatasetName.SAPIEN or self.dataset_name == DatasetName.SHAPE2MOTION:
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
            elif self.dataset_name == DatasetName.MOTIONNET:
                for object_id in object_ids:
                    object_id_path = os.path.join(object_cat_path, object_id)
                    depth_dir = os.path.join(object_id_path, self.stage1_input.render.depth_folder)
                    mask_dir = os.path.join(object_id_path, self.stage1_input.render.mask_folder)
                    depth_frames = get_files_in_format(depth_dir, '\d+(\-\d+)+_d', self.stage1_input.render.depth_ext)
                    mask_frames = get_files_in_format(mask_dir, '\d+(\-\d+)+_\d+', self.stage1_input.render.mask_ext)
                    metadata_dir = os.path.join(object_id_path, self.stage1_input.render.metadata_folder)
                    metadata_files = get_files_in_format(metadata_dir, '\d+(\-\d+)+',
                                                         self.stage1_input.render.metadata_ext)
                    articulation_ids = []
                    for filename in metadata_files:
                        articulation_id = str(0)
                        frame_name = os.path.splitext(filename)[0]
                        components = frame_name.split('-')
                        if len(components) > 3:
                            articulation_id = components[-2]
                        articulation_ids.append(articulation_id)
                    num_renders = len(depth_frames)
                    # only support one mask to depth frame mapping yet
                    df_tmp_list = []
                    for i in range(num_renders):
                        name = depth_frames[i].split('_d')[0]
                        masks_name = [mask_frame for mask_frame in mask_frames if name == mask_frame.split('_')[0]]
                            
                        df_tmp = pd.DataFrame(
                            [[object_cat, object_id, articulation_ids[i], depth_frames[i], masks_name,
                            metadata_files[i]]],
                            columns=['objectCat', 'objectId', 'articulationId',
                                    'depthFrame', 'maskFrame', 'metadata'])
                        df_tmp_list.append(df_tmp)
                    df_row = pd.concat(df_tmp_list, ignore_index=True)
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
    def __init__(self, urdf_file_path=None, meta_file_path=None, defined_rest_state=None):
        self.urdf_file_path = urdf_file_path
        self.metadata = io.read_json(meta_file_path) if meta_file_path is not None else None
        self.urdf_data = None
        if self.urdf_file_path:
            self.load(self.urdf_file_path)
        self.debug = False
        self.defined_rest_state = defined_rest_state

    def set_debug(self, debug=True):
        self.debug = debug

    def load(self, urdf_file_path):
        self.urdf_file_path = urdf_file_path
        self.urdf_data = URDF.load(self.urdf_file_path)

    def parse_urdf(self):
        assert self.urdf_data, "URDF data is empty!"

        config = None
        if self.metadata is not None and self.defined_rest_state is not None and not self.defined_rest_state.empty:
            motion = self.metadata['motions'][0]
            articulate_joint_name = 'joint_' + motion['partId']
            config = {articulate_joint_name: -float(self.defined_rest_state['restState'])}

        link_infos = []
        link_idx = 0
        link_meshes = []
        for link in self.urdf_data.links:
            fk_link, link_abs_pose = list(self.urdf_data.link_fk(cfg=config, links=[link]).items())[0]
            link_info = {'name': link.name}
            fk_visual = self.urdf_data.visual_trimesh_fk(cfg=config, links=[link])
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
                color = Viewer.rgba_by_index(link_idx)
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
        # temporary for one part only
        if self.metadata is not None:
            motion = self.metadata['motions'][0]
            part_link_name = 'link_' + motion['partId']
            base_link_name = ''
            for i, joint_info in enumerate(joint_infos):
                if joint_info.get('type') == 'fixed':
                    base_link_name = joint_info['child']
            part_children = [link_infos[i]['part_index'] for i, joint_info in enumerate(joint_infos) if
                             (joint_info.get('parent') == part_link_name or joint_info.get('child') == part_link_name)]
            base_children = [link_infos[i]['part_index'] for i, joint_info in enumerate(joint_infos) if
                             (joint_info.get('parent') == base_link_name or joint_info.get('child') == base_link_name)
                             if joint_info.get('child') != part_link_name]
            part_link_info = None
            part_joint_info = {}
            for i, link_info in enumerate(link_infos):
                if link_info['name'] == part_link_name:
                    part_link_info = link_info
                    part_link_info['part_index'] = 1
                    part_joint_info = joint_infos[i]
            base_link_info = None
            base_joint_info = {}
            for i, link_info in enumerate(link_infos):
                if link_info['name'] == base_link_name:
                    base_link_info = link_info
                    base_link_info['part_index'] = 0
                    base_joint_info = joint_infos[i]

            virtual_link_info = None
            virtual_joint_info = {}
            for i, link_info in enumerate(link_infos):
                if link_info['virtual']:
                    virtual_link_info = link_info
                    virtual_joint_info = joint_infos[i]

            assert len(part_children) + len(base_children) == len(link_infos) - (0 if virtual_link_info is None else 1)

            base_link_mesh = trimesh.base.Trimesh()
            for i in base_children:
                base_link_mesh += link_meshes[i]
            part_link_mesh = trimesh.base.Trimesh()
            for i in part_children:
                part_link_mesh += link_meshes[i]

            link_infos = [base_link_info, part_link_info] if virtual_link_info is None \
                else [virtual_link_info, base_link_info, part_link_info]
            joint_infos = [base_joint_info, part_joint_info] if virtual_link_info is None \
                else [virtual_joint_info, base_joint_info, part_joint_info]
            link_meshes = [base_link_mesh, part_link_mesh]

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
