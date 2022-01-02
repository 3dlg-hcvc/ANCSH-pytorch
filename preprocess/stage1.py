import os
import h5py
import pandas as pd
import trimesh
import yaml
import logging
import numpy as np

from PIL import Image
from yaml import CLoader as Loader, CDumper as Dumper
from scipy.spatial.transform import Rotation as R

from tools.utils import io
import utils
from utils import DataLoader, URDFReader, DatasetName, JointType

log = logging.getLogger('proc_stage1')


class ProcStage1:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_loader = DataLoader(cfg)
        self.data_loader.parse_input()
        self.input_cfg = self.cfg.paths.preprocess.stage1.input
        self.tmp_output = self.cfg.paths.preprocess.stage1.tmp_output
        self.output_cfg = self.cfg.paths.preprocess.stage1.output
        self.height = self.cfg.dataset.param.height
        self.width = self.cfg.dataset.param.width
        self.debug = self.cfg.debug

    def preprocess_motion_data(self, motion_data_df):
        for index, motion_data in motion_data_df.iterrows():
            motion_file_path = os.path.join(self.data_loader.motion_dir, motion_data['objectCat'],
                                            motion_data['objectId'], motion_data['motion'])
            assert io.file_exist(motion_file_path), f'Can not found Motion file {motion_file_path}!'
            if DatasetName[self.cfg.dataset.name] == DatasetName.SAPIEN:
                urdf_reader = URDFReader(motion_file_path)
                tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, motion_data['objectCat'],
                                            motion_data['objectId'], self.tmp_output.folder_name)
                urdf_reader.export(
                    result_data_path=tmp_data_dir,
                    rest_state_data_filename=self.tmp_output.rest_state_data,
                    rest_state_mesh_filename=self.tmp_output.rest_state_mesh
                )

    def get_metadata(self, metadata_path, frame_index):
        metadata = {}
        if DatasetName[self.cfg.dataset.name] == DatasetName.SAPIEN:
            with open(metadata_path, "r") as meta_file:
                metadata_all = yaml.load(meta_file, Loader=yaml.Loader)
            frame_metadata = metadata_all[f'frame_{frame_index}']
            metadata = {
                'projMat': np.reshape(frame_metadata['projMat'], (4, 4), order='F'),
                'viewMat': np.reshape(frame_metadata['viewMat'], (4, 4), order='F'),
                'linkAbsPoses': []
            }
            num_links = len(frame_metadata['obj'])
            for link_idx in range(num_links):
                position = frame_metadata['obj'][link_idx][4]
                # x,y,z,w
                quaternion = frame_metadata['obj'][link_idx][5]
                orientation = R.from_quat(quaternion).as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = orientation
                pose[:3, 3] = position
                metadata['linkAbsPoses'].append(pose)

        return metadata

    def process(self):
        input_data = self.data_loader.data_info
        io.ensure_dir_exists(self.cfg.paths.preprocess.tmp_dir)
        input_data.to_csv(os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.input_files))

        motion_data_df = input_data.drop_duplicates(subset=['objectCat', 'objectId', 'motion'])
        self.preprocess_motion_data(motion_data_df)
        io.ensure_dir_exists(self.cfg.paths.preprocess.output_dir)
        h5file = h5py.File(os.path.join(self.cfg.paths.preprocess.output_dir, self.output_cfg.pcd_data), 'w')
        for index, input_each in input_data.iterrows():
            depth_frame_path = os.path.join(self.data_loader.render_dir, input_each['objectCat'],
                                            input_each['objectId'], input_each['articulationId'],
                                            self.input_cfg.render.depth_folder, input_each['depthFrame'])
            mask_frame_path = os.path.join(self.data_loader.render_dir, input_each['objectCat'],
                                           input_each['objectId'], input_each['articulationId'],
                                           self.input_cfg.render.mask_folder, input_each['maskFrame'])
            metadata_path = os.path.join(self.data_loader.render_dir, input_each['objectCat'],
                                         input_each['objectId'], input_each['articulationId'],
                                         input_each['metadata'])
            tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, input_each['objectCat'],
                                        input_each['objectId'], self.tmp_output.folder_name)
            rest_state_data_path = os.path.join(tmp_data_dir, self.tmp_output.rest_state_data)

            frame_index = int(input_each['depthFrame'].split(self.input_cfg.render.depth_ext)[0])
            # float32 depth buffer, range from 0 to 1
            depth_data = np.array(h5py.File(depth_frame_path, "r")["data"])
            # uint8 mask, invalid value is 255
            mask_frame = np.asarray(Image.open(mask_frame_path))
            assert depth_data.size == mask_frame.size
            metadata = self.get_metadata(metadata_path, frame_index)
            x_range = np.linspace(-1, 1, self.width)
            y_range = np.linspace(1, -1, self.height)
            x, y = np.meshgrid(x_range, y_range)
            x = x.flatten()
            y = y.flatten()
            z = 2.0 * depth_data - 1.0
            # shape nx4
            points_tmp = np.column_stack((x, y, z, np.ones(self.height * self.width)))
            mask_tmp = mask_frame.flatten()
            # points in clip space
            points_clip = points_tmp[mask_tmp < 255]
            link_mask = mask_tmp[mask_tmp < 255]
            # check if unique value in mask match num parts
            assert points_clip.shape[0] == link_mask.shape[0]
            proj_mat = metadata['projMat']
            view_mat = metadata['viewMat']
            # transform points from clip space to camera space
            # shape 4xn
            points_camera = np.dot(np.linalg.inv(proj_mat), points_clip.transpose())
            # homogeneous normalization
            points_camera = points_camera / points_camera[-1, :]
            # shape 4xn
            points_world = np.dot(np.linalg.inv(view_mat), points_camera)

            # transform links to rest state
            rest_data_data = io.read_json(rest_state_data_path)
            points_rest_state = np.empty_like(points_world)
            parts_camera2rest_state = []
            for link_idx, link in enumerate(rest_data_data['links']):
                if link['virtual']:
                    continue
                link_points_world = points_world[:, link_mask == link_idx]
                # virtual link link_index is -1
                current_part_pose = metadata['linkAbsPoses'][link['part_index']]
                rest_state_pose = np.reshape(link['abs_pose'], (4, 4), order='F')
                transform2rest_state = np.dot(rest_state_pose, np.linalg.inv(current_part_pose))
                link_points_rest_state = np.dot(transform2rest_state, link_points_world)
                points_rest_state[:, link_mask == link_idx] = link_points_rest_state
                # points in camera space to rest state
                camera2rest_state = np.dot(transform2rest_state, np.linalg.inv(view_mat))
                # shape num parts x 16
                parts_camera2rest_state.append(camera2rest_state.flatten('F'))
            parts_camera2rest_state = np.asarray(parts_camera2rest_state)
            # shape nx3
            points_camera_p3 = points_camera.transpose()[:, :3]
            points_world_p3 = points_world.transpose()[:, :3]
            points_rest_state_p3 = points_rest_state.transpose()[:, :3]

            if self.debug:
                window_name_prefix = f'{input_each["objectCat"]}/{input_each["objectId"]}/' \
                                     + f'{input_each["articulationId"]}/{frame_index} '
                utils.visualize_point_cloud(points_camera_p3, link_mask,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_camera %
                                                                (int(input_each["articulationId"]), frame_index)),
                                            window_name=window_name_prefix + 'Camera Space', show=False)
                utils.visualize_point_cloud(points_world_p3, link_mask,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_world %
                                                                (int(input_each["articulationId"]), frame_index)),
                                            window_name=window_name_prefix + 'World Space', show=False)
                utils.visualize_point_cloud(points_rest_state_p3, link_mask,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_rest_state %
                                                                (int(input_each["articulationId"]), frame_index)),
                                            window_name=window_name_prefix + 'Rest State', show=False)

            camera2base_matrix = np.linalg.inv(view_mat).flatten('F')
            h5cat = h5file.require_group(input_each["objectCat"])
            h5obj = h5cat.require_group(input_each["objectId"])
            h5art = h5obj.require_group(input_each["articulationId"])
            h5frame = h5art.require_group(str(frame_index))
            h5frame.create_dataset("mask", shape=link_mask.shape, data=link_mask, compression="gzip")
            h5frame.create_dataset("points_camera", shape=points_camera_p3.shape, data=points_camera_p3,
                                   compression="gzip")
            h5frame.create_dataset("points_rest_state", shape=points_rest_state_p3.shape, data=points_rest_state_p3,
                                   compression="gzip")
            h5frame.create_dataset("parts_transformation", shape=parts_camera2rest_state.shape,
                                   data=parts_camera2rest_state, compression="gzip")
            h5frame.create_dataset("base_transformation", shape=camera2base_matrix.shape,
                                   data=camera2base_matrix, compression="gzip")
        h5file.close()


class ProcStage2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_cfg = self.cfg.paths.preprocess.stage2.input
        self.input_h5 = h5py.File(os.path.join(self.cfg.paths.preprocess.output_dir, self.input_cfg.pcd_data), 'r')
        self.output_dir = self.cfg.paths.preprocess.output_dir
        self.stag1_tmp_output = self.cfg.paths.preprocess.stage1.tmp_output
        self.tmp_output = self.cfg.paths.preprocess.stage2.tmp_output
        self.split_info = None
        self.debug = self.cfg.debug
        self.show = self.cfg.show
        stage1_input = self.cfg.paths.preprocess.stage1.input
        self.part_orders = io.read_json(os.path.join(self.cfg.paths.preprocess.input_dir, stage1_input.part_order_file))
        self.heatmap_threshold = self.cfg.params.joint_association_threshold
        self.epsilon = 10e-8

    def split_data(self, train_percent=.6, seed=None):
        datasets = []
        visit_leaves = lambda name, node: datasets.append(name) if isinstance(node, h5py.Dataset) else None
        self.input_h5.visititems(visit_leaves)
        df_dataset = pd.DataFrame([name.split('/') for name in datasets],
                                  columns=['objectCat', 'objectId', 'articulationId', 'frameId', 'dataName'])
        df_dataset = df_dataset[['objectCat', 'objectId', 'articulationId', 'frameId']] \
            .drop_duplicates(ignore_index=True)
        # select data in config
        selected_categories = df_dataset['objectCat'].isin(self.cfg.settings.categories) \
            if len(self.cfg.settings.categories) > 0 else df_dataset['objectCat']
        selected_object_ids = df_dataset['objectId'].isin(self.cfg.settings.object_ids) \
            if len(self.cfg.settings.object_ids) > 0 else df_dataset['objectId']
        selected_articulation_ids = df_dataset['articulationId'].isin(self.cfg.settings.articulation_ids) \
            if len(self.cfg.settings.articulation_ids) > 0 else df_dataset['articulationId']
        df_dataset = df_dataset[selected_categories & selected_object_ids & selected_articulation_ids]

        if io.file_exist(self.cfg.paths.preprocess.stage2.input.split_info, ext='.csv'):
            input_split_info = pd.read_csv(self.cfg.paths.preprocess.stage2.input.split_info)
            self.split_info = input_split_info.merge(df_dataset, how='inner',
                                                     on=['objectCat', 'objectId', 'articulationId', 'frameId'])
        else:
            # split to train, val, test
            df_size = len(df_dataset)
            if df_size:
                val_end = train_percent + (1.0 - train_percent) / 2.0
                train, val, test = np.split(df_dataset.sample(frac=1, random_state=seed),
                                            [int(train_percent * df_size), int(val_end * df_size)])

                self.split_info = pd.concat([train, val, test], keys=["train", "val", "test"], names=['set', 'index'])
            else:
                log.error('No data to split!')
                return
        self.split_info.to_csv(os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.split_info))

    def process(self):
        io.ensure_dir_exists(self.output_dir)
        if self.split_info is None or self.split_info.empty:
            log.error('No data to process!')
            return
        train = self.split_info.loc['train']
        self.process_each(train,
                          os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.train_data))
        val = self.split_info.loc['val']
        self.process_each(val,
                          os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.val_data))
        test = self.split_info.loc['test']
        self.process_each(test,
                          os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.test_data))

    def get_joint_params(self, vertices, joints, vertices_class, part_class, num_parts):
        heatmap2joints = -np.ones((vertices.shape[0], num_parts))
        unitvec2joints = np.zeros((vertices.shape[0], num_parts, 3))

        for joint in joints:
            joint_class = joint['class']
            joint_pos = joint['abs_position']
            joint_axis = joint['axis']
            joint_axis = joint_axis / np.linalg.norm(joint_axis)
            joint_axis = joint_axis.reshape((3, 1))
            if joint['type'] == JointType.revolute.value:
                vec1 = vertices - joint_pos
                # project to joint axis
                proj_len = np.dot(vec1, joint_axis)
                np.clip(proj_len, a_min=self.epsilon, a_max=None, out=proj_len)
                proj_vec = proj_len * joint_axis.transpose()
                orthogonal_vec = - vec1 + proj_vec
                heatmap = np.linalg.norm(orthogonal_vec, axis=1).reshape(-1, 1)
                unitvec = orthogonal_vec / heatmap
                heatmap2joints[vertices_class == part_class, joint_class] = heatmap.transpose()
                unitvec2joints[vertices_class == part_class, joint_class] = unitvec
            elif joint['type'] == JointType.prismatic.value:
                if part_class == joint_class:
                    heatmap2joints[vertices_class == part_class, joint_class] = 0
                    unitvec2joints[vertices_class == part_class, joint_class] = joint_axis.transpose()
            else:
                log.error(f'Invalid joint type {joint["axis"]}')

        heatmap2joints = np.where(heatmap2joints >= 0, heatmap2joints, np.inf)
        best_joint_index = heatmap2joints.argmin(axis=1)
        heatmap = heatmap2joints[np.arange(len(heatmap2joints)), best_joint_index]
        unitvec = unitvec2joints[np.arange(len(heatmap2joints)), best_joint_index]
        return heatmap, unitvec

    def process_each(self, data_info, output_path):
        # process object info
        object_df = data_info[['objectCat', 'objectId']].drop_duplicates()
        object_infos = {}
        for index, row in object_df.iterrows():
            stage1_tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, row['objectCat'], row['objectId'],
                                               self.stag1_tmp_output.folder_name)
            rest_state_data_path = os.path.join(stage1_tmp_data_dir, self.stag1_tmp_output.rest_state_data)
            rest_state_data = io.read_json(rest_state_data_path)
            object_mesh_path = os.path.join(stage1_tmp_data_dir, self.stag1_tmp_output.rest_state_mesh)
            object_dict = utils.get_mesh_info(object_mesh_path)
            part_dict = {}
            part_order = self.part_orders[row['objectCat']][row['objectId']]
            for link_index, link in enumerate(rest_state_data['links']):
                if link['virtual']:
                    continue
                part_mesh_path = os.path.join(stage1_tmp_data_dir,
                                              f'{link["name"]}_{self.stag1_tmp_output.rest_state_mesh}')
                part_dict[link_index] = utils.get_mesh_info(part_mesh_path)
                part_dict[link_index]['part_class'] = part_order.index(link['part_index'])
            if row['objectCat'] in object_infos:
                object_infos[row['objectCat']][row['objectId']] = {'object': object_dict, 'part': part_dict}
            else:
                object_infos[row['objectCat']] = {row['objectId']: {'object': object_dict, 'part': part_dict}}

        h5file = h5py.File(output_path, 'w')
        for index, row in data_info.iterrows():
            h5frame = self.input_h5[row['objectCat']][row['objectId']][row['articulationId']][row['frameId']]
            mask = h5frame['mask'][:]
            points_camera = h5frame['points_camera'][:]
            points_rest_state = h5frame['points_rest_state'][:]
            parts_camera2rest_state = h5frame['parts_transformation'][:]
            camera2base = h5frame['base_transformation'][:]

            stage1_tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, row['objectCat'], row['objectId'],
                                               self.stag1_tmp_output.folder_name)
            rest_state_data_path = os.path.join(stage1_tmp_data_dir, self.stag1_tmp_output.rest_state_data)
            rest_state_data = io.read_json(rest_state_data_path)

            part_info = object_infos[row['objectCat']][row['objectId']]['part']
            num_parts = len(part_info)

            # process points related ground truth
            object_info = object_infos[row['objectCat']][row['objectId']]['object']
            # diagonal axis aligned bounding box length to 1
            # (0.5, 0.5, 0.5) centered
            naocs_translation = - object_info['center'] + 0.5 * object_info['scale']
            naocs_scale = 1.0 / object_info['scale']
            naocs = points_rest_state + naocs_translation
            naocs *= naocs_scale

            naocs_transformation = np.reshape(camera2base, (4, 4), order='F')
            naocs_transformation[:3, 3] += naocs_translation
            naocs2cam_transformation = np.linalg.inv(naocs_transformation).flatten('F')
            naocs2cam_scale = 1.0 / naocs_scale

            points_class = np.empty_like(mask)
            npcs = np.empty_like(points_rest_state)
            parts_npcs2cam_transformation = np.empty_like(parts_camera2rest_state)
            parts_npcs2cam_scale = np.empty(num_parts)

            for link_index, link in enumerate(rest_state_data['links']):
                if link['virtual']:
                    continue
                part_points = points_rest_state[mask == link_index]
                center = part_info[link_index]['center']
                # diagonal axis aligned bounding box length to 1
                # (0.5, 0.5, 0.5) centered
                npcs_translation = - center + 0.5 * part_info[link_index]['scale']
                npcs_scale = 1.0 / part_info[link_index]['scale']
                part_points_norm = part_points + npcs_translation
                part_points_norm *= npcs_scale

                npcs[mask == link_index] = part_points_norm
                part_class = part_info[link_index]['part_class']
                points_class[mask == link_index] = part_class
                npcs_transformation = np.reshape(parts_camera2rest_state[link['part_index']], (4, 4), order='F')
                npcs_transformation[:3, 3] += npcs_translation
                npcs2cam_transformation = np.linalg.inv(npcs_transformation)
                parts_npcs2cam_transformation[part_class] = npcs2cam_transformation.flatten('F')
                parts_npcs2cam_scale[part_class] = 1.0 / npcs_scale

            # process joints related ground truth
            link_names = [link['name'] for link in rest_state_data['links']]
            # transform joints to naocs space
            naocs_joints = rest_state_data['joints']
            for i, joint in enumerate(rest_state_data['joints']):
                if not joint:
                    continue
                joint_pose = np.asarray(joint['pose2link']).reshape((4, 4), order='F')
                joint_parent = joint['parent']
                parent_link = rest_state_data['links'][link_names.index(joint_parent)]
                parent_link_abs_pose = np.asarray(parent_link['abs_pose']).reshape((4, 4), order='F')
                joint_abs_pose = np.dot(parent_link_abs_pose, joint_pose)
                joint_pos = joint_abs_pose[:3, 3]
                naocs_joint_pos = joint_pos + naocs_translation
                naocs_joint_pos *= naocs_scale
                joint_axis = np.dot(joint_abs_pose[:3, :3], joint['axis'])
                joint_axis = joint_axis / np.linalg.norm(joint_axis)

                naocs_joints[i]['abs_position'] = naocs_joint_pos
                naocs_joints[i]['axis'] = joint_axis

                joint_child = joint['child']
                child_link_class = part_info[link_names.index(joint_child)]['part_class']
                joint_class = child_link_class
                naocs_joints[i]['class'] = joint_class

                joint_type = JointType[joint['type']].value
                naocs_joints[i]['type'] = joint_type

                if self.show:
                    naocs_arrow = utils.draw_arrow(naocs_joint_pos, joint_axis,
                                                   color=utils.rgba_by_index(joint_class))
                    pcd = trimesh.points.PointCloud(vertices=naocs[points_class == child_link_class])
                    scene = trimesh.scene.Scene([pcd, naocs_arrow])
                    scene.show()

            tmp_heatmap = np.empty((num_parts, naocs.shape[0]))
            tmp_unitvec = np.empty((num_parts, naocs.shape[0], 3))
            for i, link in enumerate(rest_state_data['links']):
                if link['virtual']:
                    continue
                # find parent joints
                parent_joints = [joint for joint in naocs_joints
                                 if joint if joint['type'] >= 0 if joint['child'] == link['name']]
                # find child joints
                child_joints = [joint for joint in naocs_joints
                                if joint if joint['type'] >= 0 if joint['parent'] == link['name']]
                connected_joints = parent_joints + child_joints
                part_class = part_info[i]['part_class']
                part_heatmap, part_unitvec = self.get_joint_params(naocs, connected_joints, points_class, part_class,
                                                                   num_parts)

                tmp_heatmap[part_class] = part_heatmap
                tmp_unitvec[part_class] = part_unitvec

            joints_association = tmp_heatmap.argmin(axis=0)
            points_heatmap = tmp_heatmap[joints_association, np.arange(naocs.shape[0])]
            points_unitvec = tmp_unitvec[joints_association, np.arange(naocs.shape[0])]
            points_unitvec[points_heatmap >= self.heatmap_threshold] = np.zeros(3)
            joints_association[points_heatmap >= self.heatmap_threshold] = -1
            points_heatmap_result = 1.0 - points_heatmap / self.heatmap_threshold
            points_heatmap_result[points_heatmap >= self.heatmap_threshold] = -1
            # points with no joint association has value 0
            joints_association += 1
            joints_axis = np.zeros((naocs.shape[0], 3))
            joint_types = np.zeros(num_parts)
            for joint in naocs_joints:
                if joint:
                    joints_axis[joints_association == joint['class']+1] = joint['axis']
                    joint_types[joint['class']] = joint['type']

            instance_name = f'{row["objectCat"]}_{row["objectId"]}_{row["articulationId"]}_{row["frameId"]}'
            if self.debug:
                tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, row['objectCat'],
                                            row['objectId'], self.tmp_output.folder_name)
                io.ensure_dir_exists(tmp_data_dir)

                window_name_prefix = instance_name
                utils.visualize_point_cloud(points_camera, points_class,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_camera %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'Camera Space', show=self.show)
                utils.visualize_point_cloud(npcs, points_class,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_npcs %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'NPCS Space', show=self.show)
                utils.visualize_point_cloud(naocs, points_class,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_naocs %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'NAOCS Space', show=self.show)
                utils.verify_npcs2camera(npcs, points_class, parts_npcs2cam_transformation, parts_npcs2cam_scale,
                                         export=os.path.join(tmp_data_dir, self.tmp_output.pcd_npcs2camera %
                                                             (int(row["articulationId"]), int(row["frameId"]))),
                                         window_name=window_name_prefix + 'NPCS to Camera', show=self.show)
                utils.visualize_heatmap_unitvec(naocs, points_heatmap_result, points_unitvec,
                                                export=os.path.join(tmp_data_dir, self.tmp_output.pcd_heatmap_unitvec %
                                                                    (int(row["articulationId"]), int(row["frameId"]))),
                                                window_name=window_name_prefix + 'Heatmap and unitvec', show=self.show)
                utils.visualize_joints_axis(naocs, joints_association, joints_axis,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_joints_axis %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'Joints axis and association', show=self.show)

            h5frame = h5file.require_group(instance_name)
            h5frame.attrs['objectCat'] = row["objectCat"]
            h5frame.attrs['objectId'] = row["objectId"]
            h5frame.attrs['articulationId'] = row["articulationId"]
            h5frame.attrs['frameId'] = row["frameId"]
            h5frame.attrs['numParts'] = num_parts
            h5frame.create_dataset("seg_per_point", shape=points_class.shape, data=mask, compression="gzip")
            h5frame.create_dataset("camcs_per_point", shape=points_camera.shape, data=points_camera, compression="gzip")
            h5frame.create_dataset("npcs_per_point", shape=npcs.shape, data=npcs, compression="gzip")
            h5frame.create_dataset("naocs_per_point", shape=naocs.shape, data=naocs, compression="gzip")
            h5frame.create_dataset("heatmap_per_point", shape=points_heatmap_result.shape, data=points_heatmap_result,
                                   compression="gzip")
            h5frame.create_dataset("unitvec_per_point", shape=points_unitvec.shape, data=points_unitvec,
                                   compression="gzip")
            h5frame.create_dataset("joint_axis_per_point", shape=joints_axis.shape, data=joints_axis,
                                   compression="gzip")
            h5frame.create_dataset("joint_cls_per_point", shape=joints_association.shape, data=joints_association,
                                   compression="gzip")
            h5frame.create_dataset("joint_type", shape=joint_types.shape, data=joint_types, compression="gzip")
            # 6D transformation from npcs to camcs
            h5frame.create_dataset("npcs2cam_rt", shape=parts_npcs2cam_transformation.shape,
                                   data=parts_npcs2cam_transformation, compression="gzip")
            # scale from npcs to camcs
            h5frame.create_dataset("npcs2cam_scale", shape=parts_npcs2cam_scale.shape, data=parts_npcs2cam_scale,
                                   compression="gzip")
            h5frame.create_dataset("naocs2cam_rt", shape=naocs2cam_transformation.shape,
                                   data=naocs2cam_transformation, compression="gzip")
            h5frame.create_dataset("naocs2cam_scale", shape=(1,), data=naocs2cam_scale,
                                   compression="gzip")

        h5file.close()
