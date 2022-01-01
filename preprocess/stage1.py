import os
import h5py
import pandas as pd
import trimesh
import yaml
import numpy as np

from PIL import Image
from yaml import CLoader as Loader, CDumper as Dumper
from scipy.spatial.transform import Rotation as R

from tools.utils import io
import utils
from utils import DataLoader, URDFReader, DatasetName


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

    def split_data(self, train_percent=.6, seed=None):
        datasets = []
        visit_leaves = lambda name, node: datasets.append(name) if isinstance(node, h5py.Dataset) else None
        self.input_h5.visititems(visit_leaves)
        df_dataset = pd.DataFrame([name.split('/') for name in datasets],
                                  columns=['objectCat', 'objectId', 'articulationId', 'frameId', 'dataName'])
        df_dataset = df_dataset[['objectCat', 'objectId', 'articulationId', 'frameId']] \
            .drop_duplicates(ignore_index=True)
        if io.file_exist(self.cfg.paths.preprocess.stage2.input.split_info, ext='.csv'):
            input_split_info = pd.read_csv(self.cfg.paths.preprocess.stage2.input.split_info)
            self.split_info = input_split_info.merge(df_dataset, how='inner',
                                                     on=['objectCat', 'objectId', 'articulationId', 'frameId'])
        else:
            # split to train, val, test
            df_size = len(df_dataset)
            val_end = train_percent + (1.0 - train_percent) / 2.0
            train, val, test = np.split(df_dataset.sample(frac=1, random_state=seed),
                                        [int(train_percent * df_size), int(val_end * df_size)])

            self.split_info = pd.concat([train, val, test], keys=["train", "val", "test"], names=['set', 'index'])
        self.split_info.to_csv(os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.split_info))

    def process(self):
        io.ensure_dir_exists(self.output_dir)
        train = self.split_info.loc['train']
        self.process_each(train,
                          os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.train_data))
        val = self.split_info.loc['val']
        self.process_each(val,
                          os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.val_data))
        test = self.split_info.loc['test']
        self.process_each(test,
                          os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.test_data))

    def process_each(self, data_info, output_path):
        # process object info
        stage1_input = self.cfg.paths.preprocess.stage1.input
        part_orders = io.read_json(os.path.join(self.cfg.paths.preprocess.input_dir, stage1_input.part_order_file))

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
            part_order = part_orders[row['objectCat']][row['objectId']]
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

            object_info = object_infos[row['objectCat']][row['objectId']]['object']
            # diagonal axis aligned bounding box length to 1
            # (0.5, 0.5, 0.5) centered
            naocs_translation = - object_info['center'] + 0.5 * object_info['scale']
            naocs_scale = 1.0 / object_info['scale']
            naocs = points_rest_state + naocs_translation
            naocs *= naocs_scale

            points_class = np.empty_like(mask)
            part_info = object_infos[row['objectCat']][row['objectId']]['part']
            num_parts = len(part_info)
            mask_val = np.unique(mask)
            npcs = np.empty_like(points_rest_state)
            parts_npcs2cam_transformation = np.empty_like(parts_camera2rest_state)
            parts_npcs2cam_scale = np.empty(num_parts)
            for i, link_index in enumerate(mask_val):
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
                npcs_transformation = np.reshape(parts_camera2rest_state[i], (4, 4), order='F')
                npcs_transformation[:3, 3] += npcs_translation
                npcs2cam_transformation = np.linalg.inv(npcs_transformation)
                parts_npcs2cam_transformation[part_class] = npcs2cam_transformation.flatten('F')
                parts_npcs2cam_scale[part_class] = 1.0 / npcs_scale

            heatmap = np.random.rand(points_camera.shape[0], 1)
            unit_vec = np.random.rand(points_camera.shape[0], 3)
            joint_axis = np.random.rand(points_camera.shape[0], 3)
            joint_class = np.ones(points_camera.shape[0])
            joint_type = np.ones(num_parts)

            instance_name = f'{row["objectCat"]}_{row["objectId"]}_{row["articulationId"]}_{row["frameId"]}'
            if self.debug:
                tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, row['objectCat'],
                                            row['objectId'], self.tmp_output.folder_name)
                io.ensure_dir_exists(tmp_data_dir)

                window_name_prefix = instance_name
                utils.visualize_point_cloud(points_camera, points_class,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_camera %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'Camera Space', show=False)
                utils.visualize_point_cloud(npcs, points_class,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_npcs %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'NPCS Space', show=False)
                utils.visualize_point_cloud(naocs, points_class,
                                            export=os.path.join(tmp_data_dir, self.tmp_output.pcd_naocs %
                                                                (int(row["articulationId"]), int(row["frameId"]))),
                                            window_name=window_name_prefix + 'NAOCS Space', show=False)
                utils.verify_npcs2camera(npcs, points_class, parts_npcs2cam_transformation, parts_npcs2cam_scale,
                                         export=os.path.join(tmp_data_dir, self.tmp_output.pcd_npcs2camera %
                                                             (int(row["articulationId"]), int(row["frameId"]))),
                                         window_name=window_name_prefix + 'NPCS to Camera', show=False)

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
            h5frame.create_dataset("heatmap_per_point", shape=heatmap.shape, data=heatmap, compression="gzip")
            h5frame.create_dataset("unitvec_per_point", shape=unit_vec.shape, data=unit_vec, compression="gzip")
            h5frame.create_dataset("joint_axis_per_point", shape=joint_axis.shape, data=joint_axis, compression="gzip")
            h5frame.create_dataset("joint_cls_per_point", shape=joint_class.shape, data=joint_class, compression="gzip")
            # 6D transformation from npcs to camcs
            h5frame.create_dataset("npcs2cam_rt", shape=parts_npcs2cam_transformation.shape,
                                   data=parts_npcs2cam_transformation, compression="gzip")
            # scale from npcs to camcs
            h5frame.create_dataset("npcs2cam_scale", shape=parts_npcs2cam_scale.shape, data=parts_npcs2cam_scale,
                                   compression="gzip")
            h5frame.create_dataset("joint_type", shape=joint_type.shape, data=joint_type, compression="gzip")

        h5file.close()
