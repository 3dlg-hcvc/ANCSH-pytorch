import os
import h5py
import yaml
import logging
import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R

from tools.utils import io
import utils
from utils import DataLoader, URDFReader, DatasetName

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
            if DatasetName[self.cfg.dataset.name] == DatasetName.SAPIEN or \
                    DatasetName[self.cfg.dataset.name] == DatasetName.SHAPE2MOTION:
                urdf_reader = URDFReader(motion_file_path)
                tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, motion_data['objectCat'],
                                            motion_data['objectId'], self.tmp_output.folder_name)
                urdf_reader.export(
                    result_data_path=tmp_data_dir,
                    rest_state_data_filename=self.tmp_output.rest_state_data,
                    rest_state_mesh_filename=self.tmp_output.rest_state_mesh
                )

    def get_metadata(self, metadata_path, frame_index, num_parts):
        metadata = {}
        if DatasetName[self.cfg.dataset.name] == DatasetName.SAPIEN or \
                DatasetName[self.cfg.dataset.name] == DatasetName.SHAPE2MOTION:
            with open(metadata_path, "r") as meta_file:
                metadata_all = yaml.load(meta_file, Loader=yaml.Loader)
            frame_metadata = metadata_all[f'frame_{frame_index}']
            metadata = {
                'projMat': np.reshape(frame_metadata['projMat'], (4, 4), order='F'),
                'viewMat': np.reshape(frame_metadata['viewMat'], (4, 4), order='F'),
                'linkAbsPoses': []
            }
            num_links = len(frame_metadata['obj'])
            if num_links < num_parts:
                metadata['linkAbsPoses'].append(np.eye(4))
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
            rest_data_data = io.read_json(rest_state_data_path)
            num_parts = len([link for link in rest_data_data['links'] if link if not link['virtual']])
            assert depth_data.size == mask_frame.size
            metadata = self.get_metadata(metadata_path, frame_index, num_parts)
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
