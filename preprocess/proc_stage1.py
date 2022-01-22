import os
import h5py
import yaml
import math
import logging
import numpy as np
import pandas as pd

from PIL import Image
from scipy.spatial.transform import Rotation as R
from progress.bar import Bar
from multiprocessing import Pool, cpu_count
from omegaconf import OmegaConf

from tools.utils import io
from tools.visualization import Viewer
from utils import DataLoader, URDFReader, DatasetName

log = logging.getLogger('proc_stage1')


class ProcStage1Impl:
    def __init__(self, cfg):
        self.output_path = cfg.output_path
        self.tmp_dir = cfg.tmp_dir
        self.render_cfg = cfg.render_cfg
        self.rest_state_data_filename = cfg.rest_state_data_filename
        self.width = self.render_cfg.width
        self.height = self.render_cfg.height
        self.dataset_name = cfg.dataset_name
        self.defined_rest_state_file = cfg.defined_rest_state_file

    def get_metadata(self, object_id, metadata_path, frame_index, num_parts, rest_state_data=None):
        metadata = {}
        if DatasetName[self.dataset_name] == DatasetName.SAPIEN or \
                DatasetName[self.dataset_name] == DatasetName.SHAPE2MOTION:
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
        elif DatasetName[self.dataset_name] == DatasetName.MOTIONNET:
            frame_metadata = io.read_json(metadata_path)
            fov = frame_metadata['camera']['intrinsic']['fov'] / 180 * np.pi
            fx = self.height / (2 * math.tan(fov / 2))
            fy = fx / self.height * self.width
            cx = self.width / 2
            cy = self.height / 2
            intrinsic = np.diag([fx, fy, 1])
            intrinsic[:2, 2] = [cx, cy]
            extrinsic = np.reshape(frame_metadata['camera']['extrinsic']['matrix'], (4, 4), order='F')
            extrinsic = np.linalg.inv(extrinsic)
            orientation = extrinsic[:3, :3]
            orientation = np.dot(orientation, R.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_matrix())
            orientation = np.dot(orientation, R.from_rotvec(np.pi / 2 * np.array([0, 1, 0])).as_matrix())
            orientation = np.dot(orientation, R.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix())
            extrinsic[:3, :3] = orientation
            metadata = {
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'linkAbsPoses': []
            }
            # one part case
            # motion = frame_metadata['motions'][0]
            # link_names = [link['name'] for link in rest_state_data['links']]
            # for i, link in enumerate(rest_state_data['links']):
            #     if link['virtual']:
            #         continue
            #     joint = rest_state_data['joints'][i]
            #     pose = np.reshape(link['abs_pose'], (4, 4), order='F')
            #     if link['name'] == f'link_{motion["partId"]}':
            #         rest_state = 0
            #         if io.file_exist(self.defined_rest_state_file):
            #             defined_rest_state = pd.read_csv(self.defined_rest_state_file, dtype=str)
            #             if object_id in defined_rest_state['objectId'].tolist():
            #                 rest_state = float(
            #                     defined_rest_state[defined_rest_state['objectId'] == object_id]['restState'])
            #         joint_state = np.abs(motion['value'] - rest_state)
            #         joint_axis = joint['axis']
            #         joint_pose = np.reshape(joint['pose2link'], (4, 4), order='F')
            #         joint_parent = joint['parent']
            #         parent_link = rest_state_data['links'][link_names.index(joint_parent)]
            #         parent_link_abs_pose = np.asarray(parent_link['abs_pose']).reshape((4, 4), order='F')
            #         joint_abs_pose = np.dot(parent_link_abs_pose, joint_pose)
            #         translation = np.eye(4)
            #         translation_inv = np.eye(4)
            #         translation[:3, 3] = joint_abs_pose[:3, 3]
            #         translation_inv[:3, 3] = -joint_abs_pose[:3, 3]
            #         joint_abs_axis = np.dot(joint_abs_pose[:3, :3], joint_axis)
            #         rotation = np.eye(4)
            #         rotation[:3, :3] = R.from_rotvec(joint_state * np.array(joint_abs_axis)).as_matrix()
            #         pose = np.dot(rotation, np.dot(translation_inv, pose))
            #         pose = np.dot(translation, pose)
            #     metadata['linkAbsPoses'].append(pose)

        return metadata

    def __call__(self, idx, input_data):
        output_filepath = os.path.splitext(self.output_path)[0] + f'_{idx}' + os.path.splitext(self.output_path)[-1]
        h5file = h5py.File(output_filepath, 'w')
        bar = Bar(f'Stage1 Processing chunk {idx}', max=len(input_data))
        for index, input_each in input_data.iterrows():
            tmp_data_dir = os.path.join(self.tmp_dir, input_each['objectCat'], input_each['objectId'])
            rest_state_data_path = os.path.join(tmp_data_dir, self.rest_state_data_filename)

            if DatasetName[self.dataset_name] == DatasetName.SAPIEN or \
                    DatasetName[self.dataset_name] == DatasetName.SHAPE2MOTION:
                depth_frame_path = os.path.join(self.render_cfg.render_dir, input_each['objectCat'],
                                                input_each['objectId'], input_each['articulationId'],
                                                self.render_cfg.depth_folder, input_each['depthFrame'])
                mask_frame_path = os.path.join(self.render_cfg.render_dir, input_each['objectCat'],
                                               input_each['objectId'], input_each['articulationId'],
                                               self.render_cfg.mask_folder, input_each['maskFrame'])
                metadata_path = os.path.join(self.render_cfg.render_dir, input_each['objectCat'],
                                             input_each['objectId'], input_each['articulationId'],
                                             input_each['metadata'])

                frame_index = int(input_each['depthFrame'].split(self.render_cfg.depth_ext)[0])
                # float32 depth buffer, range from 0 to 1
                depth_data = np.array(h5py.File(depth_frame_path, "r")["data"])
            elif DatasetName[self.dataset_name] == DatasetName.MOTIONNET:
                depth_frame_path = os.path.join(self.render_cfg.render_dir, input_each['objectCat'],
                                                input_each['objectId'], self.render_cfg.depth_folder,
                                                input_each['depthFrame'])
                    
                metadata_path = os.path.join(self.render_cfg.render_dir, input_each['objectCat'],
                                             input_each['objectId'], self.render_cfg.metadata_folder,
                                             input_each['metadata'])

                frame_index = int(input_each['depthFrame'].split('_d' + self.render_cfg.depth_ext)[0].split('-')[-1])
                depth_data = np.asarray(Image.open(depth_frame_path))

                for i, mask in enumerate(input_each['maskFrame']):
                    mask_frame_path = os.path.join(self.render_cfg.render_dir, input_each['objectCat'],
                                                input_each['objectId'], self.render_cfg.mask_folder,
                                                mask)
                    mask_frame = np.asarray(Image.open(mask_frame_path))
                    empty_points = mask_frame[:, :, 3] == 0
                    mask_tmp = mask_frame[~empty_points][:, :3]
                    link_mask = -np.ones(mask_tmp.shape[0])
                    base_pixels = np.all(mask_tmp == [0, 0, 0], axis=-1)
                    part_pixels = ~base_pixels
                    print(mask)
                    link_mask[part_pixels] = float(os.path.splitext(mask)[0].split('_')[-1])
            # uint8 mask, invalid value is 255
            if DatasetName[self.dataset_name] == DatasetName.SAPIEN or \
                    DatasetName[self.dataset_name] == DatasetName.SHAPE2MOTION:
                mask_frame = np.asarray(Image.open(mask_frame_path))
                rest_state_data = io.read_json(rest_state_data_path)
                num_parts = len([link for link in rest_state_data['links'] if link if not link['virtual']])
                assert depth_data.shape[:2] == mask_frame.shape[:2]
                metadata = self.get_metadata(input_each['objectId'], metadata_path, frame_index, num_parts, rest_state_data)

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
            elif DatasetName[self.dataset_name] == DatasetName.MOTIONNET:
                num_parts = len(input_each['maskFrame']) + 1
                metadata = self.get_metadata(input_each['objectId'], metadata_path, frame_index, num_parts)
                x_range = np.linspace(0, self.width, self.width)
                y_range = np.linspace(0, self.height, self.height)
                intrinsic = metadata['intrinsic']
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx, cy = intrinsic[:2, 2]
                x, y = np.meshgrid(x_range, y_range)
                z = 1.0 * depth_data / 1000.0
                z = z.flatten()
                x = (x.flatten() - cx) * z / fx
                y = -(y.flatten() - cy) * z / fy
                z = -z
                points_tmp = np.column_stack((x, y, z, np.ones_like(z)))
                empty_points = empty_points.flatten()
                points_camera = points_tmp[~empty_points]
                # check if unique value in mask match num parts
                assert points_camera.shape[0] == link_mask.shape[0]
                extrinsic = metadata['extrinsic']
                points_camera = points_camera.transpose()
                # shape 4xn
                points_world = np.dot(np.linalg.inv(extrinsic), points_camera)
                view_mat = extrinsic

            # transform links to rest state
            # points_rest_state = np.empty_like(points_world)
            # parts_camera2rest_state = []
            # for link_idx, link in enumerate(rest_state_data['links']):
            #     if link['virtual']:
            #         continue
            #     link_points_world = points_world[:, link_mask == link_idx]
            #     # virtual link link_index is -1
            #     current_part_pose = metadata['linkAbsPoses'][link['part_index']]
            #     rest_state_pose = np.reshape(link['abs_pose'], (4, 4), order='F')
            #     transform2rest_state = np.dot(rest_state_pose, np.linalg.inv(current_part_pose))
            #     link_points_rest_state = np.dot(transform2rest_state, link_points_world)
            #     points_rest_state[:, link_mask == link_idx] = link_points_rest_state
            #     # points in camera space to rest state
            #     camera2rest_state = np.dot(transform2rest_state, np.linalg.inv(view_mat))
            #     # shape num parts x 16
            #     parts_camera2rest_state.append(camera2rest_state.flatten('F'))
            # parts_camera2rest_state = np.asarray(parts_camera2rest_state)
            # # shape nx3
            points_camera_p3 = points_camera.transpose()[:, :3]
            # points_world_p3 = points_world.transpose()[:, :3]
            # points_rest_state_p3 = points_rest_state.transpose()[:, :3]

            # viewer = Viewer(points_world_p3, colors=Viewer.rgba_by_index(0))
            # viewer.show()

            camera2base_matrix = np.linalg.inv(view_mat).flatten('F')
            # instance_name = f'{input_each["objectCat"]}_{input_each["objectId"]}_{input_each["articulationId"]}_{str(frame_index)}'
            instance_name = f'{input_each["objectCat"]}_{input_each["objectId"]}_{input_each["articulationId"]}_{str(frame_index)}'
            h5frame = h5file.require_group(instance_name)
            print(link_mask)
            h5frame.create_dataset("seg_per_point", shape=link_mask.shape, data=link_mask, compression="gzip")
            h5frame.create_dataset("camcs_per_point", shape=points_camera_p3.shape, data=points_camera_p3,
                                   compression="gzip")
            # h5frame.create_dataset("points_rest_state", shape=points_rest_state_p3.shape, data=points_rest_state_p3,
            #                        compression="gzip")
            # h5frame.create_dataset("parts_transformation", shape=parts_camera2rest_state.shape,
            #                        data=parts_camera2rest_state, compression="gzip")
            extrinsic = extrinsic.flatten(order='F')
            intrinsic = intrinsic.flatten(order='F')
            h5frame.create_dataset("intrinsic", shape=intrinsic.shape,
                                   data=intrinsic, compression="gzip")
            h5frame.create_dataset("extrinsic", shape=extrinsic.shape,
                                   data=extrinsic, compression="gzip")

            bar.next()
        bar.finish()
        h5file.close()
        return output_filepath


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
        bar = Bar('Stage1 Parse Motion Data', max=len(motion_data_df))
        for index, motion_data in motion_data_df.iterrows():
            motion_file_path = os.path.join(self.data_loader.motion_dir, motion_data['objectCat'],
                                            motion_data['objectId'], motion_data['motion'])
            assert io.file_exist(motion_file_path), f'Can not found Motion file {motion_file_path}!'

            metadata_file_path = None
            defined_rest_state = None
            if DatasetName[self.cfg.dataset.name] == DatasetName.MOTIONNET:
                metadata_file_path = os.path.join(self.data_loader.render_dir, motion_data['objectCat'],
                                                  motion_data['objectId'],
                                                  self.cfg.paths.preprocess.stage1.input.render.metadata_folder,
                                                  motion_data['metadata'])
                assert io.file_exist(metadata_file_path), f'Can not found Metadata file {metadata_file_path}!'
                defined_rest_state_file = os.path.join(self.cfg.paths.preprocess.input_dir,
                                                       self.input_cfg.defined_rest_state_file)
                defined_rest_state = pd.read_csv(defined_rest_state_file, dtype=str)
                defined_rest_state = defined_rest_state[defined_rest_state['objectId'] == motion_data['objectId']]

            urdf_reader = URDFReader(motion_file_path, metadata_file_path, defined_rest_state)
            tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name,
                                        motion_data['objectCat'], motion_data['objectId'])
            urdf_reader.export(
                result_data_path=tmp_data_dir,
                rest_state_data_filename=self.tmp_output.rest_state_data,
                rest_state_mesh_filename=self.tmp_output.rest_state_mesh
            )

            bar.next()
        bar.finish()

    def process(self):
        input_data = self.data_loader.data_info
        io.ensure_dir_exists(self.cfg.paths.preprocess.tmp_dir)
        input_data.to_csv(os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.input_files))

        # motion_data_df = input_data.drop_duplicates(subset=['objectCat', 'objectId', 'motion'])
        # self.preprocess_motion_data(motion_data_df)
        io.ensure_dir_exists(self.cfg.paths.preprocess.output_dir)

        num_processes = min(cpu_count(), self.cfg.num_workers)
        # calculate the chunk size
        chunk_size = max(1, int(input_data.shape[0] / num_processes))
        chunks = [input_data.iloc[input_data.index[i:i + chunk_size]] for i in
                  range(0, input_data.shape[0], chunk_size)]
        log.info(f'Stage1 Processing Start with {num_processes} workers and {len(chunks)} chunks')

        config = OmegaConf.create()
        config.output_path = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name,
                                          self.output_cfg.pcd_data)
        config.tmp_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name)
        render_cfg = OmegaConf.create()
        render_cfg.width = self.width
        render_cfg.height = self.height
        render_cfg.render_dir = self.data_loader.render_dir
        render_cfg.depth_ext = self.input_cfg.render.depth_ext
        render_cfg.mask_ext = self.input_cfg.render.mask_ext
        render_cfg.depth_folder = self.input_cfg.render.depth_folder
        render_cfg.mask_folder = self.input_cfg.render.mask_folder
        if DatasetName[self.cfg.dataset.name] == DatasetName.MOTIONNET:
            render_cfg.metadata_folder = self.input_cfg.render.metadata_folder
        config.render_cfg = render_cfg
        config.rest_state_data_filename = self.tmp_output.rest_state_data
        config.dataset_name = self.cfg.dataset.name
        config.defined_rest_state_file = os.path.join(self.cfg.paths.preprocess.input_dir,
                                                      self.input_cfg.defined_rest_state_file)
        with Pool(processes=num_processes) as pool:
            proc_impl = ProcStage1Impl(config)
            output_filepath_list = pool.starmap(proc_impl, enumerate(chunks))

        output_file_path = os.path.join(self.cfg.paths.preprocess.output_dir, self.output_cfg.pcd_data)
        h5file = h5py.File(output_file_path, 'w')
        for filepath in output_filepath_list:
            with h5py.File(filepath, 'r') as h5f:
                for key in h5f.keys():
                    h5f.copy(key, h5file)
        h5file.close()

        if self.debug:
            tmp_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name)
            with h5py.File(output_file_path, 'r') as h5file:
                bar = Bar('Stage1 Visualization', max=len(h5file.keys()))
                for key in h5file.keys():
                    h5group = h5file[key]
                    folder_names = key.split('_')
                    viz_output_dir = os.path.join(tmp_dir, folder_names[0], folder_names[1], folder_names[2])
                    viz_output_filename = key
                    viz_output_path = os.path.join(viz_output_dir, viz_output_filename)

                    viewer = Viewer(h5group['camcs_per_point'][:], mask=h5group['seg_per_point'][:])
                    if self.cfg.show:
                        viewer.show(window_name=viz_output_filename + '_points_camera')
                    else:
                        viewer.render(fig_path=viz_output_path + '_points_camera.jpg')
                    # if self.cfg.export:
                    #     viewer.export(mesh_path=viz_output_path + '_points_camera.ply')
                    # viewer.reset()
                    # viewer.add_geometry(h5group['points_rest_state'][:], mask=h5group['mask'][:])
                    # if self.cfg.show:
                    #     viewer.show(window_name=viz_output_filename + '_points_rest_state')
                    # else:
                    #     viewer.render(fig_path=viz_output_path + '_points_rest_state.jpg')
                    # if self.cfg.export:
                    #     viewer.export(mesh_path=viz_output_path + '_points_rest_state.ply')
                    del viewer
                    bar.next()
                bar.finish()
