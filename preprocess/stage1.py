import os
import h5py
import yaml
import numpy as np

from PIL import Image
from yaml import CLoader as Loader, CDumper as Dumper
from scipy.spatial.transform import Rotation as R

from tools.utils import io
from utils import DataLoader, URDFReader, DatasetName


class ProcStage1:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_loader = DataLoader(cfg)
        self.data_loader.parse_input()
        self.input_cfg = self.cfg.paths.preprocess.stage1.input
        self.height = self.cfg.dataset.param.height
        self.width = self.cfg.dataset.param.width

    def preprocess_motion_data(self, motion_data_df):
        for index, motion_data in motion_data_df.iterrows():
            motion_file_path = os.path.join(self.data_loader.motion_dir, motion_data['objectCat'],
                                            motion_data['objectId'], motion_data['motion'])
            assert io.file_exist(motion_file_path), f'Can not found Motion file {motion_file_path}!'
            if DatasetName[self.cfg.dataset.name] == DatasetName.SAPIEN:
                urdf_reader = URDFReader(motion_file_path)
                urdf_reader.export(
                    result_data_path=os.path.join(self.cfg.paths.preprocess.tmp_dir,
                                                  motion_data['objectCat'], motion_data['objectId']),
                    rest_state_data_filename=self.cfg.paths.preprocess.stage1.tmp_output.rest_state_data,
                    rest_state_mesh_filename=self.cfg.paths.preprocess.stage1.tmp_output.rest_state_mesh
                )

    def get_metadata(self, metadata_path, frame_index):
        metadata = {}
        if DatasetName[self.cfg.dataset.name] == DatasetName.SAPIEN:
            with open(metadata_path, "r") as meta_file:
                metadata_all = yaml.load(meta_file, Loader=yaml.Loader)
            frame_metadata = metadata_all[f'frame_{frame_index}']
            metadata = {
                'projMat': frame_metadata['projMat'],
                'viewMat': frame_metadata['viewMat'],
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

    def process_each(self):
        data_info = self.data_loader.data_info
        selected_categories = data_info['objectCat'].isin(self.cfg.settings.categories) \
            if len(self.cfg.settings.categories) > 0 else data_info['objectCat']
        selected_object_ids = data_info['objectId'].isin(self.cfg.settings.object_ids) \
            if len(self.cfg.settings.object_ids) > 0 else data_info['objectId']
        selected_articulation_ids = data_info['articulationId'].isin(self.cfg.settings.articulation_ids) \
            if len(self.cfg.settings.articulation_ids) > 0 else data_info['articulationId']
        input_data = data_info[selected_categories & selected_object_ids & selected_articulation_ids]
        input_data.to_csv(os.path.join(self.cfg.paths.preprocess.tmp_dir,
                                       self.cfg.paths.preprocess.stage1.tmp_output.input_files))

        motion_data_df = input_data.drop_duplicates(subset=['objectCat', 'objectId', 'motion'])
        self.preprocess_motion_data(motion_data_df)
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
            rest_state_data_path = os.path.join(self.cfg.paths.preprocess.tmp_dir,
                                                input_each['objectCat'], input_each['objectId'],
                                                self.cfg.paths.preprocess.stage1.tmp_output.rest_state_data)

            frame_index = int(input_each['depthFrame'].split(self.input_cfg.render.depth_ext)[0])
            depth_data = np.array(h5py.File(depth_frame_path, "r")["data"])
            # float32 depth buffer, range from 0 to 1
            depth_frame = np.reshape(depth_data, (self.height, self.width))
            # uint8 mask, invalid value is 255
            mask_frame = np.asarray(Image.open(mask_frame_path))
            metadata = self.get_metadata(metadata_path, frame_index)

