import os
import h5py
import numpy as np

from tools.utils import io
from utils import DataLoader, URDFReader, DatasetName


class ProcStage1:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_loader = DataLoader(cfg)
        self.data_loader.parse_input()

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

    def process_each(self):
        data_info = self.data_loader.data_info
        input_data = data_info[(data_info['objectCat'].isin(self.cfg.settings.categories))
                               & (data_info['objectId'].isin(self.cfg.settings.object_ids))
                               & (data_info['articulationId'].isin(self.cfg.settings.articulation_ids))]

        motion_data_df = input_data.drop_duplicates(subset=['objectCat', 'objectId', 'motion'])
        self.preprocess_motion_data(motion_data_df)
        for index, input_each in input_data.iterrows():
            depth_frame_path = os.path.join(self.data_loader.motion_dir, input_each['objectCat'],
                                            input_each['objectId'], input_each['articulationId'],
                                            input_each['depthFrame'])
            mask_frame_path = os.path.join(self.data_loader.motion_dir, input_each['objectCat'],
                                           input_each['objectId'], input_each['articulationId'],
                                           input_each['maskFrame'])
            metadata_path = os.path.join(self.data_loader.motion_dir, input_each['objectCat'],
                                         input_each['objectId'], input_each['articulationId'],
                                         input_each['metadata'])
            rest_state_data_path = os.path.join(self.cfg.paths.preprocess.tmp_dir,
                                                input_each['objectCat'], input_each['objectId'],
                                                self.cfg.paths.preprocess.stage1.tmp_output.rest_state_data)

            depth = np.array(h5py.File(depth_frame_path, "r")["data"])
            depth = np.reshape(depth, (self.height, self.width))

