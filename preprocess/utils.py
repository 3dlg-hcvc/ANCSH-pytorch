import os

from enum import Enum
from collections import namedtuple
from omegaconf import OmegaConf
import pandas as pd

from tools.utils import io

InputFrame = namedtuple('Frame', 'depth mask metadata rgb')


class DatasetName(Enum):
    SAPIEN = 0
    MULTISCAN = 1


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        dataset_name = self.cfg.dataset.name
        self.dataset_name = DatasetName[dataset_name] if isinstance(dataset_name, str) else dataset_name
        self.dataset_dir = self.cfg.paths.preprocess.input_dir
        self.stage1_input = self.cfg.paths.preprocess.stage1.input
        self.render_dir = os.path.join(self.dataset_dir, self.stage1_input.render.folder_name)
        self.motion_dir = os.path.join(self.dataset_dir, self.stage1_input.motion.folder_name)
        self.part_order = io.read_json(os.path.join(self.dataset_dir, self.stage1_input.part_order_file))
        self.data_info = pd.DataFrame()

    def parse_render_result(self):
        df_list = []
        object_cats = os.listdir(self.render_dir)
        # object categories
        for object_cat in object_cats:
            object_cat_path = os.path.join(self.render_dir, object_cat)
            object_ids = io.alphanum_ordered_folder_list(object_cat_path)
            # object instance ids
            for object_id in object_ids:
                object_part_order = self.part_order[object_cat][object_id]
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
                        [[object_cat, object_id, object_part_order, articulation_id, depth_frames[i], mask_frames[i],
                         metadata_file]],
                        columns=['objectCat', 'objectId', 'partOrder', 'articulationId',
                                 'depthFrame', 'maskFrame', 'metadata']) for i in range(num_renders)],
                        ignore_index=True)
                    df_list.append(df_row)
        self.data_info = pd.concat(df_list, ignore_index=True)


def get_articulated_object_inputs(articulated_object_paths, cfg):
    for articulated_object_path in articulated_object_paths:
        # get number of rendered instances:
        depth_files = io.alphanum_ordered_folder_list(
            os.path.join(articulated_object_path, cfg.paths.preprocess.input.depth_folder))
        mask_files = io.alphanum_ordered_folder_list(
            os.path.join(articulated_object_path, cfg.paths.preprocess.input.mask_folder))
        if io.folder_exist(os.path.join(articulated_object_path, cfg.paths.preprocess.input.depth_folder)):
            rgb_files = io.alphanum_ordered_folder_list(
                os.path.join(articulated_object_path, cfg.paths.preprocess.input.rgb_folder))
