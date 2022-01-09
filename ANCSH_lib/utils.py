import os
import random
import torch
import h5py
import numpy as np
from enum import Enum
from datetime import datetime

from tools.utils import io


class NetworkType(Enum):
    ANCSH = 'ANCSH'
    NPCS = 'NPCS'


def set_random_seed(seed):
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


def duration_in_hours(duration):
    t_m, t_s = divmod(duration, 60)
    t_h, t_m = divmod(t_m, 60)
    duration_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return duration_time


def get_prediction_vertices(pred_segmentation, pred_coordinates):
    segmentations = np.argmax(pred_segmentation, axis=1)
    coordinates = pred_coordinates[
        np.arange(pred_coordinates.shape[0]).reshape(-1, 1),
        np.arange(3) + 3 * np.tile(segmentations.reshape(-1, 1), [1, 3])]
    return segmentations, coordinates


def get_num_parts(h5_file_path):
    if not io.file_exist(h5_file_path):
        raise IOError(f'Cannot open file {h5_file_path}')
    input_h5 = h5py.File(h5_file_path, 'r')
    num_parts = input_h5[list(input_h5.keys())[0]].attrs['numParts']
    bad_groups = []
    visit_groups = lambda name, node: bad_groups.append(name) if isinstance(node, h5py.Group) and node.attrs[
        'numParts'] != num_parts else None
    input_h5.visititems(visit_groups)
    input_h5.close()
    if len(bad_groups) > 0:
        raise ValueError(f'Instances {bad_groups} in {h5_file_path} have different number of parts than {num_parts}')
    return num_parts


def get_latest_file_with_datetime(path, folder_prefix, ext, datetime_pattern='%Y-%m-%d_%H-%M-%S'):
    folders = os.listdir(path)
    folder_pattern = folder_prefix + datetime_pattern
    matched_folders = np.asarray([fd for fd in folders if fd.startswith(folder_prefix)])
    if len(matched_folders) == 0:
        return '', ''
    timestamps = np.asarray([int(datetime.strptime(fd, folder_pattern).timestamp() * 1000) for fd in matched_folders])
    sort_idx = np.argsort(timestamps)
    matched_folders = matched_folders[sort_idx]
    latest_folder = matched_folders[-1]
    files = io.alphanum_ordered_file_list(os.path.join(path, latest_folder), ext=ext)
    latest_file = files[-1]
    return latest_folder, latest_file


class AvgRecorder(object):
    """
    Average and current value recorder
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
