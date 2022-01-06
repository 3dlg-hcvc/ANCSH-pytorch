import random
import torch
import numpy as np
from enum import Enum


class NetworkType(Enum):
    ANCSH = 'ancsh'
    NPCS = 'npcs'
    NAOCS = 'naocs'


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
