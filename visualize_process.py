import h5py
import os
from tools.visualization import ANCSHVisualizer
from enum import Enum

class NetworkType(Enum):
    ANCSH = 'ANCSH'
    NPCS = 'NPCS'

h5_output_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/ANCSH-pytorch/data/eyeglasses_ancsh/test.h5"
visualization_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/ANCSH-pytorch/visualization"

if __name__ == "__main__":
    with h5py.File(h5_output_path, 'r') as h5file:
        visualizer = ANCSHVisualizer(h5file, NetworkType.ANCSH, gt=True, sampling=20)
        visualizer.point_size = 5
        visualizer.arrow_sampling = 10
        visualizer.prefix = ''
        visualizer.render(show=False, export=visualization_path, export_mesh=True)