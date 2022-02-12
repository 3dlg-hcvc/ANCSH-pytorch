import h5py
import os
from tools.visualization import OptimizerVisualizer

if __name__ == "__main__":
    with h5py.File(, "r") as h5file:
        visualizer = OptimizerVisualizer(h5file)
        export_dir = os.path.join()
        visualizer.render(show=False, export=export_dir, export_mesh=False)