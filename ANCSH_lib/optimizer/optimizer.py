import h5py
import multiprocessing
from optimize_util import optimize_with_kinematic

class ANCSHOptimizer:
    def __init__(self, cfg, ancsh_results_path, npcs_results_path):
        self.cfg = cfg
        self.f_ancsh = h5py.File(ancsh_results_path, "r+")
        self.f_npcs = h5py.File(npcs_results_path, "r+")

    def optmimize(self, process_num=16):
        pool = multiprocessing.Pool(processes=process_num)
        print(f"runing {self.cfg.optimizer.niter} iterations for ransac")
        # This should automatically change the result file
        for ins in self.f_ancsh.keys():
            pool.apply_async(
                optimize_with_kinematic,
                (
                    self.f_ancsh[ins],
                    self.f_npcs[ins],
                    self.cfg.optimizer.num_parts,
                    self.cfg.optimizer.niter,
                    self.cfg.optimizer.choose_threshold,
                ),
            )
        pool.close()
        pool.join()
