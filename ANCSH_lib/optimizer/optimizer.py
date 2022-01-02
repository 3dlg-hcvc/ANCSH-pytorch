import h5py
import multiprocessing

from numpy import isnat
from optimize_util import optimize_with_kinematic

import h5py


class ANCSHOptimizer:
    def __init__(self, cfg, ancsh_results_path, npcs_results_path):
        self.cfg = cfg
        self.f_ancsh = h5py.File(ancsh_results_path, "r+")
        self.f_npcs = h5py.File(npcs_results_path, "r+")
        self.instances = sorted(self.f_ancsh.keys())

    def optmimize(self, process_num=16):
        pool = multiprocessing.Pool(processes=process_num)
        print(f"runing {self.cfg.optimizer.niter} iterations for ransac")

        process = []
        # This should automatically change the result file
        for ins in self.instances:
            process.append(
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
            )
        pool.close()
        pool.join()

        self.results = process.get()

    def save(self):
        f = h5py.File(f"{self.cfg.paths.optimize.output_dir}/combined_result.h5", "w")
        f.attrs["network_type"] = self.f_ancsh["network_type"]
        for i, ins in enumerate(self.instances):
            result = self.results[i]
            group = f.create_group(ins)
            for k, v in self.f_ancsh[ins].items():
                # Use the pred seg and npcs from the npcs model
                if k == "pred_npcs_per_point" or k == "pred_seg_per_point":
                    group.create_dataset(k, data=self.f_npcs[ins][k], compression="gzip")
                group.create_dataset(k, data=v, compression="gzip")
            for k, v in result.items():
                group.create_dataset(k, data=v, compression="gzip")



