import h5py
import multiprocessing

from .optimize_util import optimize_with_kinematic

import numpy as np


class ANCSHOptimizer:
    def __init__(self, cfg, ancsh_results_path, npcs_results_path):
        self.cfg = cfg
        self.f_ancsh = h5py.File(ancsh_results_path, "r")
        self.f_npcs = h5py.File(npcs_results_path, "r")
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

    def print_and_save(self):
        # Calculate the mean error for each part
        errs_rotation = np.array([result["err_rotation"] for result in self.results])
        errs_translation = np.array([result["err_translation"] for result in self.results])
        mean_err_rotation = np.mean(errs_rotation, axis=0)
        mean_err_translation = np.mean(errs_translation, axis=0)
        # Calculate the accuaracy for err_rotation < 5 degree
        acc_err_rotation = np.mean((errs_rotation < 5).float(), axis=0)
        # Calculate the the accuracy for err_rt, rotation < 5 degree, translation < 5 cm
        acc_err_rt = np.mean(np.logical_and((errs_rotation < 5), (errs_translation < 0.05)).float(), axis=0)

        print(f"The mean rotation error for each part is: {mean_err_rotation}")
        print(f"The mean translation error for each part is: {mean_err_translation}")
        print(f"The accuracy for rotation error < 5 degree is: {acc_err_rotation}")
        print(f"The accuracy for rotation error < 5 degree and translation error < 5 cm is: {acc_err_rt}")

        f = h5py.File(f"{self.cfg.paths.optimize.output_dir}/combined_result.h5", "w")
        f.attrs["network_type"] = self.f_ancsh["network_type"]
        # Record the errors
        f.attrs["err_pose_rotation"] = mean_err_rotation
        f.attrs["err_pose_translation"] = mean_err_translation
        f.attrs["acc_pose_rotation"] = acc_err_rotation
        f.attrs["acc_pose_rt"] = acc_err_rt
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



