import os
import h5py
import numpy as np
import multiprocessing
import logging
from time import time

from ANCSH_lib.optimizer.optimize_util import optimize_with_kinematic
from tools.utils import io
from tools.visualization import OptimizerVisualizer


def h5_dict(ins):
    result = {}
    for k, v in ins.items():
        result[k] = np.array(v)
    return result


class ANCSHOptimizer:
    def __init__(self, cfg, ancsh_results_path, npcs_results_path, num_parts):
        start = time()
        self.cfg = cfg
        self.num_parts = num_parts
        self.log = logging.getLogger('optimizer')
        self.results = None
        self.log.info("Loading the data from results hdf5 file")
        self.f_ancsh = h5py.File(ancsh_results_path, "r")
        self.f_npcs = h5py.File(npcs_results_path, "r")
        self.instances = sorted(self.f_ancsh.keys())
        self.log.info(f"Load the data: {time()-start} seconds")

    def optimize(self, process_num=4):
        pool = multiprocessing.Pool(processes=process_num)
        self.log.info(f"runing {self.cfg.optimization.niter} iterations for ransac")
        process = []
        # This should automatically change the result file
        for ins in self.instances:
            process.append(
                pool.apply_async(
                    optimize_with_kinematic,
                    (
                        ins,
                        h5_dict(self.f_ancsh[ins]),
                        h5_dict(self.f_npcs[ins]),
                        self.num_parts,
                        self.cfg.optimization.niter,
                        self.cfg.optimization.choose_threshold,
                        self.log,
                    ),
                )
            )

            # optimize_with_kinematic(
            #     ins,
            #     h5_dict(self.f_ancsh[ins]),
            #     h5_dict(self.f_npcs[ins]),
            #     self.num_parts,
            #     self.cfg.optimization.niter,
            #     self.cfg.optimization.choose_threshold,
            #     self.log,
            # )
        pool.close()
        pool.join()

        self.results = [p.get() for p in process]

    def print_and_save(self):
        # Calculate the mean error for each part
        errs_rotation = []
        errs_translation = []
        valid_num = 0
        for result in self.results:
            if result["is_valid"][0] == True:
                valid_num += 1
                errs_rotation.append(result["err_rotation"])
                errs_translation.append(result["err_translation"])
        errs_rotation = np.array(errs_rotation)
        errs_translation = np.array(errs_translation)

        mean_err_rotation = np.mean(errs_rotation, axis=0)
        mean_err_translation = np.mean(errs_translation, axis=0)
        # Calculate the accuaracy for err_rotation < 5 degree
        acc_err_rotation = np.mean(errs_rotation < 5, axis=0)
        # Calculate the the accuracy for err_rt, rotation < 5 degree, translation < 5 cm
        acc_err_rt = np.mean(
            np.logical_and((errs_rotation < 5), (errs_translation < 0.05)),
            axis=0,
        )

        self.log.info(f"Valid Number {valid_num} / Total number {len(self.results)}")
        self.log.info(f"The mean rotation error for each part is: {mean_err_rotation}")
        self.log.info(f"The mean translation error for each part is: {mean_err_translation}")
        self.log.info(f"The accuracy for rotation error < 5 degree is: {acc_err_rotation}")
        self.log.info(
            f"The accuracy for rotation error < 5 degree and translation error < 5 cm is: {acc_err_rt}"
        )

        io.ensure_dir_exists(self.cfg.paths.optimization.output_dir)
        optimization_result_path = os.path.join(self.cfg.paths.optimization.output_dir,
                                                self.cfg.paths.optimization.optimization_result_path)
        f = h5py.File(optimization_result_path, "w")
        # Record the errors
        f.attrs["valid_num"] = valid_num
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
                    group.create_dataset(
                        k, data=self.f_npcs[ins][k], compression="gzip"
                    )
                else:
                    group.create_dataset(k, data=v, compression="gzip")
            for k, v in result.items():
                group.create_dataset(k, data=v, compression="gzip")
        f.close()

        render_cfg = self.cfg.render
        if render_cfg.render:
            with h5py.File(optimization_result_path, "r") as h5file:
                visualizer = OptimizerVisualizer(h5file)
                export_dir = os.path.join(self.cfg.paths.optimization.output_dir,
                                          self.cfg.paths.optimization.visualization_folder)
                visualizer.render(show=render_cfg.show, export=export_dir, export_mesh=render_cfg.export)
