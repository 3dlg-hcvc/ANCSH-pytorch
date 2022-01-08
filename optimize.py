import random
import torch
from time import time
import numpy as np
import h5py
import logging

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHOptimizer
from ANCSH_lib.utils import set_random_seed
from tools.utils import io

log = logging.getLogger('optimize')

def get_num_parts(h5_file_path):
    if not io.file_exist(h5_file_path):
        log.error(f'Cannot open file {h5_file_path}')
        return False
    input_h5 = h5py.File(h5_file_path, 'r')
    num_parts = input_h5[list(input_h5.keys())[0]].attrs['numParts']
    bad_groups = []
    visit_groups = lambda name, node: bad_groups.append(name) if isinstance(node, h5py.Group) and node.attrs[
        'numParts'] != num_parts else None
    input_h5.visititems(visit_groups)
    input_h5.close()
    if len(bad_groups) > 0:
        log.error(f'Instances {bad_groups} in {h5_file_path} have different number of parts than {num_parts}')
        return False
    return num_parts

@hydra.main(config_path="configs", config_name="optimize")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    ancsh_results_path = cfg.ancsh_results_path
    npcs_results_path = cfg.npcs_results_path

    set_random_seed(cfg.random_seed)

    train_path = cfg.paths.preprocess.output.train

    num_parts = get_num_parts(train_path)
    log.info(f'Instances in dataset have {num_parts} parts')

    optimizer = ANCSHOptimizer(cfg, ancsh_results_path, npcs_results_path, num_parts=num_parts)
    optimizer.optimize(process_num=16)
    optimizer.print_and_save()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    log.info(str(stop - start) + " seconds")
