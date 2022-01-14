import random
import torch
from time import time
import numpy as np
import h5py
import logging

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHOptimizer, utils
from tools.utils import io

log = logging.getLogger('optimize')


@hydra.main(config_path="configs", config_name="optimize")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    ancsh_results_path = cfg.ancsh_results_path
    npcs_results_path = cfg.npcs_results_path

    utils.set_random_seed(cfg.random_seed)

    train_path = cfg.paths.preprocess.output.train

    num_parts = utils.get_num_parts(train_path)
    # num_parts = 3
    log.info(f'Instances in dataset have {num_parts} parts')

    optimizer = ANCSHOptimizer(cfg, ancsh_results_path, npcs_results_path, num_parts=num_parts)
    optimizer.optimize(process_num=cfg.num_workers)
    optimizer.print_and_save()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    log.info(str(stop - start) + " seconds")
