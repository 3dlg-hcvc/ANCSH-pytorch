import random
import torch
from time import time
import numpy as np

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHOptimizer
from ANCSH_lib.utils import set_random_seed
from tools.utils import io


@hydra.main(config_path="configs", config_name="optimize")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    ancsh_results_path = cfg.ancsh_results_path
    npcs_results_path = cfg.npcs_results_path

    set_random_seed(cfg.random_seed)

    optimizer = ANCSHOptimizer(cfg, ancsh_results_path, npcs_results_path)
    optimizer.optimize(process_num=16)
    optimizer.print_and_save()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
