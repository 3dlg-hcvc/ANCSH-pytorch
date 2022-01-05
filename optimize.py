from ANCSH_lib import ANCSHOptimizer
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import random
import torch

def set_random_seed(seed):
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

@hydra.main(config_path="configs", config_name="optimize")
def main(cfg):
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