from ANCSH_lib import ANCSHOptimizer
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="optimize")
def main(cfg):
    ancsh_results_path = cfg.ancsh_results_path
    npcs_results_path = cfg.npcs_results_path
    optimizer = ANCSHOptimizer(cfg, ancsh_results_path, npcs_results_path)
    optimizer.optmimize(process_num=16)
    optimizer.save()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")