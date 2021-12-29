import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="preprocess")
def main(cfg : DictConfig):
    # hydra by default is on another directory
    print(cfg.paths.preprocess.output_dir)
    cwd = os.getcwd()
    os.chdir(get_original_cwd())

    os.chdir(cwd)

if __name__ == "__main__":
    main()