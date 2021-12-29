import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../config", config_name="config")
def main(cfg : DictConfig):
    # hydra by default is on another directory
    cwd = os.getcwd()
    os.chdir(get_original_cwd())

    os.chdir(cwd)