import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from tools.utils import io

from stage1 import ProcStage1


@hydra.main(config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.dataset_dir", io.to_abs_path(cfg.paths.dataset_dir, get_original_cwd()))
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    OmegaConf.update(cfg, "paths.preprocess.stage1.input", cfg.dataset.input)

    assert io.folder_exist(cfg.paths.preprocess.input_dir), "Dataset directory doesn't exist"
    io.ensure_dir_exists(cfg.paths.preprocess.output_dir)

    process_stage1 = ProcStage1(cfg)
    process_stage1.process_each()


if __name__ == "__main__":
    main()
