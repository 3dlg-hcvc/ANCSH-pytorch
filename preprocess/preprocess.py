import os
import time
import logging

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from tools.utils import io

from stage1 import ProcStage1, ProcStage2

log = logging.getLogger('preprocess')


@hydra.main(config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.dataset_dir", io.to_abs_path(cfg.paths.dataset_dir, get_original_cwd()))
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    OmegaConf.update(cfg, "paths.preprocess.stage1.input", cfg.dataset.input)
    print(cfg.paths.preprocess.input_dir)

    assert io.folder_exist(cfg.paths.preprocess.input_dir), "Dataset directory doesn't exist"
    io.ensure_dir_exists(cfg.paths.preprocess.output_dir)

    if cfg.settings.stage1.process:
        start = time.time()
        process_stage1 = ProcStage1(cfg)
        process_stage1.process()
        end = time.time()
        log.info(f'Stage1 process time {end - start}')

    OmegaConf.update(cfg, "paths.preprocess.stage2.input.split_info",
                     io.to_abs_path(cfg.paths.preprocess.stage2.input.split_info, get_original_cwd()))

    if cfg.settings.stage2.process:
        start = time.time()
        process_stage2 = ProcStage2(cfg)
        process_stage2.split_data(train_percent=cfg.settings.stage2.train_percent, seed=cfg.settings.stage2.split_seed)
        process_stage2.process()
        end = time.time()
        log.info(f'Stage2 process time {end - start}')


if __name__ == "__main__":
    main()
