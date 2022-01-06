import os
from time import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHEvaluator
from tools.utils import io


@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    combined_result_path = os.path.join(cfg.paths.optimization.output_dir,
                                        cfg.paths.optimization.optimization_result_path)
    evaluator = ANCSHEvaluator(cfg, combined_result_path)
    evaluator.process_ANCSH()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
