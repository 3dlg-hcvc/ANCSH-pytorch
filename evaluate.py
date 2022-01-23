import logging
from time import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHEvaluator, utils
from tools.utils import io

log = logging.getLogger('evaluate')


@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    combined_result_path = cfg.optimization_result_path

    num_parts = cfg.num_parts
    log.info(f'Instances in dataset have {num_parts} parts')

    evaluator = ANCSHEvaluator(cfg, combined_result_path, num_parts=num_parts)
    evaluator.process_ANCSH(do_eval=False)


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
