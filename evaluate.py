from ANCSH_lib import ANCSHEvaluator
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg):
    combined_result_path = cfg.combined_result_path
    evaluator = ANCSHEvaluator(cfg, combined_result_path)
    evaluator.process_ANCSH()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")