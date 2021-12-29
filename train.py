from ANCSH_lib import ANCSHTrainer
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf

import argparse

@hydra.main(config_path="configs", config_name="network")
def main(cfg):
    train_path = cfg.paths.dirs.preprocess_result_dir + cfg.paths.train_name
    test_path = cfg.paths.dirs.preprocess_result_dir + cfg.paths.test_name
    data_path = {"train": train_path, "test": test_path}

    trainer = ANCSHTrainer(data_path=data_path, num_parts=cfg.network.num_parts, max_epochs=cfg.network.max_epochs)
    # if not cfg.test:
    #     trainer.train()
    # else:
    #     trainer.test()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
