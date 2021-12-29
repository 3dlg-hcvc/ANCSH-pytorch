from ANCSH_lib import ANCSHTrainer
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf

import argparse


@hydra.main(config_path="configs", config_name="network")
def main(cfg):
    train_path = cfg.paths.preprocess.output.train
    test_path = cfg.paths.preprocess.output.val
    data_path = {"train": train_path, "test": test_path}

    network_type = cfg.network.network_type

    trainer = ANCSHTrainer(
        data_path=data_path,
        network_type=network_type,
        num_parts=cfg.network.num_parts,
        max_epochs=cfg.network.max_epochs,
    )
    # if not cfg.test:
    #     trainer.train()
    # else:
    #     trainer.test()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
