from ANCSH_lib import ANCSHTrainer
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import torch
import os 

def set_random_seed(seed):
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

@hydra.main(config_path="configs", config_name="network")
def main(cfg):
    train_path = cfg.paths.preprocess.output.train
    test_path = cfg.paths.preprocess.output.val
    data_path = {"train": train_path, "test": test_path}

    network_type = cfg.network.network_type

    set_random_seed(cfg.random_seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    trainer = ANCSHTrainer(
        cfg=cfg,
        data_path=data_path,
        network_type=network_type,
        num_parts=cfg.network.num_parts,
    )
    if not cfg.test:
        trainer.train()
    else:
        trainer.test(inference_model=cfg.inference_model)


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
