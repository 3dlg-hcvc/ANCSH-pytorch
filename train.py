from ANCSH_lib import ANCSHTrainer
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Train ANCSH model")
    parser.add_argument(
        "--test",
        action="store_true",
        help="indicate whether it's training or inferencing",
    )

    return parser

@hydra.main(config_path="configs", config_name="network")
def main(cfg):
    # args = get_parser().parse_args()
    # conf = OmegaConf.to_yaml(cfg)
    print(cfg.test==True)
    # trainer = ANCSHTrainer(data_path=args.data_path, max_epochs=1000)
    # if not args.test:
    #     trainer.train()
    # else:
    #     trainer.test()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
