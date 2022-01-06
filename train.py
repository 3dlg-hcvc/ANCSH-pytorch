import os
import h5py
import logging
import torch
from time import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHTrainer, utils
from tools.utils import io

log = logging.getLogger('train')


def get_num_parts(h5_file_path):
    if not io.file_exist(h5_file_path):
        log.error(f'Cannot open file {h5_file_path}')
        return False
    input_h5 = h5py.File(h5_file_path, 'r')
    num_parts = input_h5[list(input_h5.keys())[0]].attrs['numParts']
    bad_groups = []
    visit_groups = lambda name, node: bad_groups.append(name) if isinstance(node, h5py.Group) and node.attrs[
        'numParts'] != num_parts else None
    input_h5.visititems(visit_groups)
    input_h5.close()
    if len(bad_groups) > 0:
        log.error(f'Instances {bad_groups} in {h5_file_path} have different number of parts than {num_parts}')
        return False
    return num_parts


@hydra.main(config_path="configs", config_name="network")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    train_path = cfg.paths.preprocess.output.train
    test_path = cfg.paths.preprocess.output.val if cfg.test.split == 'val' else cfg.paths.preprocess.output.test
    data_path = {"train": train_path, "test": test_path}

    num_parts = get_num_parts(train_path)
    test_num_parts = get_num_parts(test_path)
    assert num_parts == test_num_parts
    log.info(f'Instances in dataset have {num_parts} parts')

    network_type = cfg.network.network_type

    utils.set_random_seed(cfg.random_seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    trainer = ANCSHTrainer(
        cfg=cfg,
        data_path=data_path,
        network_type=network_type,
        num_parts=num_parts,
    )
    if not cfg.eval_only:
        log.info(f'Train on {train_path}, validate on {test_path}')
        trainer.train()
    else:
        log.info(f'Test on {test_path}')
        trainer.test(inference_model=cfg.inference_model)


if __name__ == "__main__":
    start = time()

    main()

    stop = time()

    duration_time = utils.duration_in_hours(stop - start)
    log.info(f'Total time duration: {duration_time}')
