import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from tools.utils import io

from utils import DataLoader, URDFReader


@hydra.main(config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.dataset_dir", io.to_abs_path(cfg.paths.dataset_dir, get_original_cwd()))
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    OmegaConf.update(cfg, "paths.preprocess.stage1.input", cfg.dataset.input)

    assert io.folder_exist(cfg.paths.preprocess.input_dir), "Dataset directory doesn't exist"
    io.ensure_dir_exists(cfg.paths.preprocess.output_dir)
    data_loader = DataLoader(cfg)
    data_loader.parse_input()
    # print(data_loader.data_info)
    each = data_loader.data_info.iloc[0]
    motion_file_path = os.path.join(data_loader.motion_dir, each['objectCat'], each['objectId'], each['motion'])
    urdf_reader = URDFReader(motion_file_path)
    urdf_reader.export(result_data_path=os.path.join(cfg.paths.preprocess.tmp_dir, each['objectCat'], each['objectId']))


if __name__ == "__main__":
    main()
