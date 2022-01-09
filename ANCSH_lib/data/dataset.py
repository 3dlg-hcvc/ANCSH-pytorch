import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ANCSHDataset(Dataset):
    def __init__(self, data_path, num_points):
        self.f_data = h5py.File(data_path)
        self.instances = sorted(self.f_data)
        self.num_points = num_points

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        id = self.instances[index]
        ins = self.f_data[id]

        # Get the points index used to sample points
        input_points = ins['camcs_per_point'][:]
        input_points_num = input_points.shape[0]
        perm = np.random.permutation(input_points_num)[:self.num_points]
        if self.num_points > input_points_num:
            additional = np.random.choice(input_points_num, self.num_points - input_points_num, replace=True)
            perm = np.concatenate((perm, additional))
        assert perm.shape[0] == self.num_points, f'{perm.shape[0]}, {self.num_points}, {input_points_num}'

        # Get the camcs_per_point
        camcs_per_point = torch.tensor(input_points, dtype=torch.float32)[perm]
        # Get all other items 
        gt_dict = {}
        for k, v in ins.items():
            if k == "camcs_per_point":
                continue
            elif "per_point" in k:
                gt_dict[k] = torch.tensor(v[:], dtype=torch.float32)[perm]
            else:
                gt_dict[k] = torch.tensor(v[:], dtype=torch.float32)

        return camcs_per_point, gt_dict, id
