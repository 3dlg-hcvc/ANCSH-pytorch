import pdb
from torch.utils.data import Dataset
import h5py
import torch
import numpy as np

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
        perm = np.random.permutation(ins["camcs_per_point"].shape[0])

        # Get the camcs_per_point
        camcs_per_point = torch.tensor(ins["camcs_per_point"], dtype=torch.float32)[perm[:self.num_points]]
        # Get all other items 
        gt_dict = {}
        for k, v in ins.items():
            if k == "camcs_per_point":
                continue
            elif "per_point" in k:
                gt_dict[k] = torch.tensor(v, dtype=torch.float32)[perm[:self.num_points]]
            else:
                gt_dict[k] = torch.tensor(v, dtype=torch.float32)
        
        return (camcs_per_point, gt_dict, id)