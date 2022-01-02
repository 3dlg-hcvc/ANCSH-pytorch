from torch.utils.data import Dataset
import h5py
import torch

class ANCSHDataset(Dataset):
    def __init__(self, data_path):
        self.f_data = h5py.File(data_path)
        self.instances = sorted(self.f_data)
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        ins = self.f_data[self.instances[index]]
        # Get the camcs_per_point
        camcs_per_point = ins["camcs_per_point"]
        # Get the joint type
        joint_type = ins["joint_type"]
        # Get all other items 
        gt_dict = {}
        for k, v in ins.items():
            if k == "camcs_per_point" or k == "joint_type":
                continue
            else:
                gt_dict[k] = v
        
        return (camcs_per_point, gt_dict, joint_type)