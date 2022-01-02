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
        id = self.instances[index]
        ins = self.f_data[id]
        import pdb
        pdb.set_trace()
        # Get the camcs_per_point
        camcs_per_point = ins["camcs_per_point"]
        # Get all other items 
        gt_dict = {}
        for k, v in ins.items():
            if k == "camcs_per_point":
                continue
            else:
                gt_dict[k] = v
        
        return (camcs_per_point, gt_dict, id)