from torch.utils.data import Dataset
import h5py
import torch

class ANCSHDataset(Dataset):
    def __init__(self, gt_path):
        # self.h5file = h5py.File(gt_path)
        self.x = torch.rand(32, 1024, 3)
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        camera_per_point = self.x[index]
        gt_dict = {"filename": "TEST", "value": torch.rand(2)}
        return (camera_per_point, gt_dict)