from torch.utils.data import Dataset
import h5py

class ANCSHDataset(Dataset):
    def __init__(self, gt_path):
        self.h5file = h5py.File(gt_path)
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
        camera_per_point = None
        gt_dict = None
        return (camera_per_point, gt_dict)