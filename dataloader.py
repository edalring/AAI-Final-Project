import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from pathlib import Path
from utils.data_analysis import resolve_greyscale_channel
from utils.torch_utils import load_npy_as_ndarray, multi_processes_execute

class MNISTDataset(Dataset):
    def __init__(self, data_path, mode, idxs):
        self.data_root = Path(data_path) / mode

        if not self.data_root.exists():
            # raise error
            print(f'Dataset {self.data_root} does not exist!')
            exit(1)

        
        self.mode = mode
        self.all_data_paths = list(self.data_root.glob('**/*.npy'))
        # self.label = []
        
        if idxs is None:
            self.data_paths = self.all_data_paths
        else :
            self.data_paths = list(Path(idx)for idx in idxs) 

        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
        ])

        # self.images = multi_processes_execute(self.load_npy_as_tensor, self.data_paths, workers=16)

        # if mode == 'test':
        #     self.labels = [0] * len(self.images)
        # else:
        #     self.labels = [torch.Tensor(int(path.parent.name)) for path in self.data_paths]

    
    def load_npy_as_tensor(self, path):
        img = self.transform(load_npy_as_ndarray(path))
        return torch.transpose(img, 0, 1) 
        
        # return load_npy_as_ndarray(path)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        path = self.data_paths[index]
        img = self.load_npy_as_tensor(path)
        # print(data)
        # label = self.label[index]
        if self.mode == 'test':
            return img, 0   
        else:
            label = int(path.parent.name)

            return img, label