import numpy as np
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import torch

class MNISTDataset(Dataset):
    def __init__(self, data_path, mode):
        self.data = []
        self.mode = mode
        # self.label = []
        
        if(mode == 'train'):
            for i in range(9):
                sub_folder_path = data_path + '/train/' + str(i) + '/'
                files =  os.listdir(sub_folder_path)
                for item in files:
                    # src = np.load(sub_folder_path + item)
                    self.data.append(sub_folder_path + item)

        elif(mode == 'val'):
            for i in range(9):
                sub_folder_path = data_path + '/val/' + str(i) + '/'
                files =  os.listdir(sub_folder_path)
                for item in files:
                    self.data.append(sub_folder_path + item)

        elif(mode == 'test'):
            sub_folder_path = data_path + '/test'
            files =  os.listdir(sub_folder_path)
            for item in files:
                pic = np.load(sub_folder_path + item)
                self.data.append(pic)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        # print(data)
        # label = self.label[index]
        if self.mode == 'train' or 'val':
            img = np.load(data)
            # print(img.shape)
            img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])(img)
            img = torch.transpose(img, 0, 1) 
            label = np.zeros([10])
            id = int(data.split('/')[-2])
            label[id] = 1
        
            return img, label