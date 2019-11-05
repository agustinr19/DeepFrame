import os
import time

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset

# import dataloaders.fang_transforms as transforms
import dataloaders.transforms as transforms

DATA_EXTENSION = '.h5'

def fetch_data(directory):
    data = []
    for directory, _, files in os.walk(directory):
        for file in files:
            if file.endswith(DATA_EXTENSION):
                data.append(os.path.join(directory, file))
    return sorted(data)

def fetch_rgbd_data(h5_path):
    h5_data = h5py.File(h5_path, 'r')

    rgb = h5_data['rgb'][()]
    rgb = np.transpose(rgb, (1, 2, 0))  
    depth = h5_data['depth'][()]

    return (rgb, depth)

class CustomDataset(Dataset):

    def __init__(self, directory):
        self.directory = directory
        self.data = fetch_data(self.directory)
        self.transform = None

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        (rgb, depth) = fetch_rgbd_data(self.data[index])        
        rgb, depth = self.transform(rgb, depth)
        
        if self.dims == 'rgb':
            network_input = self.to_tensor(rgb)
            depth = self.to_tensor(depth)
            depth = depth.unsqueeze(0)
            return (network_input, depth)

        network_input = rgb
        network_input = self.to_tensor(network_input)
       
        depth = self.to_tensor(depth)
        depth = depth.unsqueeze(0)
        
        return (network_input, depth)

    def __len__(self):
        return len(self.data)

