import os
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import dataloaders.transforms as transforms

DATA_EXTENSION = '.png'
def fetch_data_loc(directory):
    rgb = []
    depth = []
    for directory, _, files in os.walk(directory):
        for file in files:
            if file.endswith(DATA_EXTENSION):
                if 'depth' in file:
                    depth.append(os.path.join(directory, file))
                else:
                    rgb.append(os.path.join(directory, file))
                print("Finished loading "+file+"...")
    return rgb,depth

def extract_data(path, rgb=True):
    img = np.array(Image.open(path))
    img = transforms.ToPILImage()(img)
    return img

class CustomDataLoader(object):
    def __init__(self, path, train=False, ratio=0.8):
        self.path = path      
        self.img_data, self.depth_data = fetch_data_loc(path)
        self.data = list(zip(self.img_data,self.depth_data))
        self.to_tensor = transforms.ToTensor()

        if train:
            self.data = self.data[:int(len(self.data)*ratio)]
        else: 
            self.data = self.data[int(len(self.data)*ratio):]


    def __getitem__(self, index):
        rgb, depth = self.data[index]
        rgb, depth = extract_data(rgb), extract_data(depth)    
        rgb, depth = self.to_tensor(rgb), self.to_tensor(depth)        
        return rgb, depth

    def __len__(self):
        return len(self.data)

