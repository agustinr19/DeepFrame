import os
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

DATA_EXTENSION = '.png'
def fetch_data_loc(directory):
    rgb = []
    depth = []
    for dirs, _, files in os.walk(directory):
        for file in files:
            if file.endswith(DATA_EXTENSION) and 'depth' not in file:
                base = file[:-len(DATA_EXTENSION)]
                depth.append(os.path.join(dirs, base+'_depth'+DATA_EXTENSION))
                rgb.append(os.path.join(dirs, file))
#                print("Finished loading "+file+"...")
    return sorted(rgb),sorted(depth)

def extract_data(path, rgb=True):
    img = np.array(Image.open(path))
    img = transforms.ToPILImage()(img)
    return img

class CustomDataLoader(object):
    def __init__(self, path, train=False, ratio=0.8, stack=False, stack_size=10, concat=False):
        self.path = path
        self.img_data, self.depth_data = fetch_data_loc(path)
        self.data_loc = list(zip(self.img_data,self.depth_data))
        self.to_tensor = transforms.ToTensor()
        self.stack_size = stack_size
        self.stack = stack
        self.concat = concat

        if train:
            self.data_loc = self.data_loc[:int(len(self.data_loc)*ratio)]
        else:
            self.data_loc = self.data_loc[int(len(self.data_loc)*ratio):]

        self.data = []
        for rgb_path,depth_path in self.data_loc:
            rgb, depth = extract_data(rgb_path),extract_data(depth_path)
            rgb, depth = self.to_tensor(rgb), self.to_tensor(depth)
            self.data.append((rgb, depth))

    def __getitem__(self, index):
        if self.stack:
            data_stack = self.data[max(0,index-self.stack_size+1):index+1]
            depth = self.data[index][1]
            rgb = [x[0] for x in data_stack] #isolate first part

            if self.concat:
                rgb = torch.cat(rgb)
            else:
                rgb = torch.stack(rgb)
                rgb = torch.sum(rgb,axis=0)/len(data_stack)
        else:
            rgb, depth = self.data[index]

        print(rgb.shape)
        return rgb, depth

    def __len__(self):
        return len(self.data)


