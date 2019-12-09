import os
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import torchvision.transforms as transforms

DATA_EXTENSION = '.png'
def fetch_data(input_dir, timespan=1): #default does not cutoff any imgs
    imgs = []
    depth = []
    for directory, _, files in os.walk(input_dir):
        files = [file for file in files if file.endswith(DATA_EXTENSION)]
        dir_imgs = [os.path.join(directory, file) for file in files if 'depth' not in file]
        dir_depths = dir_files = [os.path.join(directory, file) for file in files if 'depth' in file]

        offset = len(dir_imgs) - (len(dir_imgs) % timespan)
        imgs += dir_imgs[:offset]
        depth += dir_depths[:offset]
        
    return list(zip(sorted(imgs), sorted(depth)))

def extract_data(path):
    img = np.array(Image.open(path))
    img = transforms.ToPILImage()(img)
    return img

class CNNSingleDataLoader(object): #cnn single, cnn stack
    def __init__(self, path, train=False, ratio=0.8):
        self.path = path
        self.to_tensor = transforms.ToTensor()
        self.data = fetch_data(path)

        data_split = int(len(self.data) * 0.8)
        if train:
            self.data = self.data[:data_split]
        else:
            self.data = self.data[data_split:]

        print(len(self.data))

    def __getitem__(self, index):
        rgb_path, depth_path = self.data[index]
        rgb = extract_data(rgb_path)
        depth = extract_data(depth_path)
        rgb, depth = self.to_tensor(rgb), self.to_tensor(depth)
        return rgb, depth

    def __len__(self):
        return len(self.data)

class CNNStackDataLoader(object): #cnn single, cnn stack
    def __init__(self, path, train=False, ratio=0.8, rgbd_scenes=True, stack_size=10, concat=False):
        self.ratio = ratio
        self.rgbd_scenes = rgbd_scenes
        self.path = path
        self.to_tensor = transforms.ToTensor()
        self.stack_size = stack_size
        self.concat = concat

        if not rgbd_scenes:
            if train:
                self.path+='/train'
            else:
                self.path+='/val'

        # if stack, fetch sequential data in multiples of stack_size
        self.img_data, self.depth_data = fetch_data_loc(path,timespan=self.stack_size)
        self.data_loc = list(zip(self.img_data,self.depth_data))

        if self.rgbd_scenes: # rgbd_scenes is not split intp train and val datasets
            # split by train/val ratio and round data lengths to nearest multiple of stack_size
            data_split = self.stack_size * round((len(self.data_loc)*ratio)/self.stack_size)
            if train:
                self.data_loc = self.data_loc[:data_split]
            else:
                self.data_loc = self.data_loc[data_split:]

        #extract data
        self.data = []
        for rgb_path,depth_path in self.data_loc:
            rgb = extract_data(rgb_path,rgbd=self.rgbd_scenes,rgb_img=True)
            depth = extract_data(depth_path,rgbd=self.rgbd_scenes,rgb_img=False)
            rgb, depth = self.to_tensor(rgb), self.to_tensor(depth)
            self.data.append((rgb, depth))

    def __getitem__(self, index):
        data_stack = self.data[index:self.stack_size+index]
        depth = self.data[index][1]
        rgb = [x[0] for x in data_stack] #isolate first part

        if self.concat:
            rgb = torch.cat(rgb)
        else:
            rgb = torch.stack(rgb)
            rgb = torch.sum(rgb,axis=0)/len(data_stack)

        return rgb, depth

    def __len__(self):
        return len(self.data)-self.stack_size
