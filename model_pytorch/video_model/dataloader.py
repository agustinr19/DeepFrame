import os
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import torchvision.transforms as transforms

DATA_EXTENSION = '.png'
def fetch_data_loc(directory,timespan=1): #default does not cutoff any imgs
    rgb = []
    depth = []
    for dirs, _, files in os.walk(directory):
        dir_files = []
        dir_depth = []
        for file in files:
            if file.endswith(DATA_EXTENSION) and 'depth' not in file:
                base = file[:-len(DATA_EXTENSION)]
                dir_depth.append(os.path.join(dirs, base+'_depth'+DATA_EXTENSION))
                dir_files.append(os.path.join(dirs,file))
#                print("Finished loading "+file+"...")
        if len(dir_files) > 0:
#            print(len(dir_files),dirs)
            offset = len(dir_files) % timespan
            if offset > 0:
                dir_files = dir_files[:-offset]
                dir_depth = dir_depth[:-offset]
            assert len(dir_files) % timespan == 0
            rgb += dir_files
            depth += dir_depth
#        print(len(rgb))
    return sorted(rgb),sorted(depth)

def extract_data(path, rgbd=True, rgb_img=True):
    if rgbd: # png -> rgbd_scenes dataset
        print(path)
        img = np.array(Image.open(path))
        img = transforms.ToPILImage()(img)
    else: # h5 -> nyuv2 dataset
        h5_data = h5py.File(h5_path, 'r')
        if rgb_img:
            img = h5_data['rgb'][()]
            img = np.transpose(rgb, (1, 2, 0))
        else:
            img = h5_data['depth'][()]
    return img

class CNNSingleDataLoader(object): #cnn single, cnn stack
    def __init__(self, path, train=False, ratio=0.8, rgbd_scenes=True):
        self.rgbd_scenes = rgbd_scenes
        self.path = path
        self.ratio = ratio
        self.to_tensor = transforms.ToTensor()

        if not rgbd_scenes:
            if train:
                self.path+='/train'
            else:
                self.path+='/val'
#        print(self.path,self.rgbd_scenes)
        self.img_data, self.depth_data = fetch_data_loc(path)
        self.data_loc = list(zip(self.img_data,self.depth_data))

        if self.rgbd_scenes: # rgbd_scenes is not split intp train and val datasets
        # split by train/val ratio
            data_split = int(len(self.data_loc)*ratio)
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
        rgb, depth = self.data[index]
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
