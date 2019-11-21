import random

import numpy as np

import torch

import dataloaders.transforms as transforms
import dataloaders.dataloader as CustomDataLoader

iheight, iwidth = 480, 640

class NYUDataset(CustomDataLoader.CustomDataset):

    def __init__(self, directory, dims, output_size, train=True):
        super().__init__(directory)
        self.dims = dims
        self.output_size = output_size
        if train:
            self.transform = self.train_transform
            self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        else:
            self.transform = self.validate_transform

    def train_transform(self, rgb, depth):
        scale = np.random.uniform(low=1, high=1.5)
        depth = depth / scale

        angle = np.random.uniform(-5.0, 5.0)
        should_flip = np.random.uniform(0.0, 1.0) < 0.5

        h_offset = int((768 - 228) * np.random.uniform(0.0, 1.0))
        v_offset = int((1024 - 304) * np.random.uniform(0.0, 1.0))

        base_transform = transforms.Compose([
            transforms.Resize(250 / iheight),
            transforms.Rotate(angle),
            transforms.Resize(scale),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(should_flip),
        ])

        rgb = base_transform(rgb)
        rgb = self.color_jitter(rgb)
        rgb = rgb / 255.0

        depth = base_transform(depth)

        return (rgb, depth)

    def validate_transform(self, rgb, depth):
        h_offset = int((768 - 228) * np.random.uniform(0.0, 1.0))
        v_offset = int((1024 - 304) * np.random.uniform(0.0, 1.0))

        base_transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])

        rgb = base_transform(rgb)
        rgb = rgb / 255.0
        depth = base_transform(depth)

        return (rgb, depth)
