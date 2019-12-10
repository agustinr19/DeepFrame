import os
import numpy as np
from PIL import Image

import keras

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
	return img

class RGBDDataGenerator(keras.utils.Sequence):
	def __init__(self, path, timespan=10, train=True):
		self.data = fetch_data(path, timespan=timespan)
		self.timespan = timespan

		self.n_timespans = len(self.data) // self.timespan
		self.split = int(self.n_timespans * 0.85) * self.timespan
		if train:
			self.data = self.data[:self.split]
		else:
			self.data = self.data[self.split:]

	def __getitem__(self, index):
		elements = self.data[index:index + self.timespan]

		rgb_combined = [extract_data(rgb_path) for (rgb_path, _) in elements]
		depth_combined = extract_data(self.data[index + self.timespan - 1][1])

		return np.expand_dims(np.stack(rgb_combined, axis=0), axis=0), np.expand_dims(depth_combined, axis=0)

	def __len__(self):
		return len(self.data)-self.timespan

# dataloader = RGBDDataGenerator('../data/rgbd-scenes/background')


# timespan width height channels 