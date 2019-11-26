from PIL import Image
import numpy as np
import random
import os

class DataLoader(object):
    def __init__(self, path, training_split=0.8):
        self.samples = []

        # loads all images at path
        for r, d, f in os.walk(path):
            for file in f:
                if '_a.png' in file:
                    filepath = r+"/"+file
                    img_a = np.array(Image.open(filepath))
                    img_b = np.array(Image.open(filepath[:-5]+"b.png"))
                    self.samples.append([img_a, img_b])

        # splits samples into training and testing sets
        rand = np.random.random_sample((len(self.samples),))
        indices = np.arange(len(self.samples))
        self.training_indices = np.where(rand <= training_split, indices, None)
        self.testing_indices = np.where(rand > training_split, indices, None)
        self.training_indices = self.training_indices[self.training_indices != np.array(None)]
        self.testing_indices = self.testing_indices[self.testing_indices != np.array(None)]

    def frame_size(self):
        return self.samples[0][0].shape

    def training_set(self):
        samples = []
        for i in range(len(self.samples)):
            if i in self.training_indices:
                samples.append(np.stack(self.samples[i], axis=-1))
        return np.stack(samples, axis=0)

    def testing_set(self):
        samples = []
        for i in range(len(self.samples)):
            if i in self.testing_indices:
                samples.append(np.stack(self.samples[i], axis=-1))
        return np.stack(samples, axis=0)

