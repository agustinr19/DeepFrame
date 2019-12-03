from PIL import Image
import numpy as np
import random
import os
import re

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

class DataLoaderRGBD(object):
    def __init__(self, path, training_split=0.8, assemble_into_stacks=False, stack_length=10):
        self.samples = []
        pre_samples_a = []
        pre_samples_b = []
        self.assemble_into_stacks = assemble_into_stacks

        # loads all images at path
        for r, d, f in os.walk(path):
            for file in sorted(f, key=lambda fl: int(re.sub('[^0-9]','', fl))):
                if '.png' in file and 'depth' not in file:
                    filepath = os.path.join(r, file)
                    img_a = np.array(Image.open(filepath))
                    img_b = np.array(Image.open(filepath[:-4]+"_depth.png"))

                    if assemble_into_stacks:
                        pre_samples_a.append(img_a)
                        pre_samples_b.append(img_b)
                        if len(pre_samples_a) >= stack_length:
                            self.samples.append([np.stack(pre_samples_a), np.stack(pre_samples_b)])
                            pre_samples_a = []
                            pre_samples_b = []
                    else:
                        self.samples.append([img_a, img_b])

                    print("Finished loading "+file+"...")

        # splits samples into training and testing sets
        rand = np.random.random_sample((len(self.samples),))
        indices = np.arange(len(self.samples))
        self.training_indices = np.where(rand <= training_split, indices, None)
        self.testing_indices = np.where(rand > training_split, indices, None)
        self.training_indices = self.training_indices[self.training_indices != np.array(None)]
        self.testing_indices = self.testing_indices[self.testing_indices != np.array(None)]

    def frame_size(self):
        if self.assemble_into_stacks:
            return self.samples[0][0][0].shape
        return self.samples[0][0].shape

    def training_set(self):
        samples = []
        if self.assemble_into_stacks:
            in_samples = []
            out_samples = []

            for i in range(len(self.samples)):
                if i in self.training_indices:
                    in_samples.append(self.samples[i][0])
                    out_samples.append(self.samples[i][1])
            return [np.stack(in_samples), np.stack(out_samples)]

        for i in range(len(self.samples)):
            if i in self.training_indices:
                samples.append(np.stack(self.samples[i], axis=-1))
        return np.stack(samples, axis=0)

    def testing_set(self):
        samples = []
        if self.assemble_into_stacks:
            in_samples = []
            out_samples = []

            for i in range(len(self.samples)):
                if i in self.testing_indices:
                    in_samples.append(self.samples[i][0])
                    out_samples.append(self.samples[i][1])
            return [np.stack(in_samples), np.stack(out_samples)]

        for i in range(len(self.samples)):
            if i in self.testing_indices:
                samples.append(np.stack(self.samples[i], axis=-1))
        return np.stack(samples, axis=0)

