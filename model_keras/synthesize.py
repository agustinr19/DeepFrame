import os
import numpy as np
from PIL import Image
from util import *
import random

def generate_dataset_a(path, frame_size=(256, 256), n=200):
    prepare_path(path)
    for i in range(n):
        out_a = np.zeros((frame_size[0], frame_size[1], 3))
        out_b = np.zeros((frame_size[0], frame_size[1], 3))
        rand_color = np.array([random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)])
        out_a[:, :, :] = rand_color
        out_b[:, :, :] = rand_color*2

        # saves images
        im_a = Image.fromarray(out_a.astype(np.uint8))
        im_a.save(path+"/"+str(i)+"_a.png")
        im_b = Image.fromarray(out_b.astype(np.uint8))
        im_b.save(path+"/"+str(i)+"_b.png")

if __name__ == "__main__":
    generate_dataset_a("test_dataset")