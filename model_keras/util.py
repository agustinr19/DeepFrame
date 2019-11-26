import os
import shutil
import numpy as np
import time

def prepare_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def remove_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def save_text(text, path):
    file = open(path, 'w')
    file.write(text)
    file.close()

def swap_x_y(np_img, first_axis=0):
    return np.swapaxes(np_img, first_axis, first_axis+1)

def clean_repr(obj):
    return "".join(repr(obj).split("\n"))

def time_elapsed(func, args=[]):
    start = time.time()
    func(*args)
    end = time.time()

    print(" ** Time elapsed: "+str(end-start))
    return end-start

