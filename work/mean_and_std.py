#!/usr/bin/env python

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy

# Allow image embeding in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

image_width = image_height = 224

image_dir = '/home/joeantol/work/project-x/data/testimages/gunks/trapps'
image_file = os.path.join(image_dir, 'missbailey_20180310_152947_020.jpg')
# image_file = os.path.join(image_dir, 'missbailey-1.jpg')

# image = Image.open(image_file).convert("RGB").resize((image_width, image_height))
image = Image.open(image_file).convert("RGB").rotate(-90).resize((image_width, image_height))

img = np.array(image)

np.mean(img), np.std(img)

plt.imshow(img)

img_bw = img
img_bw[:] = img_bw.mean(axis=-1,keepdims=1) 
np.mean(img_bw), np.std(img_bw)

plt.imshow(img_bw)

img_bw[1]

stats_file = '/home/joeantol/work/project-x/data/union/gunks/trapps/trainval/training/train_stats_color.pk'

with open(stats_file, 'rb') as f:
    num_samples = pickle.load(f)
    sample_mean = pickle.load(f)
    sample_std  = pickle.load(f)
    
num_samples, sample_mean, sample_std, sample_mean.mean(), sample_std.mean()

