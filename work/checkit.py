#!/usr/bin/env python

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import keras
    
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from keras import backend as K
from keras.models import load_model

from random import shuffle

import numpy as np
import os

import pickle

from PIL import Image

image_dir = 'data/merged/21st-street/'
model_dir = '21stStreet-Adam-4986-200x200-6'
###model_file = 'model.35-0.89.hdf5'
model_file = 'model.h5'

width = height = 200

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, model_dir)

#... Load model and weights
model = load_model(os.path.join(model_path, model_file))
print('Loaded trained model from %s ' % model_path)

with open(os.path.join(model_path, 'label_map.pk'), 'rb') as f:
    label_map = pickle.load(f)
label_map = {v: k for k, v in label_map.items()}
print('Loaded label map...')

num_correct = num_wrong = 0

images = []
addr_list = os.listdir(image_dir)

for addr in addr_list:
    objs = [addr+'/' + s for s in os.listdir(os.path.join(image_dir, addr))]
    images += objs

shuffle(images)

gtruths = [s.split('/')[-2] for s in images]

for i in range(0,1000):

    image_file = images[i]
    gtruth     = gtruths[i]

    image = Image.open(os.path.join(image_dir, image_file)).convert("RGB").resize((width, height))

    img = np.array(image)
    ###img = img / 255.0
    
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    npimage = np.array([[r] + [g] + [b]], np.uint8)
    npimage = npimage.transpose(0,2,3,1)

    classes = model.predict_classes(npimage)
    pred = model.predict(npimage, verbose=2)

    print( i, image_file, gtruth, label_map[classes[0]] )
    
    if label_map[classes[0]] == gtruth:
        num_correct += 1
    else:
        num_wrong += 1
    
#     print(pred.argmax(), classes[0], label_map[classes[0]], gtruth, image_file)
#     plt.imshow(img)

print ('Right = ' + str(num_correct/(num_correct+num_wrong)*100.0) + '\n' +
       'Wrong = ' + str(num_wrong/(num_correct+num_wrong)*100.0))
