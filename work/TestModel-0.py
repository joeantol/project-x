#!/usr/bin/env python

import sys
sys.version

###from __future__ import print_function

import argparse
import csv
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import keras
    
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from keras import backend as K
from keras.models import load_model

from keras.callbacks import EarlyStopping
from keras.callbacks import History 

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import rmsprop
from keras.optimizers import SGD

from PIL import Image

from random import shuffle

import skimage.data
from sklearn.model_selection import train_test_split

import os

from keras_sequential_ascii import sequential_model_to_ascii_printout

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

sys.stdout.flush()

import mountainproject as mp

from importlib import reload
reload(mp)

# Allow image embeding in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', '-d')
parser.add_argument('--imagefile', '-f')
parser.add_argument('--learningrate', '-l', default=1e-6)
parser.add_argument('--opt', '-o')

try:
    get_ipython().__class__.__name__
    args = parser.parse_args(['-d', '21stStreet-Adam-4986-200x200-6', 
                              '-f', 'testimages/20180110_100024.jpg'
                             ]
                            )
    in_jupyter = True
    print('In Jupyter...')
except:
    args = parser.parse_args()
    in_jupyter = False
    print('NOT in Jupyter...')

model_dir  = args.modeldir
image_file = args.imagefile

model_name, opt, num_images, image_size, unique_id = model_dir.split('-')
width, height = image_size.split('x')
width = int(width)
height = int(height)

clear_devices=True
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, model_dir)

#... Load model and weights
model = load_model(os.path.join(model_path, 'model.h5'))
print('Loaded trained model from %s ' % model_path)

#... Load history
with open(os.path.join(model_path, 'history.pk'), 'rb') as f:
    hist = pickle.load(f)
print('Loaded history...')

#... Load epoch times
with open(os.path.join(model_path, 'times.pk'), 'rb') as f:
    times = pickle.load(f)
print('Loaded times...')

#... Score trained model.
with open(os.path.join(model_path, 'eval.pk'), 'rb') as f:
    eval_scores = pickle.load(f)
print('Loaded model evals...')

with open(os.path.join(model_path, 'pred.pk'), 'rb') as f:
    pred = pickle.load(f)
print('Loaded model predictions...')

with open(os.path.join(model_path, 'label_map.pk'), 'rb') as f:
    label_map = pickle.load(f)
label_map = {v: k for k, v in label_map.items()}
print('Loaded label map...')

# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# cwd = os.getcwd()
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_path = os.path.join(save_dir, model_dir)

# model = load_model(os.path.join(model_path, 'cifar10_normal_rms_ep125.h5'))
# print('Loaded trained model from %s ' % model_path)

if in_jupyter:

    #... Plot the Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(hist['loss'],'r',linewidth=3.0)
    plt.plot(hist['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(hist['acc'],'r',linewidth=3.0)
    plt.plot(hist['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

#     sequential_model_to_ascii_printout(model)

hist.keys()

# image_file = './bogusimages/aviation.jpg'
# image_file = './bogusimages/tulips.jpg'
# image_file = './testimages/20180130_095954.jpg'
# image_file = 'testimages/20180207_095919.jpg'
# image_file = 'testimages/cat-0.jpg'
# image_file = 'testimages/auto-2.jpg'
# image_file = 'testimages/kagglecatsanddogs/PetImages/Cat/0.jpg'
# image_file = 'data/gopro/21st-street/006-244W21/GOPR54152287.jpg'
# image_file = 'data/gopro/21st-street/031-221W21/GOPR541515739.jpg'
# image_file = 'data/gopro/21st-street/210W21/GOPR541523239.jpg'
# image_file = 'data/android/21st-street/010-228W21/0000003339.jpg'
# image_file = 'data/android/21st-street/026-2128thAve/0000006359.jpg'
# image_file = 'data/gopro/21st-street/228W21/GOPR5415138.jpg'
# image_file = 'data/android/21st-street/247W21/0000008524.jpg'
# image_file = 'data/android/21st-street/243W21/0000009030.jpg'
# image_file = 'data/android/21st-street/268W21/0000006023.jpg'
# image_file = 'testimages/20180110_100207.jpg'
# image_file = 'data/android/21st-street/unlabeled/0000000207.jpg'
# image_file = 'data/android/21st-street/unlabeled/0000002140.jpg'
# image_file = 'data/android/21st-street/trainval/validation/000-200W21/0000000547.jpg'
# image_file = 'data/android/21st-street/trainval/validation/010-228W21/0000003325.jpg'

# image_file = 'data/small/21st-street/200W21/0000000539.jpg'
# image_file = 'data/small/21st-street/trainval/training/200W21/0000000577.jpg'
# image_file = 'data/small/21st-street/trainval/validation/214W21/GOPR541523871.jpg'
image_file = 'data/merged/21st-street/226W21/0000003152.jpg'

# image = skimage.data.imread(image_file)
# img = np.array( Image.fromarray(image, 'RGB').resize((width, height)) )

# image = Image.open(image_file).rotate(-90)
# image = Image.open(image_file)

#... Use with 'testimages' (aka still photos taken with Android)
# image = Image.open(image_file).convert("RGB").rotate(-90).resize((width, height))
# full_image = Image.open(image_file).convert("RGB").rotate(-90)

#... Use with images scraped from video (either GoPro or Android)
image = Image.open(image_file).convert("RGB").resize((width, height))
# image = Image.open(image_file).resize((width, height))

img = np.array(image)
# img = img * 1./255.

r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

npimages = np.array([[r] + [g] + [b]], np.uint8)
npimages = npimages.transpose(0,2,3,1)

classes = model.predict_classes(npimages)
prediction = model.predict(npimages, verbose=2)

print(prediction.argmax())
print(classes)
print(label_map[classes[0]])

plt.imshow(img)

x1 = img / 255.
x1 = x1 - x1.mean()
x1

# x = ((x/255.) - 0.5) * 2.

x2 = ((img/255.) - 0.5) * 2.
x2.mean()

import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

image_file = './testimages/20180110_100046.jpg'
# image_file = 'data/android/21st-street/035-247W21/0000008524.jpg'
# image_file = 'data/android/21st-street/unlabeled/0000002140.jpg'
# image_file = 'data/android/21st-street/unlabeled/0000000207.jpg'

image = cv2.imread(image_file)
img = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
img = np.array(img)

r = img[:,:,2]
g = img[:,:,1]
b = img[:,:,0]

npimage = np.array([[r] + [g] + [b]], np.uint8)
npimage = npimage.transpose(0,2,3,1)

classes = model.predict_classes(npimage)
pred = model.predict(npimage, verbose=2)

print(pred)
print(classes)
print(label_map[classes[0]])

plt.imshow(img)

# label_map = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

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

width = height = 200

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, model_dir)

#... Load model and weights
model = load_model(os.path.join(model_path, 'model.h5'))
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

for i in range(0,100):
    
    image_file = images[i]
    gtruth     = gtruths[i]
        
    image = Image.open(os.path.join(image_dir, image_file)).convert("RGB").resize((width, height))

    img = np.array(image)
#     img = img / 255.0
    
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    npimage = np.array([[r] + [g] + [b]], np.uint8)
    npimage = npimage.transpose(0,2,3,1)

    classes = model.predict_classes(npimage)
    pred = model.predict(npimage, verbose=2)
    
    if label_map[classes[0]] == gtruth:
        num_correct += 1
    else:
        num_wrong += 1
    
#     print(pred.argmax(), classes[0], label_map[classes[0]], gtruth, image_file)
#     plt.imshow(img)

print ('Right = ' + str(num_correct/(num_correct+num_wrong)*100.0) + '\n' +
       'Wrong = ' + str(num_wrong/(num_correct+num_wrong)*100.0))
    
