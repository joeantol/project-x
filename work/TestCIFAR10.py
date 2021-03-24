#!/usr/bin/env python

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
from keras.utils import np_utils

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

# Allow image embeding in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_dir = 'cifar10'
model_path = os.path.join(save_dir, model_dir)

model = load_model(os.path.join(model_path, 'cifar10_normal_rms_ep75_mean_std.h5'))
print('Loaded trained model from %s ' % model_path)

label_map = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
width = height = 32

cifar = True

if cifar:
    image_dir = 'data/testimages/cifar10'
else:
    image_dir = 'data/testimages/non-cifar10'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

images = []
object_list = os.listdir(image_dir)
# object_list = ['airplanes', 'cars', 'birds', 'cats', 'dogs']

for obj in object_list:
    objs = [obj+'/' + s for s in os.listdir(os.path.join(image_dir, obj))]
    images += objs

shuffle(images)

gtruths = [s.split('/')[-2] for s in images]

x_test = np.empty(shape=(10000, 32, 32, 3), dtype=np.float64)
x_test.shape

num_correct = num_wrong = 0

x_test = np.empty(shape=(10000, 32, 32, 3), dtype=np.float64)

for i in range(0,10000):
    
    image_file = images[i]
    gtruth     = gtruths[i]
         
    image = Image.open(os.path.join(image_dir, image_file)).convert("RGB").resize((width, height))
    img = np.array(image)
            
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

#     npimage = np.array([[r] + [g] + [b]], np.uint8)
    npimage = np.array([[r] + [g] + [b]], np.float32)

    npimage = npimage.transpose(0,2,3,1)
    
    np.append(x_test, npimage)
    
    npimage = (npimage-mean)/(std+1e-7)

    classes = model.predict_classes(npimage)
    pred    = model.predict(npimage, verbose=2)
            
    if label_map[classes[0]] == gtruth:
        num_correct += 1
    else:
        num_wrong += 1

print ('Right = ' + str(num_correct/(num_correct+num_wrong)*100.0) + '\n' +
       'Wrong = ' + str(num_wrong/(num_correct+num_wrong)*100.0))

x_test_1 = (x_test_1-mean)/(std+1e-7)

pred_1   = model.predict_classes(x_test_1, verbose=1)

np.mean(x_test), np.std(x_test), np.mean(x_test_1), np.std(x_test_1)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))

x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

# x_train = (x_train-mean)
# x_test = (x_test-mean)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
pred   = model.predict_classes(x_test, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

x_test_1.shape

label_map = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
cifar_dir = 'data/testimages/cifar10'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for i in range(0, len(y_test)):
    img = Image.fromarray(x_test[i])
               
    obj_dir = os.path.join(cifar_dir, label_map[y_test[i][0]])
    os.makedirs(obj_dir, exist_ok=True)
               
    img.save(os.path.join(obj_dir, str(i)+'.jpg'))

