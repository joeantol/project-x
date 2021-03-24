#!/usr/bin/env python

from __future__ import print_function

import sys
sys.version

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

from random import shuffle, sample

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
    args = parser.parse_args(['-d', 'transfer_learning_VGG16-Adam-3568-224x224-107',
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
validation_dir = '/home/joeantol/work/project-x/data/android/gunks/trapps/trainval/validation'
batch_size = 32

model_name, opt, num_images, image_size, unique_id = model_dir.split('-')
image_width, image_height = image_size.split('x')
image_width = int(image_width)
image_height = int(image_height)

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

batch_size = 1
# validation_dir = '/home/joeantol/joeantolwork/project-x/data/burstsnap/21st-street'
# validation_dir = '/home/joeantol/work/project-x/data/doors/21st-street/trainval/training'
# validation_dir = '/home/joeantol/joeantolwork/project-x/data/merged/21st-street/trainval/validation'
# validation_dir = '/home/joeantol/work/project-x/data/doors/testimages'

dg = ImageDataGenerator(rescale=1./255)
vg = dg.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        follow_links=True
)

# label_map = (vg.class_indices)
# label_map = {v: k for k, v in label_map.items()}

filenames = vg.filenames

for f in filenames:
    print(f)

def norm(x):
    
    mean = [129.58818, 125.71938, 126.04443]
    std  = [52.279366, 54.35651 , 56.581802]

    x = (x - mean)/std
    
    return x

# cwd = os.getcwd()
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_path = os.path.join(save_dir, model_dir)

# #... Load model and weights
# model = load_model(os.path.join(model_path, 'model.h5'))
# print('Loaded trained model from %s ' % model_path)

image_dir = '/home/joeantol/work/project-x/data/testimages/gunks/trapps'
# image_file = os.path.join(image_dir, 'missbailey_20180310_152947_020.jpg')
image_file = os.path.join(image_dir, 'missbailey-1.jpg')
gtruth = image_file.split('/')[-2]

image = Image.open(image_file).convert("RGB").resize((image_width, image_height))
# image = Image.open(image_file).convert("RGB").rotate(-90).resize((image_width, image_height))

img = np.array(image)
# img = img * 1./255.
img = norm(img)

r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

npimages = np.array([[r] + [g] + [b]], np.float32)

npimages = npimages.transpose(0,2,3,1)

# classes = model.predict_classes(npimages)
prediction = model.predict(npimages, verbose=2)

# print(label_map[classes[0]], gtruth)

prediction, label_map[prediction.argmax()], plt.imshow(image)

image

#... Use with 'testimages' (aka still photos taken with Android)
# image = Image.open(image_file).convert("RGB").rotate(-90).resize((width, height))
# full_image = Image.open(image_file).convert("RGB").rotate(-90)

#... Use with images scraped from video (either GoPro or Android)
# image = Image.open(image_file).convert("RGB").resize((image_width, image_height))
# image = Image.open(image_file).resize((width, height))

num_correct = num_wrong = 0
filenames = vg.filenames
# filenames = sample(filenames, 100)

for f in filenames:
    gtruth = f.split('/')[0]
    image_file = os.path.join(validation_dir, f)
    image = Image.open(image_file).convert("RGB").resize((image_width, image_height))

    img = np.array(image)
    img = img * 1./255.

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    npimages = np.array([[r] + [g] + [b]], np.float32)

    npimages = npimages.transpose(0,2,3,1)

    classes = model.predict_classes(npimages)
    prediction = model.predict(npimages, verbose=2)
    
    if label_map[classes[0]] == gtruth:
        num_correct += 1
        print('Correct: ' + label_map[classes[0]] + ' == ' + gtruth + ' ' + f)
    else:
        num_wrong += 1
        print('Wrong: ' + label_map[classes[0]] + ' != ' + gtruth + ' ' + f)

print ('Right = ' + str(num_correct/(num_correct+num_wrong)*100.0) + '\n' +
       'Wrong = ' + str(num_wrong/(num_correct+num_wrong)*100.0))

