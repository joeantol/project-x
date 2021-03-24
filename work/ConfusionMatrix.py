#!/usr/bin/env python

from __future__ import print_function

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

import skimage.data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf

import os

from keras_sequential_ascii import sequential_model_to_ascii_printout

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

sys.stdout.flush()

import convnet_image_utils
from importlib import reload
reload(convnet_image_utils)
from convnet_image_utils import ProjectX

# Allow image embeding in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

X = ProjectX()

model_dir = 'transfer_learning_union_VGG16_bw-Adam-5794-224x224-109'
# model_dir = 'transfer_learning_union_VGG16-Adam-5791-224x224-108'
validation_dir = '/home/joeantol/joeantolwork/project-x/data/union/gunks/trapps/trainval/validation'

model, eval, hist, pred, times, label_map = X.load_the_model(model_dir)

eval

preproc_func = X.normalize

# datagen = ImageDataGenerator(preprocessing_function = preproc_func,)
datagen = ImageDataGenerator()

val_gen = datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        follow_links=True
)

# vg.reset()
labels = list(sorted(val_gen.class_indices.values()))

pred = model.predict_generator(val_gen, verbose=1)

eval_scores = model.evaluate_generator(val_gen, workers=8, use_multiprocessing=True)
eval_scores

labels

# pred.argmax(1)
# p = pred[0:5]
# p.argmax(axis=1), p

# labels = list(validation_generator.class_indices.keys())
# type(labels)

# validation_generator.classes, pred.argmax(axis=1)

# print(pred[0:10].argmax(axis=1))
# print(validation_generator.filenames[0:10])

p = pred.argmax(axis=1)
pred

# confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

cm = confusion_matrix(val_gen.classes, pred.argmax(axis=1), labels=labels)
print(cm)

# confusion = tf.confusion_matrix(labels=y_, predictions=y, num_classes=num_classes)

con = tf.confusion_matrix(val_gen.classes, pred.argmax(axis=1))

sess = tf.Session()
with sess.as_default():
        print(sess.run(con))

