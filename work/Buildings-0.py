#!/usr/bin/env python

### %reset -f

###from __future__ import print_function

import argparse
import ast
import csv
import datetime

import keras
from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.callbacks import History 

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

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
# %matplotlib inline

parser = argparse.ArgumentParser()
parser.add_argument('--aug', '-a', default=False)
parser.add_argument('--height', '-y')
parser.add_argument('--width', '-x')
parser.add_argument('--learningrate', '-l', default=0.001)
parser.add_argument('--loadsavedimages', '-s', default='False')
parser.add_argument('--numimages', '-n')
parser.add_argument('--opt', '-o')
parser.add_argument('--uniqueid', '-u', default=0)

try:
    get_ipython().__class__.__name__
    args = parser.parse_args(['-x', '100', '-y', '100', '-n', '1000', '-o', 'Adam', '-s=True'])
    print('In Jupyter...')
except:
    args = parser.parse_args()
    print('NOT in Jupyter...')

data_augmentation = args.aug
image_width       = int(args.width)
image_height      = int(args.height)
lr                = float(args.learningrate)
load_saved_images = ast.literal_eval(args.loadsavedimages)
num_images        = int(args.numimages)
opt               = args.opt
unique_id         = int(args.uniqueid)

batch_size = 32
epochs = 100

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, '21stStreet' + '-' + opt + '-' + str(num_images) + '-' 
                          + str(image_width) + 'x' + str(image_height) + '-' + str(unique_id))
os.makedirs(model_path, exist_ok=True)
print('Saving model at: '+ model_path)

mp.set_reproducable_results(False)

#... Create a bunch of optimizer objects for later use
sgd = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
RMSprop = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-6, decay=0.0)
Adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.0)

opts = {'Adam'    : Adam,
        'RMSprop' : RMSprop,
        'SGD'     : sgd
       }

reload(mp)

image_dir = os.path.join(cwd, 'buildings/21st-street')
label_file = os.path.join(cwd, 'buildinglabels.csv')
pre_processed_images_dir = os.path.join(cwd, 'pre_processed_images')

xref = []

if load_saved_images:
    xref, _ = mp.load_xref(label_file)
    
    images = np.load(os.path.join(pre_processed_images_dir, 'images-' + str(num_images) + '.npy'))
    labels = np.load(os.path.join(pre_processed_images_dir, 'labels-' + str(num_images) + '.npy'))
    print('Loaded saved images...')

else:
    #... The labels file can't be in the image dir since it's cloud storage, not file system
    images, labels, xref = mp.load_building_data(image_dir, './buildinglabels.csv', 
                                                 num_images=num_images, 
                                                 dim=(image_width,image_height))

    #... Serialize 'cause loading and processing the images takes ages
    np.save(os.path.join(pre_processed_images_dir, 'images-' + str(num_images)), images)
    np.save(os.path.join(pre_processed_images_dir, 'labels-' + str(num_images)), labels)
    
x = images
y = labels

images.shape

x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('')
print('y_train shape:', y_train.shape)
print(y_train.shape[0], 'train labels')
print(y_test.shape[0], 'test labels')

x_train.shape[1:]

num_classes = len(xref)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

filters = 32
model.add(Conv2D(filters, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))
model.add(Conv2D(filters, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters*2, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters*2, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(filters*16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=opts[opt],
              metrics=['accuracy'])

opts[opt]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

hist = History()
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
time_callback = mp.TimeHistory()

data_augmentation = False
if not data_augmentation:
    print('Not using data augmentation...')
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test),
                     shuffle=True,
                     callbacks=[early_stopping, time_callback]
                    )

else:
    print('Using real-time data augmentation.')
    #... This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    hist = model.fit_generator(datagen.flow(x_train, y_train,
                               batch_size=batch_size),
                               epochs=epochs,
                               validation_data=(x_test, y_test),
                               workers=4,
                               callbacks=[early_stopping, time_callback]
                              )

#... Save model and weights
model.save(os.path.join(model_path, 'model.h5'))
print('Saved trained model at %s ' % model_path)

#... Save history
with open(os.path.join(model_path, 'history.pk'), 'wb') as f:
    pickle.dump(hist.history, f)
print('Saved history at %s ' % model_path)

#... Save epoch times
with open(os.path.join(model_path, 'times.pk'), 'wb') as f:
    pickle.dump(time_callback.times, f)
print('Saved epoch times at %s ' % model_path)

#... Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
with open(os.path.join(model_path, 'scores.pk'), 'wb') as f:
    pickle.dump(scores, f)
print('Saved scores at %s ' % model_path)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

