#!/usr/bin/env python

import sys
sys.version

### %reset -f

###from __future__ import print_function

import argparse
import ast
import csv
import datetime
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import keras
    
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    
from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.callbacks import History 
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint

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
parser.add_argument('--batchsize', '-b', default=32)
parser.add_argument('--height', '-y')
parser.add_argument('--imagedir', '-i')
parser.add_argument('--learningrate', '-l', default=1e-6)
parser.add_argument('--numimages', '-n')
parser.add_argument('--opt', '-o')
parser.add_argument('--uniqueid', '-u', default=0)
parser.add_argument('--width', '-x')

try:
    get_ipython().__class__.__name__
    args = parser.parse_args(['-x 100', '-y 100', '-o=Adam', 
                              '-a=False', '-u 4'])
    print('In Jupyter...')
except:
    args = parser.parse_args()
    print('NOT in Jupyter...')
    
print(args)

data_augmentation = args.aug
batch_size        = int(args.batchsize)
image_height      = int(args.height)
image_dir         = os.path.join(os.getcwd(), args.imagedir)
lr                = float(args.learningrate)
opt               = args.opt
unique_id         = int(args.uniqueid)
image_width       = int(args.width)

# image_dir      = '/home/joeantol/work/project-x/data/21st-street/trainval/'
training_dir   = os.path.join(image_dir, 'training')
validation_dir = os.path.join(image_dir, 'validation')

# training_dir   = '/home/joeantol/work/project-x/data/gopro/21st-street'
# validation_dir = '/home/joeantol/work/project-x/data/android/21st-street'

classes = []

for d in os.listdir(training_dir):
    if os.path.isdir(os.path.join(training_dir, d)):
        classes.append(d)
        
# print("Image dir: " + image_dir)
print("Training dir: " + training_dir)
print("Validation dir: " + validation_dir)
print("Classes: " + str(classes))

num_images = 0

for dirs, subdirs, files in os.walk(training_dir):
    for file in files:
        num_images += 1

for dirs, subdirs, files in os.walk(validation_dir):
    for file in files:
        num_images += 1

print('Number of images: ' + str(num_images))

batch_size = 32
epochs = 500

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, '21stStreet' + '-' + opt + '-' + str(num_images) + '-' 
                          + str(image_width) + 'x' + str(image_height) + '-' + str(unique_id))
os.makedirs(model_path, exist_ok=True)
print('Saving model at: '+ model_path)

mp.set_reproducable_results(False)

#... Create a bunch of optimizer objects for later use
sgd = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.001, nesterov=False)
RMSprop = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-6, decay=0.001)
Adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.001)

opts = {'Adam'    : Adam,
        'RMSprop' : RMSprop,
        'SGD'     : sgd
       }

print("Build the model...")

model = Sequential()

filters = 32
model.add(Conv2D(filters, (3, 3), padding='same', input_shape=(image_width, image_height, 3)))

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
model.add(Dense(len(classes)))
model.add(Activation('softmax'))

print("Compiling...")
model.compile(loss='categorical_crossentropy',
              optimizer=opts[opt],
              metrics=['accuracy'])

print("Create training and validation generators...")

#... Per Stanford,we want to zero-mean, but not normalize variance or do PCA or whitening
if data_augmentation:
    print("Using data augmentation...")
    train_datagen = ImageDataGenerator(
                                       ###rescale = 1./255,
                                       rotation_range = 30,
                                       width_shift_range = 0.3,
                                       height_shift_range = 0.3,
                                       zoom_range = 0.25,
                                       horizontal_flip=True,
                                       vertical_flip=True
                                      )
else:
    train_datagen = ImageDataGenerator(
                                       ###rescale=1./255
                                      )

test_datagen = ImageDataGenerator(
                                  ###rescale=1./255
                                 )

train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        follow_links=True
)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        follow_links=True
)

label_map = (train_generator.class_indices)

train_generator.samples//batch_size + 1, validation_generator.samples//batch_size + 1

print("Train the model...")

hist = History()
early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=2, mode='auto')
time_callback = mp.TimeHistory()
lambda_callback = LambdaCallback(on_batch_end=lambda batch,logs:print(logs))

checkpoint_file = os.path.join(model_path, 'model.{epoch:02d}-{val_acc:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, 
                                   save_weights_only=False, mode='auto', period=1)

hist = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//batch_size + 1,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//batch_size + 1,
        use_multiprocessing=True,
        workers=8,
        callbacks=[early_stopping, time_callback, model_checkpoint]
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
# predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
pred = model.predict_generator(validation_generator, workers=8, use_multiprocessing=True, verbose=1)
with open(os.path.join(model_path, 'pred.pk'), 'wb') as f:
    pickle.dump(pred, f)
print('Saved predictions at %s ' % model_path)

eval_scores = model.evaluate_generator(validation_generator, workers=8, use_multiprocessing=True)
with open(os.path.join(model_path, 'eval.pk'), 'wb') as f:
    pickle.dump(eval_scores, f)
print('Saved eval at %s ' % model_path)

with open(os.path.join(model_path, 'label_map.pk'), 'wb') as f:
    pickle.dump(label_map, f)
print('Saved label map at %s ' % model_path)

print('Test loss:', eval_scores[0])
print('Test accuracy:', eval_scores[1])

#... TODO: Confustion matrix

print("Successful completion of Buildings-1...")

