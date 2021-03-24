#!/usr/bin/env python

from __future__ import print_function

import sys
sys.version

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

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
    
from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.callbacks import History 
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint

from keras.datasets import cifar10

from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, GlobalAveragePooling2D

from keras.models import Sequential, Model

from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

from keras_sequential_ascii import sequential_model_to_ascii_printout

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import sys

sys.stdout.flush()

import convnet_image_utils
from importlib import reload
reload(convnet_image_utils)
from convnet_image_utils import ProjectX

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allow image embeding in notebook
# %matplotlib inline

def pretrained_model(input_model, opt, num_classes, shape=(224,224,3), verbose=False):
        
    if verbose: input_model.summary()
        
    x = input_model.layers[-2].output

    predictions = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=input_model.input, outputs=predictions)

    for layer in input_model.layers:
        layer.trainable = False
    
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if verbose: model.summary()

    return model

def setup_to_transfer_learn(model, base_model):
    
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, num_classes):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    
    predictions = Dense(num_classes, activation='softmax')(x) #new softmax layer

    model = Model(input=base_model.input, output=predictions)

    return model

def fine_tune_vgg16(verbose=False, shape=(224,224,3)):
    
    image_input = Input(shape=shape)

    model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')

    if verbose: model.summary()

    last_layer = model.get_layer('block5_pool').output
    
    x= Flatten(name='flatten')(last_layer)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(len(X.classes), activation='softmax', name='output')(x)
    
    custom_vgg_model = Model(image_input, out)
    
    if verbose: custom_vgg_model.summary()

    #... Freeze all the layers except the dense layers
    for layer in custom_vgg_model.layers[:-3]:
            layer.trainable = False

    if verbose: custom_vgg_model.summary()

    custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    
    return custom_vgg_model

#... Fine tune the resnet 50  
def fine_tune_resnet(verbose=False):
    
    image_input = Input(shape=(X.image_width, X.image_height, 3))
    model = ResNet50(weights='imagenet',include_top=False)
    
    if verbose: model.summary()
    
    last_layer = model.output

    #... Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(last_layer)

    #... Add fully-connected & dropout layers
    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='fc-2')(x)
    x = Dropout(0.5)(x)

    out = Dense(len(X.classes), activation='softmax',name='output_layer')(x)

    #... This is the model we will train
    custom_resnet_model = Model(inputs=model.input, outputs=out)

    if verbose: custom_resnet_model.summary()

    for layer in custom_resnet_model.layers[:-6]:
        layer.trainable = False

    custom_resnet_model.layers[-1].trainable

    custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return custom_resnet_model

def training_resnet_classifier_alone(verbose=False):

    image_input = Input(shape=(X.image_height, X.image_width, 3))
    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    
    if verbose: model.summary()

    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(len(X.classes), activation='softmax', name='output_layer')(x)

    custom_resnet_model = Model(inputs=image_input, outputs=out)
    
    if verbose: custom_resnet_model.summary()

    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable = False

    custom_resnet_model.layers[-1].trainable

    custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    return custom_resnet_model

def image_data_generator (X):
    
#     preproc_func = X.normalize

    #... Use pre-proc function from Keras
    preproc_func = preprocess_input
    
    #... Per Stanford,we want to zero-mean, but not normalize variance or do PCA or whitening
    if X.data_augmentation:
        print("Using data augmentation...")
        train_datagen = ImageDataGenerator(
                                           rotation_range = 30,
                                           width_shift_range = 0.3,
                                           height_shift_range = 0.3,
                                           zoom_range = 0.25,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           featurewise_center=False,
                                           featurewise_std_normalization=False,
                                           preprocessing_function = preproc_func,

                                          )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function = preproc_func,)

    test_datagen = ImageDataGenerator(preprocessing_function = preproc_func,)
    
    return train_datagen, test_datagen

#... Create a bunch of optimizer objects for later use
def optimizers(lr):

    sgd = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.001, nesterov=False)
    RMSprop = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-6, decay=0.001)
    Adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.001)

    opts = {'Adam'    : Adam,
            'RMSprop' : RMSprop,
            'SGD'     : sgd
           }
    
    return opts

def main():

    args = ['-x=224', '-y=224', '-i=data/union/gunks/trapps/trainval', '-u=112', '-l=1e-4', 
            '-p=transfer_learning_union_VGG16']
    
    try:
        logfile_name = os.path.splitext(__file__)[0]
        program_name = __file__
    except:
        logfile_name='projectx'
        program_name = 'climbatron-1.py'

    X = ProjectX(logfile_name=logfile_name)

    X.args(args)
    X.get_classes_from_dirs()
    X.create_model_dir(program_name=program_name)
    
    #... Make sure we only read and/or calculate once
    X.mean, X.std = X.sample_mean_and_std(batch_size=500)

    opts = optimizers(X.lr)
    
    train_datagen, val_datagen = image_data_generator(X)

    train_gen = X.create_generators(train_datagen, os.path.join(X.image_dir, 'training'))
    val_gen   = X.create_generators(val_datagen, os.path.join(X.image_dir, 'validation'))
    
    model_args = {'include_top' : False, 'weights' : 'imagenet'}
    base_model = VGG16(model_args)
    
    model = add_new_last_layer(base_model, X.num_classes)

#     model = pretrained_model(input_model, opts['Adam'], train_gen.num_classes, verbose=True)

#     X.train_the_model(model, train_gen, val_gen, epochs=X.epochs)
#     X.save_the_model(model, val_gen)
    
def main_loop():
    args = ['-x=224', '-y=224', '-i=data/union/gunks/trapps/trainval', '-u=111', '-l=1e-4', 
            '-p=transfer_learning_union']
    
    try:
        logfile_name = os.path.splitext(__file__)[0]
        program_name = __file__
    except:
        logfile_name='projectx'
        program_name = 'climbatron-1.py'

    X = ProjectX(logfile_name=logfile_name)

    X.args(args)
    X.get_classes_from_dirs()
    
    #... Make sure we only read and/or calculate once
    X.mean, X.std = X.sample_mean_and_std(batch_size=500)

    opts = optimizers(X.lr)

    train_datagen, val_datagen = image_data_generator(X)

    train_gen = X.create_generators(train_datagen, os.path.join(X.image_dir, 'training'))
    val_gen   = X.create_generators(val_datagen, os.path.join(X.image_dir, 'validation'))
    
    project_name = X.project_name
    
    model_args = {'include_top' : False, 
                  'weights'     : 'imagenet', 
                  'input_shape' : (X.image_width, X.image_height, 3)
                 }
    
    base_models = {
              'InceptionResNetV2' : InceptionResNetV2(model_args),
              'InceptionV3'       : InceptionV3(model_args),
              'Xception'          : Xception(model_args),
              'MobileNet'         : MobileNet(model_args),
              'VGG16'             : VGG16(model_args),
              'VGG19'             : VGG19(model_args),    
              'ResNet50'          : ResNet50(model_args),
             }

    for m_name in models:

        X.project_name = project_name + '_' + m_name
        X.create_model_dir(program_name=program_name)

        model = pretrained_model(models[m_name], opts['Adam'], train_gen.num_classes, verbose=True)

        X.train_the_model(model, train_gen, val_gen, epochs=X.epochs)
        X.save_the_model(model, val_gen)
        
        print("Successful completion of " + m_name +  "...")
  
model_args = {'include_top' : False, 
              'weights'     : 'imagenet', 
              'input_shape' : (224,224,3)}

base_models = {
          'InceptionResNetV2' : InceptionResNetV2(model_args),
          'InceptionV3'       : InceptionV3(model_args),
          'Xception'          : Xception(model_args),
#           'MobileNet'         : MobileNet(model_args),
          'VGG16'             : VGG16(model_args),
          'VGG19'             : VGG19(model_args),    
          'ResNet50'          : ResNet50(model_args),
         }

for m_name in base_models:
    print(m_name)

    base_model = base_models[m_name]

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(int(x.shape[-1]), activation='relu')(x) 
    predictions = Dense(47, activation='softmax')(x) 
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(model.name)

# model.summary()

m = MobileNet()

if __name__ == '__main__':
    main()
#     main_loop()
    
    print("Successful completion of climbatron-1 ...")

