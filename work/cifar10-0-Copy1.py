#!/usr/bin/env python

#... https://www.kaggle.com/c/cifar-10/discussion/40237

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np

import os

from keras.callbacks import EarlyStopping

cwd = os.getcwd()
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, 'cifar10')
os.makedirs(model_path, exist_ok=True)
print('Saving model at: '+ model_path)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))

# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7)

x_train = (x_train-mean)
x_test = (x_test-mean)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

baseMapNum = 32
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), 
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)

#training
batch_size = 64
epochs=25

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,
                    validation_data=(x_test,y_test)
                   )

# model.save_weights('cifar10_normal_rms_ep75.h5')
model.save(os.path.join(model_path,'cifar10_normal_rms_ep75_mean.h5'))

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

# opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
# model.compile(loss='categorical_crossentropy',
#         optimizer=opt_rms,
#         metrics=['accuracy'])
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=x_train.shape[0] // batch_size,epochs=4*epochs,verbose=1,
#                     validation_data=(x_test,y_test)
#                    )

# # model.save_weights('cifar10_normal_rms_ep100.h5')
# model.save(os.path.join(model_path,'cifar10_normal_rms_ep100.h5'))

# scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
# print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

# opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
# model.compile(loss='categorical_crossentropy',
#         optimizer=opt_rms,
#         metrics=['accuracy'])
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=x_train.shape[0] // batch_size,epochs=5*epochs,verbose=1,
#                     validation_data=(x_test,y_test)
#                    )

# # model.save_weights('cifar10_normal_rms_ep125.h5')
# model.save(os.path.join(model_path,'cifar10_normal_rms_ep125.h5'))

#testing - no kaggle eval
# scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
# print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

