#!/usr/bin/env python

import warnings

#... Supress TensorFlow warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import keras

from keras.preprocessing.image import ImageDataGenerator

from datetime import datetime
import time

datagen = ImageDataGenerator()
train_dir = './data/21st-street/training'
val_dir = './data/21st-street/validation'

start_time = time.time()
print( str(datetime.now()) )

train_gen = datagen.flow_from_directory(
        train_dir,
        ###target_size=(500,500),
        batch_size=32,
        class_mode='categorical',
        follow_links=True
)

val_gen = datagen.flow_from_directory(
        val_dir,
        ###target_size=(500,500),
        batch_size=32,
        class_mode='categorical',
        follow_links=True
)
print(train_gen, val_gen)

print("--- %s seconds ---" % (time.time() - start_time))
