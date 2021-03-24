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

from keras_sequential_ascii import sequential_model_to_ascii_printout

from keras.applications.resnet50 import ResNet50
    
from keras import backend as K

from keras.callbacks import EarlyStopping, History, LambdaCallback, ModelCheckpoint

from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D 
from keras.layers import MaxPooling2D, ZeroPadding2D, BatchNormalization

from keras.models import Sequential, Model
from keras.models import load_model, model_from_json

from keras.optimizers import Adam, RMSprop, SGD

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import sys
import time

sys.stdout.flush()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allow image embeding in notebook
# %matplotlib inline

class ProjectX:
    def __init__ (self, logfile_name='project-x'):
        self.logfile_name = logfile_name
        self.mean = self.std = []
        
    def args(self, args):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--aug', '-a', default=True)
        parser.add_argument('--batchsize', '-b', default=32)
        parser.add_argument('--epochs', '-e', default=500)
        parser.add_argument('--height', '-y')
        parser.add_argument('--imagedir', '-i')
        parser.add_argument('--learningrate', '-l', default=1e-6)
        parser.add_argument('--numimages', '-n')
        parser.add_argument('--opt', '-o', default='Adam')
        parser.add_argument('--projectname', '-p', default='projectx')
        parser.add_argument('--saveprefix', '-s', default='project-x')
        parser.add_argument('--uniqueid', '-u', default=0)
        parser.add_argument('--width', '-x')

        print('Args: ' + ' '.join(args) + '\n')

        args = parser.parse_args(args)

        try:
            get_ipython().__class__.__name__
            self.in_jupyter = True
            print('In Jupyter...')

        except:
            args = parser.parse_args()
            self.in_jupyter = False
            print('NOT in Jupyter...')

        self.data_augmentation = args.aug
        self.batch_size        = int(args.batchsize)
        self.epochs            = int(args.epochs)
        self.image_height      = int(args.height)
        self.image_dir         = os.path.join(os.getcwd(), args.imagedir)
        self.lr                = float(args.learningrate)
        self.opt               = args.opt
        self.project_name      = args.projectname
        self.unique_id         = int(args.uniqueid)
        self.image_width       = int(args.width)
        self.save_prefix       = args.saveprefix

        if not self.in_jupyter:
            log_file = os.path.join(os.getcwd(), 'logs', self.logfile_name + '-' 
                                    + str(self.unique_id) + '.log')
            print("Log file: " + log_file)
            sys.stdout = open(log_file, 'w')

    def get_classes_from_dirs(self):

        self.training_dir   = os.path.join(self.image_dir, 'training')
        self.validation_dir = os.path.join(self.image_dir, 'validation')

        self.classes = []

        for d in os.listdir(self.training_dir):
            if os.path.isdir(os.path.join(self.training_dir, d)):
                self.classes.append(d)

        print("Training dir: " + self.training_dir)
        print("Validation dir: " + self.validation_dir + '\n')
        print("Classes: " + str(self.classes) + '\n')

        self.num_images = 0

        for dirs, subdirs, files in os.walk(self.training_dir):
            for file in files:
                self.num_images += 1

        for dirs, subdirs, files in os.walk(self.validation_dir):
            for file in files:
                self.num_images += 1

        print('Number of images: ' + str(self.num_images) + '\n')

    def create_model_dir(self, program_name):

        cwd = os.getcwd()
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_path = os.path.join(save_dir, self.project_name + '-' + self.opt + '-' + str(self.num_images) + '-' 
                                  + str(self.image_width) + 'x' + str(self.image_height) + '-' + str(self.unique_id))
        os.makedirs(model_path, exist_ok=True)

        #... Make a copy of the main script and library
        if not self.in_jupyter:
            shutil.copy2(__file__, model_path)
            shutil.copy2(program_name, model_path)

        self.model_path = model_path
        
        print('Saving model at: '+ self.model_path)
        
    def create_generators(self, datagen, image_dir):
        
        image_width    = self.image_width
        image_height   = self.image_height
        batch_size     = self.batch_size
                
        gen = datagen.flow_from_directory(
                image_dir,
                target_size=(image_width, image_height),
                batch_size=batch_size,
                class_mode='categorical',
                follow_links=True,
                color_mode='rgb',
                shuffle=True,
                seed=42,
        )
        
        return gen

    def train_the_model (self, model, train_gen, val_gen, epochs=500):

        print("Train the model...")
        
        batch_size = self.batch_size
        model_path = self.model_path
              
        hist = History()
        early_stopping = EarlyStopping(monitor='val_acc', patience=35, verbose=2, mode='auto')
        time_callback = TimeHistory()
        lambda_callback = LambdaCallback(on_batch_end=lambda batch,logs:print(logs))

        checkpoint_file = os.path.join(model_path, 'model.{epoch:02d}-{val_acc:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, 
                                           save_weights_only=False, mode='auto', period=1)
        
        hist = model.fit_generator(
                train_gen,
                steps_per_epoch=train_gen.samples//batch_size + 1,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=val_gen.samples//batch_size + 1,
                use_multiprocessing=True,
                workers=8,
                callbacks=[early_stopping, time_callback, model_checkpoint]
        )

        self.hist = hist
        self.time_callback = time_callback
        self.label_map = (train_gen.class_indices)

    def save_the_model(self, model, val_gen):
        
        model_path    = self.model_path
        hist          = self.hist
        time_callback = self.time_callback
        label_map     = self.label_map

        #... Save the architecture
#         model_json = model.to_json()
#         with open(os.path.join(model_path, 'model_arch.json'), "w") as json_file:
#             json_file.write(model_json)
#         print('Saved model architecture at %s ' % model_path)
            
#         #... Save the weights
#         model.save_weights(os.path.join(model_path, 'model_weights.hdf5'))
#         print('Saved model weights at %s ' % model_path)
            
        #... Save model and weights
        model.save(os.path.join(model_path, 'model.hdf5'))
        print('Saved trained model at %s ' % model_path)

        #... Save history
        with open(os.path.join(model_path, 'history.pk'), 'wb') as f:
            pickle.dump(hist.history, f)
        print('Saved history at %s ' % model_path)

        #... Save epoch times
        with open(os.path.join(model_path, 'times.pk'), 'wb') as f:
            pickle.dump(time_callback.times, f)
        print('Saved epoch times at %s ' % model_path)

        print('Score the trained model...')
        pred = model.predict_generator(val_gen, workers=8, use_multiprocessing=True, verbose=1)
        with open(os.path.join(model_path, 'pred.pk'), 'wb') as f:
            pickle.dump(pred, f)
        print('Saved predictions at %s ' % model_path)

        eval_scores = model.evaluate_generator(val_gen, workers=8, use_multiprocessing=True)
        with open(os.path.join(model_path, 'eval.pk'), 'wb') as f:
            pickle.dump(eval_scores, f)
        print('Saved eval at %s ' % model_path)

        with open(os.path.join(model_path, 'label_map.pk'), 'wb') as f:
            pickle.dump(label_map, f)
        print('Saved label map at %s ' % model_path)

        print('Test loss:', eval_scores[0])
        print('Test accuracy:', eval_scores[1])
        
    def load_the_model(self, model_dir, model_file='model.hdf5', model_arch='model_arch.json', model_weights='model_weights.hdf5'):
        
        model_stats = {}
        
        model_name, opt, num_images, image_size, unique_id = model_dir.split('-')
        width, height = image_size.split('x')
        image_width = int(width)
        image_height = int(height)
        
        model_stats['image_width']  = image_width
        model_stats['image_height'] = image_height
        
        cwd = os.getcwd()
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_path = os.path.join(save_dir, model_dir)
        
        # load json and create model
#         with open(os.path.join(model_path, model_arch), 'r') as json_file:
#             model_json = json_file.read()
#             model = model_from_json(model_json)
#             model_stats['model'] = model
#             print('Loaded model architecture from %s ' % model_path)

#         model.load_weights(os.path.join(model_path, model_weights))
#         model_stats['model_weights'] = model_weights
#         print('Loaded model weights from %s ' % model_path)

        #... Load model and weights
        model = load_model(os.path.join(model_path, model_file))
        model_stats['model'] = model
        print('Loaded trained model from %s ' % model_path)
                        
        with open(os.path.join(model_path, 'eval.pk'), 'rb') as f:
            eval = pickle.load(f)
            model_stats['eval'] = eval
            print('Loaded evals...')
                  
        with open(os.path.join(model_path, 'history.pk'), 'rb') as f:
            hist= pickle.load(f)
            model_stats['hist'] = hist
            print('Loaded history...')
                  
        with open(os.path.join(model_path, 'pred.pk'), 'rb') as f:
            pred = pickle.load(f)
            model_stats['pred'] = pred
            print('Loaded pred...')
                  
        with open(os.path.join(model_path, 'times.pk'), 'rb') as f:
            times = pickle.load(f)
            model_stats['times'] = times
            print('Loaded times...')
                  
        with open(os.path.join(model_path, 'label_map.pk'), 'rb') as f:
            label_map = pickle.load(f)
            model_stats['label_map'] = label_map
            print('Loaded label_map...')
                  
        return model_stats
        
    def sample_mean_and_std(self, batch_size=0):

        training_dir = os.path.join(self.image_dir, 'training')
        image_width  = self.image_width
        image_height = self.image_height
        batch_size   = self.batch_size if batch_size == 0 else batch_size
        
        sample_mean = sample_var = sample_std = 0
        num_batches = num_samples = 0
        
        stats_file = os.path.join(training_dir, 'train_stats.pk')
        
        dg = ImageDataGenerator()

        tg = dg.flow_from_directory(
                training_dir,
                target_size=(image_width, image_height),
                batch_size=batch_size,
                class_mode='categorical',
                follow_links=True,
        )

        if os.path.isfile(stats_file):
            print("Reading existing stats file...")
            with open(stats_file, 'rb') as f:
                num_samples = pickle.load(f)
                sample_mean = pickle.load(f)
                sample_std  = pickle.load(f)
        else:
            print("Stats file does not exist...")
            
        if num_samples > 0 and num_samples != tg.samples:
            print("Number of samples differs between current and stored...") 

        if num_samples != tg.samples:
            
            print("Recalculating training mean and std...")
            
            for img, label in tg:
                sample_mean += img.mean(axis=(1,2), keepdims=True).mean(axis=0, keepdims=True)
                sample_var  += ((img.std(axis=(1,2), keepdims=True))**2).mean(axis=0, keepdims=True)

                num_batches += 1

                print(num_batches, img.mean(), sample_mean/num_batches, 
                      img.std(), (np.sqrt(sample_var / num_batches)))

                #... Generators are in an infinite loop
                if num_batches*batch_size > tg.samples:
                    break

            sample_mean /= num_batches
            sample_std  = (np.sqrt(sample_var / num_batches) + K.epsilon())

            with open(stats_file, 'wb') as f:
                pickle.dump(tg.samples, f)
                pickle.dump(sample_mean, f)
                pickle.dump(sample_std, f)

        return sample_mean, sample_std
    
    def normalize(self, x):
        
        if self.mean.size == 0 or self.std.size == 0:
            raise ValueError('Training mean and/or std is less than zero...')

        x = (x - self.mean)/self.std

        return x

    def greyscale(self, x):
        x[:] = x.mean(axis=-1,keepdims=1)
        
        x = self.normalize(x)
        
        return x

#... TODO: Confustion matrix

class TimeHistory(keras.callbacks.Callback):
    
    import time
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

if __name__ == '__main__':
    print('In main...')

    X = ProjectX()
    
    args = ['-x=224', '-y=224', '-i=data/union/gunks/trapps/trainval', '-u=110', '-l=1e-4']
    
    X.args(args)
        
#     sample_mean, sample_std = X.sample_mean_and_std(batch_size=100)

