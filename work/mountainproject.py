#!/usr/bin/env python

# %reset -f

import keras
from keras import callbacks

import numpy as np
import os
import random as rn
import skimage.data
import tensorflow as tf
import time

from PIL import Image

# Allow image embeding in notebook
# %matplotlib inline

def test_this_function(msg = None):
    print(msg)

def load_building_data(image_dir, label_file, num_images=1e100, dim=(100,100)):
    
    import csv
    import os
    import random
    import skimage.data
    import numpy as np
    
    from PIL import Image
    
    """Loads a data set and returns three lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    xref: a list of names that cross refs to label numbers
    """
    
    labels     = []
    images     = []
    label_data = []
    
    i         = 0
    address   = None
    cur_label = None
     
    xref, label_data = load_xref(label_file)
    
    #... Create empty arrays (faster than appending)
    npimages = np.zeros(shape=(num_images+1, dim[0], dim[1], 3))
    nplabels = np.zeros(shape=(num_images+1))
            
    for rec in label_data:
        
        start_time = time.time()

        _, image_file = rec[0].split('-')
        label = rec[2]
        
        image_file = os.path.join(image_dir, image_file)
                
        if os.path.isfile(image_file): 
                        
            image = skimage.data.imread(image_file)
        
            if address != rec[1]:
                address = rec[1]
                xref.append(address)
            
            try:
                if address != cur_label:
                    print('Processing images for ' + address )
                    cur_label = address
                    
                if i % 100 == 0:
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print( '   Loading[' + str(i) + ']: ' + image_file )

                img = np.array( Image.fromarray(image, 'RGB').resize(dim) )

                r = img[:,:,0]
                g = img[:,:,1]
                b = img[:,:,2]

                if i == 0:
                    npimage = np.array([[r] + [g] + [b]], np.uint8)
                    npimages[i] = npimage.transpose(0,2,3,1)
                    
                    nplabels[i] = np.array([label], np.uint8)

                new_array = np.array([[r] + [g] + [b]], np.uint8)
                new_array = new_array.transpose(0,2,3,1)
#                 npimages  = np.append(npimages, new_array, 0) 
                npimages[i] = new_array

                new_label = np.array([label], np.uint8)
#                 nplabels = np.append(nplabels, new_label, 0)
                nplabels[i] = new_label

            except ValueError as e:  #... Need to fix so only catch 'not enough data' error
                print(str(e))
                break
#                 print(' >>>> WARNING: Cannot process image for ' + xref[nplabels[i]] 
#                       + ' ' + str(e) + ' <<<<')

            i += 1
                    
        #... For testing
        if i > num_images:
            break
            
    return npimages, nplabels, xref

def load_xref(image_file):
    import os
    import csv
    
    xref       = []
    label_data = []

    print('Loading cross reference data from ' + image_file + '...')

    with open(image_file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        label_data = list(csv_reader)

        for i in [row[1] for row in label_data]:
            if i not in xref:
                xref.append(i)
                
    return xref, label_data

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

def set_reproducable_results(reproducable_results=True):

    if reproducable_results:
        os.environ['PYTHONHASHSEED'] = '0'

        # The below is necessary for starting Numpy generated random numbers
        # in a well-defined initial state.

        np.random.seed(42)

        # The below is necessary for starting core Python generated random numbers
        # in a well-defined state.

        rn.seed(12345)

        # Force TensorFlow to use single thread.
        # Multiple threads are a potential source of
        # non-reproducible results.
        # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

        tf.set_random_seed(1234)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

def load_data(data_dir):
    
    import os
    
    """Loads a data set and returns three lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    labels_xref: a list of names that cross refs to label numbers
    """

    labels = []
    images = []
    labels_xref = []
    label_num = 0

    dirs = os.listdir(data_dir)

    for climb_name in dirs:
        
        image_dir = os.path.join(data_dir, climb_name)
        
        if climb_name == '.done':
            continue
        
        #... Skip climbs w/ no images
        if os.listdir(image_dir) == []:
            continue

        labels_xref.append(climb_name)

        for image in os.listdir(image_dir):

            print("Loading[" + str(label_num) + ']: ', os.path.join(image_dir, image))
            labels.append(label_num)
            images.append(skimage.data.imread(os.path.join(image_dir, image)))
            
        label_num += 1
            
    return images, labels, labels_xref

def convert_to_cifar(images, labels, labels_xref, dim=(100,100)):

    cur_label = None
        
    for i in range(len(images)):
                
        try:
            if labels_xref[labels[i]] != cur_label:
                print('Processing images for ' + labels_xref[labels[i]] )
                cur_label = labels_xref[labels[i]]
                
            img = np.array( Image.fromarray(images[i], 'RGB').resize(dim) )
            
            r = img[:,:,0]
            g = img[:,:,1]
            b = img[:,:,2]

            if i == 0:
                npimages = np.array([[r] + [g] + [b]], np.uint8)
                npimages = npimages.transpose(0,2,3,1)
                nplabels = np.array([labels[i]], np.uint8)

            new_array = np.array([[r] + [g] + [b]], np.uint8)
            new_array = new_array.transpose(0,2,3,1)
            npimages = np.append(npimages, new_array, 0) 
            
            new_label = np.array([labels[i]], np.uint8)
            nplabels = np.append(nplabels, new_label, 0)
            
        except ValueError as e:  #... Need to fix so only catch 'not enough data' error
            print(' >>>> WARNING: Cannot process image for ' + labels_xref[labels[i]] + ' ' + str(e) + ' <<<<')
                              
    return npimages, nplabels

def display_images_and_labels(images, labels, labels_xref):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        
        #... Pick the first image for each label.
        image = images[labels.index(label)]
        
        #... 
        plt.subplot(5, 5, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        ###plt.title("Label {0} ({1})".format(label, labels.count(label)))
        plt.title("({1} [{0}])".format(label, labels_xref[labels.count(label)]))
        i += 1
        _ = plt.imshow(image)
    plt.show()

class TimeHistory(keras.callbacks.Callback):
    
    import time
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

