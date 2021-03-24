#!/usr/bin/env python

import argparse
import os
import sys

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--imagedir', '-i')
parser.add_argument('--keep', '-k', default=1.0) #... Used to 'down sample' images culled from video
parser.add_argument('--numimages', '-n', default=1e6) #... Mostly used for testing
parser.add_argument('--split', '-s', default=0.2)

try:
    get_ipython().__class__.__name__
    args = parser.parse_args(['-i=data/union/gunks/trapps'])
    print('In Jupyter...')
except:
    args = parser.parse_args()
    print('NOT in Jupyter...')
    
image_dir  = os.path.join(os.getcwd(), args.imagedir.strip())
num_images = args.numimages
split      = args.split
keep       = args.keep

#... NB: Trailing dash is ok
training_dir   = os.path.join(image_dir, 'trainval', 'training' + '-')
validation_dir = os.path.join(image_dir, 'trainval', 'validation' + '-')

image_list = []
class_list = []
i = 0
done = False

class_list = sorted(os.listdir(image_dir))

if 'trainval' in class_list:
    class_list.remove('trainval')
if 'unlabeled' in class_list:
    class_list.remove('unlabeled')

for subdir in class_list:
            
    for file in sorted(os.listdir(os.path.join(image_dir, subdir))):
        i += 1
        
        if i > num_images:
            done = True
            break
                
        if i % (1.0/keep) == 0:
            image_list.append(os.path.join(image_dir, subdir, file))
            
    if done: break

os.makedirs(training_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

class_list

for subdir in class_list:
    print(subdir)
    os.makedirs(os.path.join(training_dir,   subdir), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, subdir), exist_ok=True)

train_samples, validation_samples = train_test_split(image_list, test_size=0.2)
num_samples = len(train_samples) + len(validation_samples)
print(len(train_samples), len(validation_samples))

for src in validation_samples:
    
    label, file = src.split('/')[-2:]
    link = os.path.join(validation_dir, label, file)
    
    if not os.path.islink(link):
        try:
            print(src, link)
            os.symlink(src,link)
        except:
            print('ERROR: Could not create link: ' + src, link)

for src in train_samples:
    label, file = src.split('/')[-2:]
    link = os.path.join(training_dir, label, file)
    
    if not os.path.islink(link):
        try:
            print(src, link)
            os.symlink(src,link)
        except:
            print('ERROR: Could not create link: ' + src, link)

cwd = os.getcwd()
os.chdir(os.path.join(image_dir, 'trainval'))

os.rename(training_dir, training_dir + str(num_samples))
os.rename(validation_dir, validation_dir + str(num_samples))

for link in ['training', 'validation']:
    if os.path.islink(link):
        os.unlink(link)
        
    os.symlink(link + '-' + str(num_samples), link)

os.chdir(cwd)

print('Successful completion of train_test_split...')

