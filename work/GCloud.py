#!/usr/bin/python

import csv
import os

import mountainproject as mp

from importlib import reload
reload(mp)

src_dir = '/home/joeantol/work/project-x/joesandroid/21st-street'
target_dir = os.path.join(src_dir, 'trainval')

os.chdir(src_dir)
cwd = os.getcwd()

label_file = os.path.join(cwd, 'buildinglabels.csv')
xref, labels_csv = mp.load_xref(label_file)

# os.chdir(os.path.join(cwd, 'trainval'))
# cwd = os.getcwd()

for rec in labels_csv:
#     _, file     = rec[0].split('-')
#     address_dir = rec[2] + '-' + rec[1]
    
    file = rec[0]
    address_dir = rec[1]
        
    address_dir = os.path.join(cwd, 'trainval', address_dir)
        
    if not os.path.exists(address_dir):
        os.makedirs(address_dir)
        
    src    = os.path.join(cwd, file)
    target = os.path.join(cwd, address_dir, file)
            
    if not os.path.exists(target):
        print('Moving ' + src + ' >>> ' + target)
        os.rename(src, target)
    else:
        print('Already moved.  Skipping: ' + target)
        
    #... Testing    
#     if rec[1] == '002-204W21': break

