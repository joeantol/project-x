#!/usr/bin/env python

import os
import re
import zipfile

file_dir = '/home/joeantol/work/project-x/joesandroid/gunks/trapps'
os.chdir(file_dir)

for z in os.listdir():
    d = z.split('-')[0]
    
    if os.path.isdir(z): continue
        
    os.makedirs(d, exist_ok=True)
    
    if os.path.isfile(z):
        print('Moving file: ' + z + ' to ' + d)
        os.rename(z, os.path.join(d,z))
        
    print('Extracting: ' + z + '\n')  
    with zipfile.ZipFile(os.path.join(d, z),"r") as zip_ref:
        zip_ref.extractall(d)
        
        for j in os.listdir(os.path.join(d)):
            base, ext = j.split('.')
            
            #... Don't rename zip files
            if ext == 'zip': continue
            
            if re.search('^[0-9]*', j).group():
                os.rename(os.path.join(d,j), os.path.join(d, d + '_' + j))

# file = '20180310_142154_029.jpg'
file = 'doublechin_20180310_142137_030.jpg'
m = re.search('^[0-9]*', file)

if re.search('^[0-9]*', file).group():
    print('Need to rename...')
else:
    print('Already done...')

