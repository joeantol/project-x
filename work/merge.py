#!/usr/bin/env python

import argparse
import os

data_dir    = os.path.join(os.getcwd(), 'data')

args = ['-t=gunks/trapps', '-a=1.0', '-g=0.2', '-m=union']
print('Args: ' + ' '.join(args) + '\n')

parser = argparse.ArgumentParser()

parser.add_argument('--akeep', '-a', default=1.0)
parser.add_argument('--gkeep', '-g', default=1.0)
parser.add_argument('--merge', '-m', default='union')
parser.add_argument('--targetdir', '-t')

try:
    get_ipython().__class__.__name__
    args = parser.parse_args(args)
    print('In Jupyter..\n')
except:
    args = parser.parse_args()
    print('NOT in Jupyter...\n')

akeep = float(args.akeep)
gkeep = float(args.gkeep)
merge = args.merge #... Used to name dir with merged image files

target_dir   = os.path.join(os.path.join(data_dir, merge), args.targetdir.strip())
android_dir  = os.path.join(os.path.join(data_dir, 'android'), args.targetdir.strip())
gopro_dir    = os.path.join(os.path.join(data_dir, 'gopro'), args.targetdir.strip())

print("Target Dir: " + target_dir)
print("Android Dir: " + android_dir)
print("GoPro Dir:" + gopro_dir)

android_dirs = sorted(os.listdir(android_dir))
gopro_dirs   = sorted(os.listdir(gopro_dir))

if os.path.exists(os.path.join(android_dir, 'trainval')):
    android_dirs.remove('trainval')
if os.path.exists(os.path.join(android_dir, 'unlabeled')):
    android_dirs.remove('unlabeled')
if os.path.exists(os.path.join(gopro_dir, 'trainval')):
    gopro_dirs.remove('trainval')
    
#... Intersection or union
if merge == 'inter':
    target_dirs = sorted(list(set(android_dirs).intersection(gopro_dirs)))
elif merge == 'union' :
    target_dirs = sorted(list(set(android_dirs).union(gopro_dirs)))
else:
    print("The --merge option must be 'inter' or 'union'")
    exit
    
for d in target_dirs:
    os.makedirs(os.path.join(target_dir, d), exist_ok=True)

for addr in android_dirs:
    
    i = 0
    
    if not addr in target_dirs: continue
        
    address_dir = os.path.join(target_dir, target_dirs[target_dirs.index(addr)])
    src_dir    = os.path.join(android_dir, addr)
    
    for jpg in os.listdir(os.path.join(android_dir, addr)):
        
        i += 1
        
        if i % (1.0/akeep) == 0:
            link = os.path.join(address_dir, jpg)
            src  = os.path.join(src_dir, jpg)

            if not os.path.islink(link):
                try:
                    print(src, link)
                    os.symlink(src, link)
                except:
                    print('ERROR: Could not create link: ' + src, link)

for addr in gopro_dirs:
    
    i = 0
    
    if not addr in target_dirs: continue
    
    address_dir = os.path.join(target_dir, target_dirs[target_dirs.index(addr)])
    src_dir    = os.path.join(gopro_dir, addr)
    
    for jpg in os.listdir(os.path.join(gopro_dir, addr)):
        
        i += 1
        
        if i % (1.0/gkeep) == 0:
            link = os.path.join(address_dir, jpg)
            src  = os.path.join(src_dir, jpg)

            if not os.path.islink(link):
                try:
                    print(src, link)
                    os.symlink(src, link)
                except:
                    print('ERROR: Could not create link: ' + src, link)

print("Successful completion of merge...")

