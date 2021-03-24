#!/usr/bin/env python

import sys
sys.version

import cv2
import os

# parser = argparse.ArgumentParser()
# parser.add_argument('--file', '-f')

# try:
#     get_ipython().__class__.__name__
#     args = parser.parse_args(['-f ])
#     print('In Jupyter...')
# except:
#     args = parser.parse_args()
#     print('NOT in Jupyter...')

video_dir = './gopro/gunks/trapps/'
# video_dir = './gopro/gunks/trapps/raubenheimerspecial/'

# vidcap = cv2.VideoCapture(os.path.join('./joesgopro/gunks/trapps/apecall', 'apecall.m4v'))
# success, image = vidcap.read()
dirs = [f for f in os.listdir(video_dir) if not os.path.isfile(os.path.join(video_dir, f))]
dirs

dirs = [f for f in os.listdir(video_dir) if not os.path.isfile(os.path.join(video_dir, f))]

# dirs = ['herdiegerdieblock']

for d in dirs:
    for vid in os.listdir(os.path.join(video_dir, d)):
        name = os.path.splitext(vid)[0]
                
        if d != name: continue 

        count = 0

        vidcap = cv2.VideoCapture(os.path.join(video_dir, d, vid))
        success,image = vidcap.read()
        success = True

        while success:
            success,image = vidcap.read()

            if not success: 
                print("Failure to read frame...")
                continue

            fname = os.path.join(video_dir, d, name + '-gopro-' + '{:010}'.format(count) + '.jpg')
            
            if os.path.isfile(fname):
                print('Skipping: ' + fname)
            else:
                print('Saving: ' + fname)
                cv2.imwrite(fname, image)

            count += 1

print('Successful completion of ExtractJpegs...')

