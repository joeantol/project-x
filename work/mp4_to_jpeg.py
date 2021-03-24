#!/usr/bin/env python

import cv2
print(cv2.__version__)

mp4file = '/home/joeantol/work/Project-X/joesandroid/21st-street-0.mp4'
vidcap = cv2.VideoCapture(mp4file)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
