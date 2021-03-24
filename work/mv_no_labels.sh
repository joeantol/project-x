#!/bin/sh

cd /home/joeantol/work/Project-X/buildings/21st-street

for i in `find -maxdepth 1 -name '*.jpg'`
do
   echo $i
   mv $i /home/joeantol/work/Project-X/buildings/21st-street-nolabels
done

exit
