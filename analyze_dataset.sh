#!/bin/bash

directory=$1

pids=""
RESULT=0

mkdir -p ssims
mkdir -p temp

resolutions="256:144 426:240 640:360 720:480 960:540 1280:720"

for resolution in $resolutions
do
    ./analyze_scale.sh $directory $resolution &
    pids="${pids} $!"

done;

for pid in $pids
do
    wait $pid || let "RESULT=1"
done;

if [ "$RESULT" == "1" ];
    then
       exit 1
fi
