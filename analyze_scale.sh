#!/bin/bash

directory=$1
resolution=$2

echo "Starting resolution ${resolution}"

for file in "${directory}/*.mp4"
do
    filename="${file##*/}"
    filename="${filename%.*}"

    ffmpeg -hide_banner -loglevel panic -i "${directory}/${filename}.mp4" -vf scale=$resolution -sws_flags lanczos -y "temp/${resolution##*:}_scaled_video.mp4"
    ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_scaled_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upscaled_video.mp4"

    ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_upscaled_video.mp4" -i "${directory}/${filename}.mp4" -lavfi ssim="ssims/${filename}_${resolution##*:}_ref_ssim.log" -f null -

    ./analyze_video.sh $directory $filename $resolution

done;
echo "Finished resolution ${resolution}"
