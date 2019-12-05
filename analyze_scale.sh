#!/bin/bash

directory=$1
resolution=$2

filenames="bus_25fps laser_25fps overbridge_25fps robot_25fps shelf_25fps square_25fps toys_calendar_25fps tractor_25fps train_25fps tube_25fps"
#echo "Starting resolution ${resolution}"
#echo "${directory}"
for filename in $filenames
do
#    filename="${file##*/}"
#    filename="${filename%.*}"

#    echo "${directory}/${filename}.mp4"
#    echo "temp/${resolution##*:}_scaled_video.mp4"
#    echo "temp/${resolution##*:}_upscaled_video.mp4"
#    echo "${directory}/${filename}.mp4"
#    echo "ssims/${filename}_${resolution##*:}_ref_ssim.log"

    ffmpeg -hide_banner -loglevel panic -i "${directory}/${filename}.mp4" -vf scale=$resolution -sws_flags lanczos -y "temp/${resolution##*:}_scaled_video.mp4"
    ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_scaled_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upscaled_video.mp4"

    ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_upscaled_video.mp4" -i "${directory}/${filename}.mp4" -lavfi ssim="ssims/${filename}_${resolution##*:}_ref_ssim.log" -f null -

    ./analyze_video.sh $directory $filename $resolution

done;
echo "Finished resolution ${resolution}"
