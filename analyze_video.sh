#!/bin/bash

#date

directory=$1
file=$2
resolution=$3

#resolutions="256:144 426:240 640:360 720:480 960:540 1280:720"
qps="1 6 11 16 21 26 31 36 41 46 51"

#echo "Starting Video ${file}"

# ffmpeg -hide_banner -loglevel panic -i "${directory}/${file}.mp4" -vf scale=$resolution -sws_flags lanczos -y "temp/${resolution##*:}_scaled_video.mp4"
# ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_scaled_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upscaled_video.mp4"
#
#ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_upscaled_video.mp4" -i "${directory}/${file}.mp4" -lavfi ssim="ssims/${file}_${resolution##*:}_ref_ssim.log" -f null -

#echo "ssims/${file}_${resolution##*:}_ref_ssim.log"
#echo "Processed reference video ${file} at resolution ${resolution##*:}p"

for qp in $qps
do

echo "${file}, ${resolution}, ${qp}"
ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_scaled_video.mp4" -vcodec libx264 -q $qp -y "temp/${resolution##*:}_comp_video.mp4"
ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_comp_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upcomp_video.mp4"

ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_comp_video.mp4" -i "temp/${resolution##*:}_scaled_video.mp4" -lavfi ssim="ssims/${file}_${resolution##*:}_${qp}_comp_ssim.log" -f null -
ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_upcomp_video.mp4" -i "${directory}/${file}.mp4" -lavfi ssim="ssims/${file}_${resolution##*:}_${qp}_true_ssim.log" -f null -

#echo "ssims/${file}_${resolution##*:}_${qp}_comp_ssim.log"
#echo "ssims/${file}_${resolution##*:}_${qp}_true_ssim.log"

done;

#echo "Finished video ${file}"
