#!/bin/bash

directory=$1
file=$2
resolution=$3

qps="1 6 11 16 21 26 31 36 41 46 51"

for qp in $qps
do

echo "${file}, ${resolution}, ${qp}"
ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_scaled_video.mp4" -vcodec libx264 -q $qp -y "temp/${resolution##*:}_comp_video.mp4"
ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_comp_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upcomp_video.mp4"

ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_comp_video.mp4" -i "temp/${resolution##*:}_scaled_video.mp4" -lavfi ssim="ssims/${file}_${resolution##*:}_${qp}_comp_ssim.log" -f null -
ffmpeg -hide_banner -loglevel panic -i "temp/${resolution##*:}_upcomp_video.mp4" -i "${directory}/${file}.mp4" -lavfi ssim="ssims/${file}_${resolution##*:}_${qp}_true_ssim.log" -f null -

done;

