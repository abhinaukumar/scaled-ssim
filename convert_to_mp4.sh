#!/bin/bash
data_dir=$1
echo $data_dir
# if [ ! -d ${data_dir}/rgb ]; then
#     mkdir -p ${data_dir}/rgb;
# fi;

for f in ${data_dir}/yuv/*.yuv; do
last=${f##*_}
fps=${last%fps*}
name=${f##*/}
echo $last
echo $fps
echo $name
echo "${data_dir}/rgb/${name%.*}.mp4"
cp "${data_dir}/yuv/${name%.*}.yuv" /home/abhinau/Documents/temp/yuv_video.yuv
ffmpeg -hide_banner -loglevel panic -f rawvideo -vcodec rawvideo -s 1920x1088 -r $fps -pix_fmt yuv420p -i /home/abhinau/Documents/temp/yuv_video.yuv -c:v libx264 -qp 0 -y /home/abhinau/Documents/temp/mp4_video.mp4
cp /home/abhinau/Documents/temp/mp4_video.mp4 "${data_dir}/rgb/${name%.*}.mp4"
done
