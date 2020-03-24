#!/bin/bash

# ref_directory=$1
# dist_directory=$2

# declare -A fps_dict

# for file in ${ref_directory}/*.mp4
# do
#     echo $file
#     filename="${file##*/}"
#     filename="${filename%.*}"

#     OLDIFS=$IFS
#     IFS='_'
#     read -ra name_arr <<< "${filename}"
#     IFS=$OLDIFS
#     fps=${name_arr[1]}
#     fps=${fps%fps*}

#     fps_dict[${name_arr[0]}]=$fps
# done

# declare -A scales=( ["288"]="352:288" ["384"]="512:384" ["480"]="720:480" ["720"]="1280:720" ["1080"]="1920:1080")

# for ref_file in ${ref_directory}/*.mp4
# do

#     ref_filename="${ref_file##*/}"
#     ref_filename="${ref_filename%.*}"

#     OLDIFS=$IFS
#     IFS='_'
#     read -ra name_arr <<< "${ref_filename}"
#     IFS=$OLDIFS
#     ref_name=${name_arr[0]}

#     for dist_file in ${dist_directory}/${ref_name}*.mp4
#     do

#         dist_filename="${dist_file##*/}"
#         dist_filename="${dist_filename%.*}"

#         OLDIFS=$IFS
#         IFS='_'
#         read -ra name_arr <<< "${dist_filename}"
#         IFS=$OLDIFS
#         res=${name_arr[2]}
#         echo $res
#         resolution=${scales[$res]}

#         ffmpeg -i "${ref_directory}/${ref_filename}.mp4" -vf scale=$resolution -sws_flags lanczos -y "temp/${resolution##*:}_scaled_video.mp4"
#         ffmpeg -i "temp/${resolution##*:}_scaled_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upscaled_video.mp4"
#         ffmpeg -i "temp/${resolution##*:}_upscaled_video.mp4" -i "${ref_directory}/${ref_filename}.mp4" -lavfi ssim="ssims/${dist_filename}_ref_ssim.log" -f null -

#         echo $dist_filename

#         ffmpeg -i "${dist_directory}/${dist_filename}.mp4" -vf scale=$resolution -sws_flags lanczos -y "temp/${resolution##*:}_comp_video.mp4"

#         # ./analyze_video.sh $directory $filename $resolution

#         ffmpeg -i "temp/${resolution##*:}_comp_video.mp4" -i "temp/${resolution##*:}_scaled_video.mp4" -lavfi ssim="ssims/${dist_filename}_comp_ssim.log" -f null -
#         ffmpeg -i "${dist_directory}/${dist_filename}.mp4" -i "${ref_directory}/${ref_filename}.mp4" -lavfi ssim="ssims/${dist_filename}_true_ssim.log" -f null -


#     done;

# done;


directory=$1
dist_folder=$2

scale_names="360p 720p 1080p 2160p"
declare -A scales=( ["360p"]="480:360" ["720p"]="1280:720" ["1080p"]="1920:1080" ["2160p"]="3840:2160")

for ref_file in ${directory}/src_videos/*.mp4
do
    ref_filename="${ref_file##*/}"
    ref_filename="${ref_filename%.*}"
    echo $ref_filename
    pattern=${directory}/${dist_folder}/avpvs/${ref_filename}_[1-9]+kbps_*.mkv

    for scale_name in $scale_names
    do
        
        ffmpeg -i $ref_file -vf scale=${scales[${scale_name}]} -sws_flags lanczos -y temp/${dist_folder}_${scale_name}_scaled_video.mp4
        ffmpeg -i temp/${dist_folder}_${scale_name}_scaled_video.mp4 -vf scale=1920:1080 -sws_flags lanczos -y temp/${dist_folder}_${scale_name}_upscaled_video.mp4

        for dist_file in ${directory}/${dist_folder}/avpvs/${ref_filename}*${scale}*.mkv
        do

            dist_filename="${dist_file##*/}"
            dist_filename="${dist_filename%.*}"

            metadata="${dist_filename#*${ref_filename}}"
            OLDIFS=$IFS
            IFS='_'
            read -ra metadata_arr <<< "${metadata}"
            IFS=$OLDIFS

            test=${metadata_arr[1]}

            if [[ "$test" == *kbps ]]
            then
                echo $dist_filename
                echo $ref_filename
                res=${metadata_arr[1]}

                ffmpeg -i "${ref_directory}/${ref_filename}.mp4" -vf scale=$resolution -sws_flags lanczos -y "temp/${resolution##*:}_scaled_video.mp4"
                ffmpeg -i "temp/${resolution##*:}_scaled_video.mp4" -vf scale=1920:1080 -sws_flags lanczos -y "temp/${resolution##*:}_upscaled_video.mp4"   
                ffmpeg -i "temp/${resolution##*:}_upscaled_video.mp4" -i "${ref_directory}/${ref_filename}.mp4" -lavfi ssim="ssims/${dist_filename}_ref_ssim.log" -f null -

            fi
        done
    done
done