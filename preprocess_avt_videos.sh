#!/bin/bash

directory=$1
dist_folder=$2
avt_directory=$3

test_directories="test_1 test_2 test_3 test_4"

for ref_file in ${directory}/src_videos/*.mp4
do
    ref_filename="${ref_file##*/}"
    ref_filename="${ref_filename%.*}"

        for dist_file in ${directory}/${dist_folder}/segments/${ref_filename}*.mp4
        do
            dist_filename="${dist_file##*/}"
            dist_filename="${dist_filename%.*}"

            echo $ref_filename $dist_filename
            ${avt_directory}/gen_avpvs.py --avpvs_folder ${directory}/${dist_folder}/avpvs $dist_file $ref_file
        done
done
