import numpy as np
import pandas as pd

import time
import os

from scipy.io import savemat
import argparse

parser = argparse.ArgumentParser(description='Code to collect results from log files')
parser.add_argument('--data_path', help='Directory containing pristine videos', required=True)
parser.add_argument('--log_path', help='Directory containing SSIM logs from FFMPEG', required=True)
args = parser.parse_args()

videos_dir = args.data_path
logs_dir = args.log_path

# Scales at which compression will be done
scales = np.array([[256, 144], [426, 240], [640, 360], [720, 480], [960,540], [1280, 720]])
n_scales = scales.shape[0]

# QPs used while compressing at each scale
qps = np.arange(1,52,5)
n_qps = len(qps)

# Directory containing all videos
file_list = os.listdir(videos_dir)
file_list = [v for v in file_list if v[-3:] == 'mp4']
n_files = len(file_list)

print('Collecting scaling SSIM')
scale_data = np.empty((n_files,), dtype=object)
for f in range(n_files):
    temp_scale = []
    for s in range(n_scales):
        df = pd.read_csv(logs_dir + '/' + file_list[f][:-4] + '_' + str(scales[s,1]) + '_ref_ssim.log', sep=':|\ ', engine='python', header=None)
        temp_scale.append(df.values[:,3].astype('float32'))
    scale_data[f] = np.array(temp_scale).T # Order as n_frames, n_scales

print('Collecting compression SSIM')
comp_data = np.empty((n_files,), dtype=object)
for f in range(n_files):
    temp_scale = []
    for s in range(n_scales):
        temp_qp = []
        for q in range(n_qps):
            df = pd.read_csv(logs_dir + '/' + file_list[f][:-4] + '_' + str(scales[s,1]) + '_' + str(qps[q]) + '_comp_ssim.log', sep=':|\ ', engine='python', header=None)
            temp_qp.append(df.values[:,3].astype('float32'))
        temp_scale.append(temp_qp)
    comp_data[f] = np.array(temp_scale).transpose((2,0,1)) # Order as n_frames, n_scales, n_qps

print('Collecting true SSIM')
true_data = np.empty((n_files,), dtype=object)
for f in range(n_files):
    temp_scale = []
    for s in range(n_scales):
        temp_qp = []
        for q in range(n_qps):
            df = pd.read_csv(logs_dir + '/' + file_list[f][:-4] + '_' + str(scales[s,1]) + '_' + str(qps[q]) + '_true_ssim.log', sep=':|\ ', engine='python', header=None)
            temp_qp.append(df.values[:,3].astype('float32'))
        temp_scale.append(temp_qp)
    true_data[f] = np.array(temp_scale).transpose((2,0,1)) # Order as n_frames, n_scales, n_qps

savemat('data/ssim_data.mat', {'scale_data': scale_data, 'comp_data':comp_data, 'true_data':true_data})