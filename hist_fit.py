import numpy as np
import cv2
import os
from os import system
from skimage.metrics import structural_similarity as ssim_index
import time

import argparse

from scipy.io import savemat
# from scipy.stats import spearmanr, pearsonr


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    inds = (idx > 0)

    idx[inds] = idx[inds] - (idx[inds] == len(array))
    idx[inds] = idx[inds] - (np.abs(value[inds] - array[idx[inds]-1]) < np.abs(value[inds] - array[idx[inds]]))

    return idx


def match_histograms(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get histogram and bin ids for every value in source
    s_counts, s_bins = np.histogram(source, bins=1000, density=True)
    t_counts, t_bins = np.histogram(template, bins=1000, density=True)

    s_values = (s_bins[:-1] + s_bins[1:])/2
    t_values = (t_bins[:-1] + t_bins[1:])/2

    bin_idx = find_nearest(s_values, source)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


parser = argparse.ArgumentParser(description='Code to test histogram matching')
parser.add_argument('--data_path', help='Directory containing pristine videos', required=True)
parser.add_argument('--interval', help='Interval at which to sample the reference histogram', type=int, default=5)
parser.add_argument('--scale_index', help='Index of compression scale (0-5)', type=int, required=True)
args = parser.parse_args()

assert args.interval > 0, "Interval must be a positive integer"
assert args.scale_index >= 0 and args.scale_index < 6, "Scale index must be in the range 0-5"

s = args.scale_index
# Scales at which compression will be done
scales = np.array([[1280, 720], [426, 240], [640, 360], [720, 480], [960, 540], [1280, 720]])
n_scales = scales.shape[0]

# QPs used while compressing at each scale
qps = np.arange(1, 52, 5)
n_qps = len(qps)

# Directory containing all videos
videos_dir = args.data_path
file_list = os.listdir(videos_dir)
file_list = [v for v in file_list if v[-3:] == 'mp4']
n_files = len(file_list)

# Video parameters
width = 1920
height = 1080

# Data storing SSIM at compression scale
comp_ssim_data = np.empty((n_files, n_qps), dtype=object)
# Data storing SSIM at rendering scale
true_ssim_data = np.empty((n_files, n_qps), dtype=object)
# Data storing predicted SSIM at rendering scale
pred_ssim_data = np.empty((n_files, n_qps), dtype=object)

start = time.time()
for f in range(n_files):

    # for s in range(n_scales):

    # Downsample video to compression scale
    system("ffmpeg -hide_banner -loglevel panic -i " + videos_dir + file_list[f] +
           " -filter:v scale=" + str(scales[s, 0]) + "x" + str(scales[s, 1]) +
           " -sws_flags lanczos" +
           " -y temp/hist_" + str(s) + "_scaled_video.mp4")

    print("Processed Reference Video " + str(f) +
          " at scale " + str(scales[s, 0]) + "x" + str(scales[s, 1]))
    print("Time elapsed: " + str(time.time() - start) + " s")

    for q in range(n_qps):

        # Compress scaled video
        system("ffmpeg -hide_banner -loglevel panic -i temp/hist_" + str(s) + "_scaled_video.mp4" +
               " -vcodec libx264 -crf " + str(qps[q]) +
               " -y temp/hist_" + str(s) + "_comp_video.mp4")

        # Resize compressed video back to original scale
        system("ffmpeg -hide_banner -loglevel panic -i temp/hist_" + str(s) + "_comp_video.mp4" +
               " -filter:v scale=" + str(width) + "x" + str(height) +
               " -sws_flags lanczos" +
               " -y temp/hist_" + str(s) + "_upscaled_comp_video.mp4")

        v1 = cv2.VideoCapture(videos_dir + file_list[f])
        v2 = cv2.VideoCapture("temp/hist_" + str(s) + "_scaled_video.mp4")
        v3 = cv2.VideoCapture("temp/hist_" + str(s) + "_comp_video.mp4")
        v4 = cv2.VideoCapture("temp/hist_" + str(s) + "_upscaled_comp_video.mp4")

        k = 0

        true_ssims = []
        comp_ssims = []
        pred_ssims = []

        # Calculate and predict SSIM before and after compression at both scales
        while(v1.isOpened() and v2.isOpened() and v3.isOpened() and v4.isOpened()):

            ret1, RGB_original = v1.read()
            ret2, RGB_scaled = v2.read()
            ret3, RGB_comp = v3.read()
            ret4, RGB_upcomp = v4.read()

            if ret1 and ret2 and ret3 and ret4:

                Y_original = cv2.cvtColor(RGB_original, cv2.COLOR_BGR2GRAY)
                Y_scaled = cv2.cvtColor(RGB_scaled, cv2.COLOR_BGR2GRAY)
                Y_comp = cv2.cvtColor(RGB_comp, cv2.COLOR_BGR2GRAY)
                Y_upcomp = cv2.cvtColor(RGB_upcomp, cv2.COLOR_BGR2GRAY)

                [temp, ssim_map_comp] = ssim_index(Y_comp, Y_scaled, gaussian_weights=False, full=True)

                if k % args.interval == 0:
                    [temp, ssim_map_true] = ssim_index(Y_upcomp, Y_original, gaussian_weights=False, full=True)
                    ssim_map_ref = ssim_map_true
                    true_ssims.append(np.mean(ssim_map_true))
                else:
                    true_ssims.append(ssim_index(Y_upcomp, Y_original, gaussian_weights=False, full=False))

                ssim_map_trans = match_histograms(ssim_map_comp, ssim_map_ref)

                comp_ssims.append(np.mean(ssim_map_comp))
                pred_ssims.append(np.mean(ssim_map_trans))

                k += 1
            else:
                break

        comp_ssim_data[f, q] = comp_ssims
        true_ssim_data[f, q] = true_ssims
        pred_ssim_data[f, q] = pred_ssims

        print("Processed Video " + str(f) +
              " at scale " + str(scales[s, 0]) + "x" + str(scales[s, 1]) +
              " and QP " + str(qps[q]))
        print("Time elapsed: " + str(time.time() - start) + " s")

savemat('results/hist_' + str(s) + '_ssim_data.mat', {'comp_ssim_data': comp_ssim_data, 'true_ssim_data': true_ssim_data, 'pred_ssim_data': pred_ssim_data})

# pcc = np.zeros((n_scales, n_qps))
# srocc = np.zeros((n_scales, n_qps))

# for s in range(n_scales):
#     for q in range(n_qps):
#         true_data = []
#         pred_data = []

#         for f in range(n_files):
#             true_data.extend(true_ssim_data[f, s, q])
#             pred_data.extend(pred_ssim_data[f, s, q])

#         pcc[s, q] = pearsonr(true_data, pred_data)[0]
#         srocc[s, q] = spearmanr(true_data, pred_data)[0]

# true_data = []
# pred_data = []

# for s in range(n_scales):
#     for q in range(n_qps):
#         for f in range(n_files):
#             true_data.append(true_ssim_data[f, s, q])
#             pred_data.append(pred_ssim_data[f, s, q])

# true_data = np.concatenate(true_data, axis=0)
# pred_data = np.concatenate(pred_data, axis=0)

# all_pcc = pearsonr(true_data, pred_data)[0]
# all_srocc = spearmanr(true_data, pred_data)[0]

# savemat('results/hist_scale_qp_analysis.mat', {'hist_pcc': pcc, 'hist_srocc': srocc, 'hist_all_pcc': all_pcc, 'hist_all_srocc': all_srocc})
