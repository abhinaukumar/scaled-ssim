import numpy as np
import cv2
import os
from os import system
from skimage.metrics import structural_similarity as ssim_index
import time

import argparse

from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

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
# parser.add_argument('--interval', help='Interval at which to sample the reference histogram', type=int, default=5)
args = parser.parse_args()

# assert args.interval > 0, "Interval must be a positive integer"

# Possible scales at which compression will be don
scales = {288: [352, 288], 384: [512, 384], 480: [720, 480], 720: [1280, 720], 1080: [1920, 1080]}
n_scales = len(scales)

# Intervals at which to sample the true quality map
intervals = [2, 5, 7, 10, 12, 15]
n_intervals = len(intervals)
ssim_map_refs = [None]*n_intervals

# Directory containing all videos
videos_dir = args.data_path
ref_file_list = os.listdir(os.path.join(videos_dir, 'ref', 'rgb'))
ref_file_list = sorted([v for v in ref_file_list if v[-3:] == 'mp4'], key=lambda v: v.lower())
n_ref_files = len(ref_file_list)

dist_file_list = os.listdir(os.path.join(videos_dir, 'dis', 'rgb'))
dist_file_list = sorted([v for v in dist_file_list if v[-3:] == 'mp4'], key=lambda v: v.lower())
n_dist_files = len(dist_file_list)

# Video parameters
width = 1920
height = 1080

# Data storing SSIM at compression scale
comp_ssim_data = np.empty((n_dist_files,), dtype=object)
# Data storing SSIM at rendering scale
true_ssim_data = np.empty((n_dist_files,), dtype=object)
# Data storing predicted SSIM at rendering scale
pred_ssim_data = np.empty((n_dist_files,), dtype=object)

start = time.time()
i_dist = 0

for f1 in range(n_ref_files):

    ref_filename = ref_file_list[f1][:-4].split('_')[0]
    # for s in range(n_scales):
    while(i_dist < n_dist_files and ref_filename in dist_file_list[i_dist]):

        dis_filename = dist_file_list[i_dist][:-4]
        s = int(dis_filename.split('_')[2])

        # Downsample video to compression scale
        system("ffmpeg -hide_banner -loglevel panic -i " + os.path.join(videos_dir, 'ref', 'rgb', ref_file_list[f1]) +
               " -vf scale=" + str(scales[s][0]) + "x" + str(scales[s][1]) +
               " -sws_flags lanczos" +
               " -y temp/hist_" + str(s) + "_scaled_video.mp4")

        system("ffmpeg -hide_banner -loglevel panic -i " + os.path.join(videos_dir, 'dis', 'rgb', dist_file_list[i_dist]) +
               " -vf scale=" + str(scales[s][0]) + "x" + str(scales[s][1]) +
               " -sws_flags lanczos" +
               " -y temp/hist_" + str(s) + "_comp_video.mp4")

        v1 = cv2.VideoCapture(os.path.join(videos_dir, 'ref', 'rgb', ref_file_list[f1]))
        v2 = cv2.VideoCapture("temp/hist_" + str(s) + "_scaled_video.mp4")
        v3 = cv2.VideoCapture("temp/hist_" + str(s) + "_comp_video.mp4")
        v4 = cv2.VideoCapture(os.path.join(videos_dir, 'dis', 'rgb', dist_file_list[i_dist]))

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
                [true_ssim, ssim_map_true] = ssim_index(Y_upcomp, Y_original, gaussian_weights=False, full=True)

                # if k % args.interval == 0:
                #     [temp, ssim_map_true] = ssim_index(Y_upcomp, Y_original, gaussian_weights=False, full=True)
                #     ssim_map_ref = ssim_map_true
                #     true_ssims.append(np.mean(ssim_map_true))
                # else:
                #     true_ssims.append(ssim_index(Y_upcomp, Y_original, gaussian_weights=False, full=False))

                for i_interval, interval in enumerate(intervals):
                    if k % interval == 0:
                        ssim_map_refs[i_interval] = ssim_map_true.copy()

                # ssim_map_trans = match_histograms(ssim_map_comp, ssim_map_ref)
                ssim_map_trans_list = []
                for ssim_map_ref in ssim_map_refs:
                    ssim_map_trans_list.append(match_histograms(ssim_map_comp, ssim_map_ref))

                comp_ssims.append(np.mean(ssim_map_comp))
                true_ssims.append(np.mean(ssim_map_true))
                pred_ssims.append([np.mean(ssim_map_trans) for ssim_map_trans in ssim_map_trans_list])

                k += 1
            else:
                break

        comp_ssim_data[i_dist] = comp_ssims
        true_ssim_data[i_dist] = true_ssims
        pred_ssim_data[i_dist] = pred_ssims

        i_dist += 1
        print("Processed Video " + str(i_dist) +
              " at scale " + str(scales[s][0]) + "x" + str(scales[s][1]))
        print("Time elapsed: " + str(time.time() - start) + " s")

savemat('results/hist_multi_interval_nflx_results.mat', {'comp_ssim_data': comp_ssim_data, 'true_ssim_data': true_ssim_data, 'pred_ssim_data': pred_ssim_data})

pred_data = np.array([np.mean(v) for v in pred_ssim_data])
true_data = np.array([np.mean(v) for v in true_ssim_data])

f = loadmat('data/nflx_repo_scores.mat')
scores = f['scores'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - true_data, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - true_data) - b2))) + b3 * (1 - true_data) + b4)
true_pcc = pearsonr(scores_pred, scores)[0]
true_srocc = spearmanr(scores_pred, scores)[0]

[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - pred_data, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - pred_data) - b2))) + b3 * (1 - pred_data) + b4)
pred_pcc = pearsonr(scores_pred, scores)[0]
pred_srocc = spearmanr(scores_pred, scores)[0]

savemat('results/hist_multi_interval_nflx_results.mat', {'comp_ssim_data': comp_ssim_data, 'true_ssim_data': true_ssim_data, 'pred_ssim_data': pred_ssim_data,
                                          'true_pcc': true_pcc, 'true_srocc': true_srocc, 'pred_pcc': pred_pcc, 'pred_srocc': pred_srocc})
