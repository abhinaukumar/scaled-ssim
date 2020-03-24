import numpy as np
import pandas as pd

import os

import torch
import pickle
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
import argparse

parser = argparse.ArgumentParser(description='Code to collect results from log files')
parser.add_argument('--log_path', help='Directory containing SSIM logs from FFMPEG', required=True)

args = parser.parse_args()

logs_dir = args.log_path

dist_log_list = os.listdir(logs_dir)
scale_log_list = sorted([v for v in dist_log_list if ('ref' in v)], key=lambda v: v.lower())
comp_log_list = sorted([v for v in dist_log_list if ('comp' in v)], key=lambda v: v.lower())
true_log_list = sorted([v for v in dist_log_list if ('true' in v)], key=lambda v: v.lower())
n_dist_files = len(comp_log_list)

print('Collecting scaling SSIM')
scale_data = np.empty((n_dist_files,), dtype=object)
for f in range(n_dist_files):
    df = pd.read_csv(logs_dir + '/' + scale_log_list[f], sep=':| ', engine='python', header=None)
    scale_data[f] = df.values[:, 3].astype('float32')

print('Collecting compression SSIM')
comp_data = np.empty((n_dist_files,), dtype=object)
for f in range(n_dist_files):

    df = pd.read_csv(logs_dir + '/' + comp_log_list[f], sep=':| ', engine='python', header=None)
    comp_data[f] = df.values[:, 3].astype('float32')

print('Collecting true SSIM')
true_data = np.empty((n_dist_files,), dtype=object)
for f in range(n_dist_files):

    df = pd.read_csv(logs_dir + '/' + true_log_list[f], sep=':| ', engine='python', header=None)
    true_data[f] = df.values[:, 3].astype('float32')

savemat('data/ssim_test_data.mat', {'scale_data': scale_data, 'comp_data': comp_data, 'true_data': true_data})

# Evaluate all models
true_ssims = np.array([np.mean(v.squeeze()) for v in true_data])
f = loadmat('data/nflx_repo_scores.mat')
scores = f['scores'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - true_ssims, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - true_ssims) - b2))) + b3 * (1 - true_ssims) + b4)

true_pcc = pearsonr(scores_pred, scores)[0]
true_srocc = spearmanr(scores_pred, scores)[0]

prod_pred_ssims = np.array([np.mean(v1*v2) for (v1, v2) in zip(scale_data, comp_data)])
[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - prod_pred_ssims, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - prod_pred_ssims) - b2))) + b3 * (1 - prod_pred_ssims) + b4)
prod_pcc = pearsonr(scores_pred, scores)[0]
prod_srocc = spearmanr(scores_pred, scores)[0]

# NN
net = pickle.load(open('results/phase3/nn_2_model.pkl', 'rb'))

nn_pred_ssims = []
for i in range(n_dist_files):
    scale_ssims = scale_data[i].squeeze()
    comp_ssims = comp_data[i].squeeze()
    feats = torch.stack([torch.from_numpy(scale_ssims).float().cuda(), torch.from_numpy(comp_ssims).float().cuda()]).T
    nn_pred_ssims.append(np.mean(net.forward(feats).mean().detach().cpu().numpy()))

nn_pred_ssims = np.array(nn_pred_ssims)

[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - nn_pred_ssims, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - nn_pred_ssims) - b2))) + b3 * (1 - nn_pred_ssims) + b4)
nn_pcc = pearsonr(scores_pred, scores)[0]
nn_srocc = spearmanr(scores_pred, scores)[0]

scaler = pickle.load(open('results/phase3/svm_scaler.pkl', 'rb'))

# Linear SVM
clf = pickle.load(open('results/phase3/linear_svm_2_model.pkl', 'rb'))
lin_pred_ssims = []
for i in range(n_dist_files):
    scale_ssims = scale_data[i].squeeze()
    comp_ssims = comp_data[i].squeeze()
    feats = np.vstack([scale_ssims, comp_ssims]).T
    feats = scaler.transform(feats)
    lin_pred_ssims.append(np.mean(clf.predict(feats)))

lin_pred_ssims = np.array(lin_pred_ssims)

[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - lin_pred_ssims, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - lin_pred_ssims) - b2))) + b3 * (1 - lin_pred_ssims) + b4)
lin_pcc = pearsonr(scores_pred, scores)[0]
lin_srocc = spearmanr(scores_pred, scores)[0]

# Gaussian SVM
clf = pickle.load(open('results/phase3/gaussian_svm_2_model.pkl', 'rb'))
gauss_pred_ssims = []
for i in range(n_dist_files):
    scale_ssims = scale_data[i].squeeze()
    comp_ssims = (comp_data[i]).squeeze()
    feats = np.vstack([scale_ssims, comp_ssims]).T
    feats = scaler.transform(feats)
    gauss_pred_ssims.append(np.mean(clf.predict(feats)))

gauss_pred_ssims = np.array(gauss_pred_ssims)

[[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                      1 - gauss_pred_ssims, scores, p0=0.5*np.ones((5,)), maxfev=20000)

scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - gauss_pred_ssims) - b2))) + b3 * (1 - gauss_pred_ssims) + b4)
gauss_pcc = pearsonr(scores_pred, scores)[0]
gauss_srocc = spearmanr(scores_pred, scores)[0]
