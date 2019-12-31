import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
plt.ion()

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
models_dict = {'linear_svm_2': 'Linear SVM (2)',
               'gaussian_svm_2': 'Gaussian SVM (2)',
               'nn_2': 'Neural Net (2)',
               'linear_svm_4': 'Linear SVM (4)',
               'gaussian_svm_4': 'Gaussian SVM (4)',
               'nn_4': 'Neural Net (4)'}

n_models = len(models_dict.keys())

# Plot variation with training size
n_trains = np.arange(2500, 26000, 2500)
fig, axs = plt.subplots(1, 2)
for i, model in enumerate(models_dict.keys()):
    f = loadmat(os.path.join('results', 'phase2', model + '_train_size_analysis.mat'))

    axs[0].plot(n_trains, f[model + '_train_pcc'].squeeze(), linestyle='-', color=colors[i], label=models_dict[model] + ' - Train PCC')
    axs[0].plot(n_trains, f[model + '_test_pcc'].squeeze(), linestyle='--', color=colors[i], label=models_dict[model] + ' - Test PCC')

    axs[1].plot(n_trains, f[model + '_train_srocc'].squeeze(), linestyle='-', color=colors[i], label=models_dict[model] + ' - Train SROCC')
    axs[1].plot(n_trains, f[model + '_test_srocc'].squeeze(), linestyle='--', color=colors[i], label=models_dict[model] + ' - Test SROCC')

axs[0].set_title('PCC')
axs[0].legend()
axs[1].set_title('SROCC')
axs[1].legend()

# Plot variation with scale and QP
scales = np.array([144, 240, 360, 480, 540, 720])
n_scales = len(scales)
qps = np.arange(1, 52, 5)
n_qps = len(qps)

plt.figure()
for i, model in enumerate(models_dict.keys()):
    f = loadmat(os.path.join('results', 'phase2', model + '_scale_qp_analysis.mat'))

    plt.subplot(2, n_models, i + 1)
    plt.imshow(f[model + '_train_pcc'], vmin=0.9, vmax=1, aspect='equal')
    plt.title(models_dict[model] + 'Train PCC')
    plt.xticks(np.arange(n_qps), qps)
    plt.yticks(np.arange(n_scales), scales)

    plt.subplot(2, n_models, i + n_models + 1)
    plt.imshow(f[model + '_test_pcc'], vmin=0.9, vmax=1, aspect='equal')
    plt.title(models_dict[model] + 'Test PCC')
    plt.xticks(np.arange(n_qps), qps)
    plt.yticks(np.arange(n_scales), scales)

plt.figure()
for i, model in enumerate(models_dict.keys()):
    f = loadmat(os.path.join('results', 'phase2', model + '_scale_qp_analysis.mat'))

    plt.subplot(2, n_models, i + 1)
    plt.imshow(f[model + '_train_srocc'], vmin=0.9, vmax=1, aspect='equal')
    plt.title(models_dict[model] + 'Train SROCC')
    plt.xticks(np.arange(n_qps), qps)
    plt.yticks(np.arange(n_scales), scales)

    plt.subplot(2, n_models, i + n_models + 1)
    plt.imshow(f[model + '_test_srocc'], vmin=0.9, vmax=1, aspect='equal')
    plt.title(models_dict[model] + 'Test SROCC')

    plt.xticks(np.arange(n_qps), qps)
    plt.yticks(np.arange(n_scales), scales)
