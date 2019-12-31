import argparse

from sklearn import svm
from sklearn.preprocessing import StandardScaler

import pickle
from scipy.io import loadmat, savemat
from scipy.stats import spearmanr, pearsonr

import numpy as np

parser = argparse.ArgumentParser(description='Code to train SVM models')
parser.add_argument("--mode", help="Analysis mode - analyze performance by scale - qp or training set size", required=True)
parser.add_argument("--train_size", help="Number of frames to use for training", type=int, default=None)
parser.add_argument("--model", help="Model to use for training", default='linear')
parser.add_argument("--n_frames", help="Total number of frames in the dataset", type=int, required=True)
parser.add_argument("--n_feats", help="Number of features to supply to model", type=int, default=2)
parser.add_argument("--data_path", help="Path to dataset", required=True)
parser.add_argument("--max_iter", help="Maximum number of iterations to run SVM", type=int, default=2000)
args = parser.parse_args()

models = ['linear', 'gaussian']

assert args.n_feats in [2, 4], "Invalid number of features"

assert args.train_size is None or args.train_size < args.n_frames, "Size of training set must be smaller than the total number of frames in the dataset"
assert args.model in models, "Invalid choice of model"
assert args.mode in ['analyze_training_size', 'analyze_scale_qp'], "Invalid choice of analysis"
assert args.train_size is not None or args.mode == 'analyze_training_size', "Must set training size to analyze scale - QP"

n_feats = args.n_feats
n_frames = args.n_frames
n_scales = 6
n_qps = 11

n_trains = np.arange(2500, 26000, 2500)
scales = np.array([144, 240, 360, 480, 540, 720])
qps = np.arange(1, 52, 5)

if args.model == 'linear':
    clf = svm.SVR(kernel='linear', verbose=True, max_iter=args.max_iter)
else:
    clf = svm.SVR(kernel='rbf', verbose=True, max_iter=args.max_iter)

f = loadmat(args.data_path)
scale_data = np.concatenate(f['scale_data'].squeeze(), 0)
comp_data = np.concatenate(f['comp_data'].squeeze(), 0)
true_data = np.concatenate(f['true_data'].squeeze(), 0)

scale_data = np.tile(np.expand_dims(scale_data, -1), [1, 1, n_qps])

if args.mode == 'analyze_training_size':

    train_pcc = np.zeros((len(n_trains),))
    train_srocc = np.zeros((len(n_trains),))
    test_pcc = np.zeros((len(n_trains),))
    test_srocc = np.zeros((len(n_trains),))

    models = []

    for i in range(len(n_trains)):

        print("Training size: ", n_trains[i])

        if args.n_feats == 2:
            train_feats = np.vstack([scale_data[:n_trains[i], :, :].flatten(), comp_data[:n_trains[i], :, :].flatten()]).T
            test_feats = np.vstack([scale_data[n_trains[i]:, :, :].flatten(), comp_data[n_trains[i]:, :, :].flatten()]).T
        else:
            train_feats = np.vstack([scale_data[:n_trains[i], :, :].flatten(), comp_data[:n_trains[i], :, :].flatten(), np.tile(np.expand_dims(1080/scales, -1),
                                    [n_trains[i], 1, n_qps]).flatten(), np.tile(51/qps, [n_trains[i], n_scales, 1]).flatten()]).T
            test_feats = np.vstack([scale_data[n_trains[i]:, :, :].flatten(), comp_data[n_trains[i]:, :, :].flatten(), np.tile(np.expand_dims(1080/scales, -1),
                                    [n_frames - n_trains[i], 1, n_qps]).flatten(), np.tile(51/qps, [n_frames - n_trains[i], n_scales, 1]).flatten()]).T

        train_targets = true_data[:n_trains[i], :, :].flatten()
        test_targets = true_data[n_trains[i]:, :, :].flatten()

        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)

        clf.fit(train_feats, train_targets)
        train_preds = clf.predict(train_feats)
        test_preds = clf.predict(test_feats)

        temp = pearsonr(train_preds.squeeze(), train_targets.squeeze())
        train_pcc[i] = temp[0]
        temp = pearsonr(test_preds.squeeze(), test_targets.squeeze())
        test_pcc[i] = temp[0]
        temp = spearmanr(train_preds.squeeze(), train_targets.squeeze())
        train_srocc[i] = temp[0]
        temp = spearmanr(test_preds.squeeze(), test_targets.squeeze())
        test_srocc[i] = temp[0]

    savemat('results/' + args.model+'_svm_' + str(args.n_feats) + '_train_size_analysis.mat', {args.model+'_svm_' + str(args.n_feats) + '_train_pcc': train_pcc,
            args.model+'_svm_' + str(args.n_feats) + '_test_pcc': test_pcc, args.model+'_svm_' + str(args.n_feats) + '_train_srocc': train_srocc,
            args.model+'_svm_' + str(args.n_feats) + '_test_srocc': test_srocc})

elif args.mode == 'analyze_scale_qp':

    n_train = args.train_size

    if args.n_feats == 2:
        train_feats = np.vstack([scale_data[:n_train, :, :].flatten(), comp_data[:n_train, :, :].flatten()]).T
        test_feats = np.vstack([scale_data[n_train:, :, :].flatten(), comp_data[n_train:, :, :].flatten()]).T
    else:
        train_feats = np.vstack([scale_data[:n_train, :, :].flatten(), comp_data[:n_train, :, :].flatten(), np.tile(np.expand_dims(1080/scales, -1),
                                [n_train, 1, n_qps]).flatten(), np.tile(51/qps, [n_train, n_scales, 1]).flatten()]).T
        test_feats = np.vstack([scale_data[n_train:, :, :].flatten(), comp_data[n_train:, :, :].flatten(), np.tile(np.expand_dims(1080/scales, -1),
                                [n_frames - n_train, 1, n_qps]).flatten(), np.tile(51/qps, [n_frames - n_train, n_scales, 1]).flatten()]).T

    train_targets = true_data[:n_train, :, :].flatten()
    test_targets = true_data[n_train:, :, :].flatten()

    scaler = StandardScaler()
    scaler.fit(train_feats)
    train_feats = scaler.transform(train_feats)
    test_feats = scaler.transform(test_feats)

    clf.fit(train_feats, train_targets)

    train_preds = clf.predict(train_feats)
    test_preds = clf.predict(test_feats)

    temp = pearsonr(train_preds, train_targets)
    print("Train PCC: ", temp[0])

    temp = pearsonr(test_preds, test_targets)
    print("Test PCC: ", temp[0])

    temp = spearmanr(train_preds, train_targets)
    print("Train SROCC: ", temp[0])

    temp = spearmanr(test_preds, test_targets)
    print("Test SROCC: ", temp[0])

    all_train_pcc = np.zeros((n_scales, n_qps))
    all_test_pcc = np.zeros((n_scales, n_qps))
    all_train_srocc = np.zeros((n_scales, n_qps))
    all_test_srocc = np.zeros((n_scales, n_qps))

    for s in range(n_scales):
        for q in range(n_qps):

            print("Resolution:", scales[s], "QPS:", qps[q])

            train_scale_data = scale_data[:n_train, s, q]
            test_scale_data = scale_data[n_train:, s, q]

            train_comp_data = comp_data[:n_train, s, q]
            test_comp_data = comp_data[n_train:, s, q]

            train_true_data = true_data[:n_train, s, q]
            test_true_data = true_data[n_train:, s, q]

            if args.n_feats == 2:
                train_feats = np.vstack([train_scale_data, train_comp_data]).T
                test_feats = np.vstack([test_scale_data, test_comp_data]).T
            else:
                train_feats = np.vstack([train_scale_data, train_comp_data, 1080/(scales[s]*np.ones((n_train,))), 51/(qps[q]*np.ones((n_train,)))]).T
                test_feats = np.vstack([test_scale_data, test_comp_data, 1080/(scales[s]*np.ones((n_frames - n_train,))), 51/(qps[q]*np.ones((n_frames - n_train,)))]).T

            train_targets = train_true_data
            test_targets = test_true_data

            train_preds = clf.predict(train_feats)
            test_preds = clf.predict(test_feats)

            all_train_pcc[s, q] = pearsonr(train_preds, train_targets)[0]
            all_train_srocc[s, q] = spearmanr(train_preds, train_targets)[0]
            all_test_pcc[s, q] = pearsonr(test_preds, test_targets)[0]
            all_test_srocc[s, q] = spearmanr(train_preds, train_targets)[0]

    savemat('results/' + args.model+'_svm_' + str(args.n_feats) + '_scale_qp_analysis.mat', {args.model+'_svm_' + str(args.n_feats) + '_train_pcc': all_train_pcc,
            args.model+'_svm_' + str(args.n_feats) + '_test_pcc': all_test_pcc, args.model+'_svm_' + str(args.n_feats) + '_train_srocc': all_train_srocc,
            args.model+'_svm_' + str(args.n_feats) + '_test_srocc': all_test_srocc})
    pickle.dump(clf, 'results/' + args.model+'_svm_' + str(args.n_feats) + '_model.pkl')
