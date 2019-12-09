import argparse

import torch
from torch import nn

import models

from scipy.io import loadmat, savemat
from scipy.stats import spearmanr, pearsonr

import numpy as np 

parser = argparse.ArgumentParser(description='Code to train NN models')
parser.add_argument("--mode", help = "Analysis mode - analyze performance by scale - qp or training set size", required = True)
parser.add_argument("--train_size", help = "Number of frames to use for training", type = int, default = None)
parser.add_argument("--model", help = "Model to use for training", default = 'FCNet')
parser.add_argument("--epochs", help = "Number of epochs to train the model for", type = int, default = 50)
parser.add_argument("--n_frames", help = "Total number of frames in the dataset", type = int, required = True)
parser.add_argument("--n_feats", help = "Number of features to supply to model", type = int, default = 2)
parser.add_argument("--data_path", help = "Path to dataset", required = True)
args = parser.parse_args()

assert args.n_feats in [2, 4], "Invalid number of features"

assert args.train_size is None or args.train_size < args.n_frames, "Size of training set must be smaller than the total number of frames in the dataset"
assert args.model in models.model_class, "Invalid choice of model"
assert args.mode in ['analyze_training_size', 'analyze_scale_qp'], "Invalid choice of analysis"
assert args.train_size is not None or args.mode == 'analyze_training_size', "Must set training size to analyze scale - QP"

n_feats = args.n_feats
h_size = 2*n_feats
n_frames = args.n_frames
n_scales = 6
n_qps = 11
batch_size = 10*n_scales*n_qps
n_trains = np.arange(2500,26000,2500)

scales = np.array([144, 240, 360, 480, 540, 720])
qps = np.arange(1, 52, 5)

Net = models.model_class[args.model]
data_gen_function = models.data_generator[args.model]

if args.mode == 'analyze_training_size':

    train_pcc = np.zeros((len(n_trains),))
    train_srocc = np.zeros((len(n_trains),))
    test_pcc = np.zeros((len(n_trains),))
    test_srocc = np.zeros((len(n_trains),))
    
    models = []

    for i in range(len(n_trains)):

        print("Training size: ", n_trains[i])
        
        net = Net(n_feats,h_size).cuda()

        train_gen = data_gen_function(n_trains[i],n_feats,batch_size,args.data_path,'train')
        net.train(train_gen,int(n_trains[i]/10),int(args.epochs))
        test_gen = data_gen_function(n_trains[i],n_feats,batch_size,args.data_path,'test')

        train_preds = np.zeros((n_trains[i]*n_scales*n_qps,1))
        train_targets = np.zeros((n_trains[i]*n_scales*n_qps,1))
        test_preds = np.zeros(((n_frames-n_trains[i])*n_scales*n_qps,1))
        test_targets = np.zeros(((n_frames-n_trains[i])*n_scales*n_qps,1))

        for j in range(int(n_trains[i]/10)):
            (x,y) = next(train_gen)
            train_preds[j*batch_size:(j+1)*batch_size,:] = net.forward(x).cpu().detach().numpy()
            train_targets[j*batch_size:(j+1)*batch_size,:] = y.cpu().detach().numpy()

        for j in range(int((n_frames - n_trains[i])/10)):
            (x,y) = next(test_gen)
            test_preds[j*batch_size:(j+1)*batch_size,:] = net.forward(x).cpu().detach().numpy()
            test_targets[j*batch_size:(j+1)*batch_size,:] = y.cpu().detach().numpy()
            
        temp = pearsonr(train_preds.squeeze(),train_targets.squeeze())
        train_pcc[i] = temp[0]
        temp = pearsonr(test_preds.squeeze(),test_targets.squeeze())
        test_pcc[i] = temp[0]
        temp = spearmanr(train_preds.squeeze(),train_targets.squeeze())
        train_srocc[i] = temp[0]
        temp = spearmanr(test_preds.squeeze(),test_targets.squeeze())
        test_srocc[i] = temp[0]

        models.append(net)

    savemat('results/nn_' + str(args.n_feats) + '_training_size_analysis.mat',{'nn_' + str(args.n_feats) + '_train_pcc':train_pcc, 'nn_' + str(args.n_feats) + '_test_pcc': test_pcc, 'nn_' + str(args.n_feats) + '_train_srocc': train_srocc, 'nn_' + str(args.n_feats) + '_test_srocc': test_srocc})

elif args.mode == 'analyze_scale_qp':

    n_train = args.train_size
    
    Net = models.model_class[args.model]
    data_gen_function = models.data_generator[args.model]

    net = Net(n_feats, h_size).cuda()
    train_gen = data_gen_function(n_train,n_feats,batch_size,args.data_path,'train')
    net.train(train_gen,int(n_train/10),int(args.epochs))
    test_gen = data_gen_function(n_train,n_feats,batch_size,args.data_path,'test')

    train_preds = np.zeros((n_train*n_scales*n_qps,))
    train_targets = np.zeros((n_train*n_scales*n_qps,))
    test_preds = np.zeros(((n_frames-n_train)*n_scales*n_qps,))
    test_targets = np.zeros(((n_frames-n_train)*n_scales*n_qps,))

    for j in range(int(n_train/10)):
        (x,y) = next(train_gen)
        train_preds[j*batch_size:(j+1)*batch_size] = net.forward(x).cpu().detach().numpy().squeeze()
        train_targets[j*batch_size:(j+1)*batch_size] = y.cpu().detach().numpy().squeeze()

    for j in range(int((n_frames - n_train)/10)):
        (x,y) = next(test_gen)
        test_preds[j*batch_size:(j+1)*batch_size] = net.forward(x).cpu().detach().numpy().squeeze()
        test_targets[j*batch_size:(j+1)*batch_size] = y.cpu().detach().numpy().squeeze()
        
    temp = pearsonr(train_preds.squeeze(),train_targets.squeeze())
    print("Train PCC: ", temp[0])

    temp = pearsonr(test_preds.squeeze(),test_targets.squeeze())
    print("Test PCC: ", temp[0])

    temp = spearmanr(train_preds.squeeze(),train_targets.squeeze())
    print("Train SROCC: ", temp[0])

    temp = spearmanr(test_preds.squeeze(),test_targets.squeeze())
    print("Test SROCC: ", temp[0])

    f = loadmat(args.data_path)
    scale_data = np.concatenate(f['scale_data'].squeeze(),0)
    comp_data = np.concatenate(f['comp_data'].squeeze(),0)
    true_data = np.concatenate(f['true_data'].squeeze(),0)

    all_train_pcc = np.zeros((n_scales,n_qps))
    all_test_pcc = np.zeros((n_scales,n_qps))
    all_train_srocc = np.zeros((n_scales,n_qps))
    all_test_srocc = np.zeros((n_scales,n_qps))

    for s in range(n_scales):
        for q in range(n_qps):
            train_scale_data = scale_data[:n_train,s]
            test_scale_data = scale_data[n_train:,s]

            train_comp_data = comp_data[:n_train,s,q]
            test_comp_data = comp_data[n_train:,s,q]

            train_true_data = true_data[:n_train,s,q]
            test_true_data = true_data[n_train:,s,q]

            if args.n_feats == 2:
                train_feats = np.vstack([train_scale_data, train_comp_data]).T
                test_feats = np.vstack([test_scale_data, test_comp_data]).T
            else:
                train_feats = np.vstack([train_scale_data, train_comp_data,1080/(scales[s]*np.ones((n_train,))),51/(qps[q]*np.ones((n_train,)))]).T
                test_feats = np.vstack([test_scale_data, test_comp_data,1080/(scales[s]*np.ones((n_frames - n_train,))),51/(qps[q]*np.ones((n_frames - n_train,)))]).T               

            train_targets = train_true_data    
            test_targets = test_true_data

            train_preds = net.forward(torch.from_numpy(train_feats).float().cuda()).cpu().detach().numpy().squeeze()
            test_preds = net.forward(torch.from_numpy(test_feats).float().cuda()).cpu().detach().numpy().squeeze()

            all_train_pcc[s,q] = pearsonr(train_preds,train_targets)[0]
            all_train_srocc[s,q] = spearmanr(train_preds,train_targets)[0]
            all_test_pcc[s,q] = pearsonr(test_preds,test_targets)[0]
            all_test_srocc[s,q] = spearmanr(train_preds,train_targets)[0]
    savemat('results/nn_' + str(args.n_feats) + '_scale_qp_analysis.mat', {'nn_' + str(args.n_feats) + '_train_pcc': all_train_pcc, 'nn_' + str(args.n_feats) + '_test_pcc': all_test_pcc, 'nn_' + str(args.n_feats) + '_train_srocc': all_train_srocc, 'nn_' + str(args.n_feats) + '_test_srocc': all_test_srocc})
