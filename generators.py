from scipy.io import loadmat
import numpy as np
import torch

def DistMapGen(path):
    f = loadmat(path)
    data = f['data']
    n_train = data.shape[-2]
    while True:
        inds = np.random.permutation(n_train)
        for i in range(n_train):
            x = torch.from_numpy(data[:,:,inds[i],0]).unsqueeze(0).unsqueeze(0).float().cuda()
            y = torch.from_numpy(data[:,:,inds[i],1]).unsqueeze(0).unsqueeze(0).float().cuda()
            yield (x,y)

def SSIMDataGen(n_train, n_feats, batch_size, path, mode = 'train', shuffle = None):

    assert n_feats in [2, 4], 'Invalid number of features'

    if n_feats == 2:
        f = loadmat(path)
        ref_data = f['ref_data']
        comp_data = f['comp_data']
        
        n_frames = comp_data.shape[0]
        n_scales = comp_data.shape[1]
        n_qps = comp_data.shape[2]

        if mode == 'train':

            if shuffle is None:
                shuffle = True

            selected_ref_data = np.tile(np.expand_dims(ref_data[:n_train,:],-1),[1,1,n_qps]).flatten()
            selected_comp_data = np.reshape(comp_data[:n_train,:,:,:],[n_train*n_scales*n_qps,2])
            numel = n_train*n_scales*n_qps

            """ prod_preds = (comp_data[:,:,:,0]*np.tile(np.expand_dims(ref_data,-1),[1,1,11])).flatten() """

        elif mode == 'test':

            if shuffle is None:
                shuffle = False

            selected_ref_data = np.tile(np.expand_dims(ref_data[n_train:,:],-1),[1,1,n_qps]).flatten()
            selected_comp_data = np.reshape(comp_data[n_train:,:,:,:],[(n_frames - n_train)*n_scales*n_qps,2])
            numel = (n_frames - n_train)*n_scales*n_qps

        feats = np.vstack([selected_ref_data,selected_comp_data[:,0]]).T

        targets = selected_comp_data[:,1]

        while True:
            if shuffle:
                inds = np.random.permutation(numel)
            else:
                inds = np.arange(numel)

            for i in range(int(numel/batch_size)):
                x = torch.from_numpy(feats[inds[i*batch_size:(i+1)*batch_size],:]).float().cuda()
                y = torch.from_numpy(targets[inds[i*batch_size:(i+1)*batch_size]]).unsqueeze(1).float().cuda()
                yield (x,y)

    # Two SSIMs and s and q
    if n_feats == 4:
        f = loadmat(path)
        ref_data = f['ref_data']
        comp_data = f['comp_data']

        n_frames = comp_data.shape[0]
        n_scales = 6
        n_qps = 11

        scales = np.array([144, 240, 360, 480, 540, 720])
        qps = np.arange(1, 52, 5)

        if mode == 'train':

            if shuffle is None:
                shuffle = True

            selected_ref_data = np.tile(np.expand_dims(ref_data[:n_train,:],-1),[1,1,n_qps])
            selected_comp_data = comp_data[:n_train,:,:,:]
            numel = n_train*n_scales*n_qps
            n_frames_used = n_train

        elif mode == 'test':

            if shuffle is None:
                shuffle = False

            selected_ref_data = np.tile(np.expand_dims(ref_data[n_train:,:],-1),[1,1,n_qps])
            selected_comp_data = comp_data[n_train:,:,:,:]
            numel = (n_frames - n_train)*n_scales*n_qps
            n_frames_used = n_frames - n_train

        while True:

            if shuffle:
                inds = np.random.permutation(numel)
            else:
                inds = np.arange(numel)

            (f_inds, s_inds, q_inds) = np.unravel_index(inds,(n_frames_used,n_scales,n_qps))

            for i in range(int(numel/batch_size)):

                feats = np.vstack([selected_ref_data[f_inds[i*batch_size:(i+1)*batch_size], s_inds[i*batch_size:(i+1)*batch_size], q_inds[i*batch_size:(i+1)*batch_size]], \
                                    selected_comp_data[f_inds[i*batch_size:(i+1)*batch_size], s_inds[i*batch_size:(i+1)*batch_size], q_inds[i*batch_size:(i+1)*batch_size],0], \
                                    1080/(scales[s_inds[i*batch_size:(i+1)*batch_size]]), \
                                    51/(qps[q_inds[i*batch_size:(i+1)*batch_size]])]).T
                targets = selected_comp_data[f_inds[i*batch_size:(i+1)*batch_size], s_inds[i*batch_size:(i+1)*batch_size], q_inds[i*batch_size:(i+1)*batch_size],1]
                x = torch.from_numpy(feats).float().cuda()
                y = torch.from_numpy(targets).unsqueeze(1).float().cuda()
                yield (x,y)

    # AGGD fits to dist map MSCN coefficients
    elif n_feats == 8:
        f = loadmat(path)
        ref_data = f['ref_data']
        comp_data = f['comp_data']

        n_frames = comp_data.shape[0]
        n_scales = comp_data.shape[1]
        n_qps = comp_data.shape[2]

        if mode == 'train':
            selected_ref_data = np.reshape(np.tile(np.expand_dims(ref_data[:n_train,:],-2),[1,1,n_qps,1]),[n_train*n_scales*n_qps,4])
            selected_comp_data = np.reshape(comp_data[:n_train,:,:,:],[n_train*n_scales*n_qps,5])
            numel = n_train*n_scales*n_qps

        elif mode == 'test':
            selected_ref_data = np.reshape(np.tile(np.expand_dims(ref_data[n_train:,:],-2),[1,1,n_qps,1]),[(n_frames - n_train)*n_scales*n_qps,4])
            selected_comp_data = np.reshape(comp_data[n_train:,:,:,:],[(n_frames - n_train)*n_scales*n_qps,5])
            numel = (n_frames - n_train)*n_scales*n_qps

        feats = np.concatenate([selected_ref_data,selected_comp_data[:,:-1]],axis=-1)

        targets = selected_comp_data[:,-1]

        while True:
            inds = np.random.permutation(numel)
            for i in range(int(numel/batch_size)):
                x = torch.from_numpy(feats[inds[i*batch_size:(i+1)*batch_size],:]).float().cuda()
                y = torch.from_numpy(targets[inds[i*batch_size:(i+1)*batch_size]]).unsqueeze(1).float().cuda()
                yield (x,y) 

def LogSSIMDataGen(n_train, batch_size, mode = 'train'):
    f = loadmat('comp_data.mat')
    ref_data = f['ref_data']
    comp_data = f['comp_data']

    n_frames = comp_data.shape[0]
    n_scales = comp_data.shape[1]
    n_qps = comp_data.shape[2]

    if mode == 'train':
        selected_ref_data = np.tile(np.expand_dims(ref_data[:n_train,:],-1),[1,1,n_qps]).flatten()
        selected_comp_data = np.reshape(comp_data[:n_train,:,:,:],[n_train*n_scales*n_qps,2])
        numel = n_train*n_scales*n_qps

        """ prod_preds = (comp_data[:,:,:,0]*np.tile(np.expand_dims(ref_data,-1),[1,1,11])).flatten() """

    elif mode == 'test':
        selected_ref_data = np.tile(np.expand_dims(ref_data[n_train:,:],-1),[1,1,n_qps]).flatten()
        selected_comp_data = np.reshape(comp_data[n_train:,:,:,:],[(n_frames - n_train)*n_scales*n_qps,2])
        numel = (n_frames - n_train)*n_scales*n_qps

    feats = np.vstack([selected_ref_data,selected_comp_data[:,0]]).T

    targets = selected_comp_data[:,1]

    while True:
        inds = np.random.permutation(numel)
        for i in range(int(numel/batch_size)):
            x = torch.from_numpy(feats[inds[i*batch_size:(i+1)*batch_size],:]).float().cuda()
            y = torch.from_numpy(np.log(targets[inds[i*batch_size:(i+1)*batch_size]])).unsqueeze(1).float().cuda()
            yield (x,y)