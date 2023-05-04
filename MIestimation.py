import numpy as np
import argparse
import pickle
from tishby_net import Net
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import time
from sklearn.neighbors import KernelDensity
import torch.distributed as dist


def parse_arguments():
    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_replicas', type=int, help='number of generated samples for each x in X', default=100)
    parser.add_argument('--num_data_X', type=int, help='batch size', default=1024)
    parser.add_argument('--load_pre_dump', type=int, help='load pre-dump layer values(1) or compute them through propagation(0)', default=1)

    parser.add_argument('--saved_path', type=str, help='location of saved models', default='./saved/')
    parser.add_argument('--modelID', type=int, help='model ID', default=1)
    parser.add_argument('--estID', type=int, help='estimation ID', default=11)

    parser.add_argument('--epoch_subsampling', type=int, help='exponential (0) or linear (1) subsampling', default=0)
    parser.add_argument('--custom_max_epoch', type=int, help='customized max number of epochs', default=0)
    parser.add_argument('--num_epoch_splits', type=int, help='number of splits in subsampled epochs', default=1)
    parser.add_argument('--ind_epoch_split', type=int, help='index of epochs chunk to work on ', default=0)
    parser.add_argument('--epoch_i', type=int, help='index of epoch to work on in the selected chunk', default=-1)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--bin_size', type=float, help='bin size for the binning method', default=0.007)
    parser.add_argument('--num_MC_samples', type=int, help='number of MC samples', default=1000)
    parser.add_argument('--layers_to_compute', type=int, nargs='+', help='which layers to use for computation', default=[-1])
    parser.add_argument('--shared_file', type=str, help='file to share among parallel jobs', default='dist_shared_file')
    parser.add_argument('--n_parallel', type=int, help='number of parallel jobs to run MC method', default=1)
    parser.add_argument('--rank', type=int, help='id of this parallel job', default=0)


    args = parser.parse_args()

    D = pickle.load(open(args.saved_path + 'tishby_args_' + str(args.modelID) + '.pkl', 'rb'))

    args.n_0 = D['n_0']
    args.n_K = D['n_K']
    args.n_i = D['n_i']
    args.K = len(args.n_i) + 1  # number of layers K (total = K+1)

    args.sigma_z = D['sigma_z']

    print("args.sigma_z = ", args.sigma_z)

    args.max_epochs = D['max_epochs']
    args.num_subsampled_epochs = D['num_subsampled_epochs']
    args.nonlinearity = D['nonlinearity']
    args.leaky_slope = D['leaky_slope']

    return args


# Tishby Dataset
class TishbyDataset(Dataset):

    def __init__(self, Dtype, uniform_sample=False):

        data = np.load('./datasets/IB_data.npz')

        if Dtype == 'train':
            self.X, self.y = data['X_train'], data['y_train']
        elif Dtype == 'test':
            self.X, self.y = data['X_test'], data['y_test']
        # ALL: train + test
        else:
            self.X, self.y = np.concatenate([data['X_train'], data['X_test']], axis=0), np.concatenate([data['y_train'], data['y_test']])

        if uniform_sample:
            r_ind = np.random.randint(0, len(self.y), len(self.y))
            self.X = self.X[r_ind, :]
            self.y = self.y[r_ind]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, ind):
        return (self.X[ind, :], self.y[ind])



def entropyMC(layer_vals, args, doParallel):  # monte carlo integration

    N = np.shape(layer_vals)[0]
    d = np.shape(layer_vals)[1]

    # Get density of Gaussian mixture (equivalent to beta-wide KDE)
    kde = KernelDensity(kernel='gaussian', bandwidth=args.sigma_z, rtol=1e-3).fit(layer_vals)

    # Generate n_int points from the Gaussian mixture
    inds = np.repeat(range(N), args.num_MC_samples)

    # Monte Carlo entropy estimate via averaging log density [since entropy(p(x)) = - E[log(p(x))|x ~ p(x)]]

    if doParallel:
        inds_splits = np.array_split(inds, args.n_parallel)

        idx = inds_splits[args.rank]
        X = layer_vals[idx, :] + args.sigma_z * np.random.randn(len(idx), d)
        result = - kde.score(X) / (N * args.num_MC_samples)

        result = torch.DoubleTensor([result])
        dist.reduce(result, 0)
        result = result.numpy()[0]
    else:
        X = layer_vals[inds, :] + args.sigma_z * np.random.randn(N * args.num_MC_samples, d)
        result = - kde.score(X) / (N * args.num_MC_samples)

    return result


def MI_estimator_MC(data, layer_ind, args):

    data = torch.from_numpy(data)
    #dist.broadcast(data, 0)
    data = data.numpy()

    # data: num_samples X layer_dim X num_replicas
    d = np.shape(data)[1]
    N = np.shape(data)[0]

    # layer_vals = data[:, :, 0]
    layer_vals = np.moveaxis(data, 1, 2).reshape((-1, d))

    s = time.time()
    H_T_i = entropyMC(layer_vals, args, False)
    e = time.time()
    print('h(T) time = ', e-s)

    HX = []

    s = time.time()

    if layer_ind > 0:

        inds = range(N)
        inds_splits = np.array_split(inds, args.n_parallel)
        idx = inds_splits[args.rank]

        for data_ind in idx:
            layer_vals = data[data_ind, :, :].transpose()
            val = entropyMC(layer_vals, args, False)
            HX.append(val/len(inds)) # add elements already divided by N

        H_T_i_given_X = np.sum(HX)

        H_T_i_given_X = torch.DoubleTensor([H_T_i_given_X])
        #dist.reduce(H_T_i_given_X, 0)
        H_T_i_given_X = H_T_i_given_X.numpy()[0]

    else:
        H_T_i_given_X = d / 2.0 * np.log(2.0 * np.pi * np.exp(1) * args.sigma_z ** 2)

    e = time.time()
    print('h(T|X) time = ', e - s)

    mi = H_T_i - H_T_i_given_X
    mi = mi if mi > 0 else 0

    return mi, H_T_i, H_T_i_given_X


def main():

    args = parse_arguments()

    if not args.load_pre_dump:
    # ------------------------------- Data Loaders ---------------------------------------
        train_dataset = TishbyDataset('all', uniform_sample=True)

        loaders = {}
        loaders['all'] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=1,
                                                     pin_memory=True)
    # ------------------------------------------------------------------------------------


    # determine over how many epochs to run
    # exponential splits
    if args.epoch_subsampling == 0:
        epoch_subsample = np.unique(np.round(np.logspace(np.log10(1), np.log10(args.max_epochs - 1), num=args.num_subsampled_epochs, endpoint=True)).astype(int))
    # linear splits
    else:
        epoch_subsample = np.unique(np.round(np.linspace(1, args.max_epochs - 1, num=args.num_subsampled_epochs, endpoint=True)).astype(int))


    epoch_splits = np.array_split(epoch_subsample, args.num_epoch_splits)

    split = epoch_splits[args.ind_epoch_split]

    if args.epoch_i > len(split)-1:
        return

    # extract only single epoch (otherwise, go through all epochs in a split)
    if args.epoch_i >= 0:
        split = split[args.epoch_i:args.epoch_i+1]

    print('split = ', split)

    if args.load_pre_dump:
        fileName = './saved/modelTishby_' + str(args.modelID) + '_layer_data.p'
        f = open(fileName, 'rb')
        preDump = pickle.load(f)
        f.close()
    else:
        if args.nonlinearity == "tanh":
            nonlinearity = F.tanh
        elif args.nonlinearity == "relu":
            nonlinearity = F.relu
        elif args.nonlinearity == "lrelu":
            nonlinearity = nn.LeakyReLU(args.leaky_slope)
        elif args.nonlinearity == "lin":
            nonlinearity = None
        else:
            assert (False)

        model = Net(args.n_0, args.n_i, args.n_K, args.K, args.sigma_z, nonlinearity)
        model.train()
        model.prepare_layer_data_saving(args.num_data_X, args.num_replicas)


    if args.layers_to_compute[0] >= 0:
        layers_to_compute = args.layers_to_compute
    else:
        layers_to_compute = range(args.K - 1)

    #dist.init_process_group(backend='tcp',
    #                        world_size=args.n_parallel,
    #                        rank=args.rank,
    #                        init_method='file://saved/' + args.shared_file + '_' + str(args.modelID) + '_' + str(args.estID))

    for epoch in split:

        print('epoch = ', epoch)

        start_time = time.time()

        # ------------------------ forward pass to compute samples ------------------------------
        if not args.load_pre_dump and args.rank == 0:

            # propagate through network to get the layer values
            model.load_state_dict(torch.load(args.saved_path + 'modelTishby_' + str(args.modelID) + '_ep_' + str(epoch) + '.pt'))
            if torch.cuda.is_available():
                model.cuda()

            for replica in range(args.num_replicas):

                # reset counter
                model.ind_samples_start = 0

                for it_num, (data, _) in enumerate(loaders['all']):

                    data = Variable(data.type(torch.FloatTensor))
                    if torch.cuda.is_available():
                        data = data.cuda()

                    model(data, save_layer_vals=1, ind_replica=replica)

                    if model.ind_samples_start == args.num_data_X:
                        break


            print('Done with forward pass')
        # ------------------------ end of forward pass to compute samples ---------------------------


        # ------------------------- MI estimation ---------------------------------------------------
        mi_estimates = np.zeros((len(layers_to_compute), 3))

        for idx, layer_ind in enumerate(layers_to_compute):

            if args.load_pre_dump:
                ep_ind = np.nonzero(epoch_subsample == epoch)[0][0]
                data_noiseless = preDump['data'][layer_ind][ep_ind, :args.num_data_X, :, :args.num_replicas]
            else:
                data_noiseless = model.layer_values[layer_ind]

            i_mc, H_T_i, H_T_given_X = MI_estimator_MC(data_noiseless, layer_ind, args)

            mi_estimates[idx, 0] = i_mc
            mi_estimates[idx, 1] = H_T_i
            mi_estimates[idx, 2] = H_T_given_X

            print('i_mc = ', i_mc, 'H_T_i = ', H_T_i, ', H_T_given_X = ', H_T_given_X)
            print('Done with layer = ', layer_ind)
        # ------------------------- end MI estimation -----------------------------------------------


        # ---------------------------- save results -------------------------------------------------
        if args.rank == 0:
            fileName = './saved/MIresultsTishby_' + str(args.modelID) + '_' + str(args.estID) + '.p'

            if not os.path.exists(fileName):

                tmp_MI = np.zeros((len(epoch_subsample), len(layers_to_compute), 3))

                f = open(fileName, 'wb')
                pickle.dump({'mi': tmp_MI}, f)
                os.fsync(f)
                time.sleep(1)
                f.close()

            f = open(fileName, 'rb')
            M = pickle.load(f)
            f.close()

            ep_ind = np.nonzero(epoch_subsample == epoch)[0][0]

            M['mi'][ep_ind, :, :] = mi_estimates

            f = open(fileName, 'wb')
            pickle.dump(M, f)

            os.fsync(f)
            time.sleep(1)
            f.close()
        # ---------------------------- end save results ---------------------------------------------

        print('Elapsed time = ', time.time() - start_time)


if __name__ == "__main__":
    main()
