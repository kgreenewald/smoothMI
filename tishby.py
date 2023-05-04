import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from tishby_net import Net
import argparse
import numpy as np
import pickle
from sklearn.utils import shuffle


def parse_arguments():
    # arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_0', type=int, help='input dimension (X = T_0)', default=12)
    parser.add_argument('--n_K', type=int, help='output dimension (T_K)', default=2)
    parser.add_argument('--n_i', type=int, nargs='+', help='dimension of hidden layers', default=[10, 7, 5, 4, 3])
    parser.add_argument('--sigma_z', type=float, help='sigma_z = sqrt(beta)', default=0.1)
    parser.add_argument('--nonlinearity', type=str, help='network nonlinearity', default="tanh")
    parser.add_argument('--leaky_slope', type=float, help='slope for Leaky_ReLU', default=0.01)
    parser.add_argument('--max_epochs', type=int, help='maximum number of epochs', default=10000)
    parser.add_argument('--ID', type=int, help='ID', default=0)
    parser.add_argument('--orth_udpate_alpha', type=float, help='alpha for orth update on weights', default=-1.0)
    parser.add_argument('--num_data_X', type=int, help='# of X samples to use in saturation estim', default=1024)
    parser.add_argument('--num_replicas', type=int, help='number of generated samples for each x in X', default=100)
    parser.add_argument('--num_subsampled_epochs', type=int, help='number of subsampled epochs from max_epochs', default=100)
    parser.add_argument('--save_layer_data', type=int, help='save layer data', default=1)
    parser.add_argument('--save_trained', type=int, help='save trained model?', default=1)
    parser.add_argument('--shuffle_labels', type=int, help='shuffle labels?', default=0)
    parser.add_argument('--data_location', type=str, help='location of data', default='./datasets/IB_data.npz')

    args_tmp = parser.parse_known_args()
    if args_tmp[0].save_trained:
        pickle.dump(vars(args_tmp[0]), open('./saved/tishby_args_' + str(args_tmp[0].ID) + '.pkl', 'wb'))

    parser.add_argument('--lr', type=float, help='learning rate', default=0.0004)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--summary', type=str, help='summary path', default='./summary/')
    parser.add_argument('--saved_model_path', type=str, help='location of saved models', default='./saved/')
    parser.add_argument('--save_evaluation', type=int, help='save train/test evaluations?', default=1)

    args = parser.parse_args()

    # number of layers K (total = K+1)
    args.K = len(args.n_i) + 1

    return args


# Tishby Dataset
class TishbyDataset(Dataset):

    def __init__(self, Dtype, args, uniform_sample=False):

        data = np.load(args.data_location)

        if Dtype == 'train':
            self.X, self.y = data['X_train'], data['y_train']
            if args.shuffle_labels:
                self.y = shuffle(self.y, random_state=0)
        elif Dtype == 'test':
            self.X, self.y = data['X_test'], data['y_test']
        # ALL: train + test
        else:
            self.X, self.y = np.concatenate([data['X_train'], data['X_test']], axis=0), np.concatenate([data['y_train'], data['y_test']])
            if args.shuffle_labels:
                self.y = shuffle(self.y, random_state=0)

        if uniform_sample:
            r_ind = np.random.randint(0, len(self.y), len(self.y))
            self.X = self.X[r_ind, :]
            self.y = self.y[r_ind]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, ind):
        return (self.X[ind, :], self.y[ind])



def run_test(model, criterion, loaders):

    model.eval()

    # run a test loop
    loss = 0
    correct = 0
    for data, target in loaders['test_loader']:
        data, target = Variable(data.type(torch.FloatTensor), volatile=True).cuda(), Variable(target.type(torch.LongTensor)).cuda()

        net_out = model(data)

        # sum up batch loss
        loss += criterion(net_out, target).data[0]

        pred = net_out.data.max(1)[1]  # get the index of the max log-probability

        correct += pred.eq(target.data).sum()

    test_loss = loss / len(loaders['test_loader'].dataset)

    model.train()

    return test_loss, correct


def evaluate(model, criterion, loaders, use_eval=True):

    if use_eval:
        model.eval()

    # evaluate on test set
    loss = 0
    correct_test = 0
    for data, target in loaders['test_loader']:
        data, target = Variable(data.type(torch.FloatTensor), volatile=True).cuda(), Variable(target.type(torch.LongTensor)).cuda()

        net_out = model(data)

        # sum up batch loss
        loss += criterion(net_out, target).data[0]

        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct_test += pred.eq(target.data).sum()

    test_loss = loss / len(loaders['test_loader'].dataset)
    correct_test /= len(loaders['test_loader'].dataset)

    # evaluate on train set
    loss = 0
    correct_train = 0
    for data, target in loaders['train_loader']:
        data, target = Variable(data.type(torch.FloatTensor), volatile=True).cuda(), Variable(target.type(torch.LongTensor)).cuda()

        net_out = model(data)

        # sum up batch loss
        loss += criterion(net_out, target).data[0]

        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct_train += pred.eq(target.data).sum()

    train_loss = loss / len(loaders['train_loader'].dataset)
    correct_train /= len(loaders['train_loader'].dataset)

    if use_eval:
        model.train()

    return train_loss, test_loss, correct_train, correct_test


def run_train(model, optimizer, criterion, args, loaders, epoch_subsample, all_dataset):

    model.train()

    Train = []; Test = []; Ctrain = []; Ctest = []

    # ------------------- Prepare to save data ----------------------------
    if args.save_layer_data:
        layer_data = [np.array([])] * args.K
        for ind in range(args.K - 1):
            layer_data[ind] = np.zeros((len(epoch_subsample), args.num_data_X, args.n_i[ind], args.num_replicas))

        layer_data[-1] = np.zeros((len(epoch_subsample), args.num_data_X, args.n_K, args.num_replicas))

        model.prepare_layer_data_saving(args.num_data_X, args.num_replicas)
    # ---------------------------------------------------------------------


    for epoch in range(args.max_epochs):

        # --------------------- Train -------------------------------------
        for _, (data, target) in enumerate(loaders['train_loader']):

            data, target = Variable(data.type(torch.FloatTensor)).cuda(), Variable(target.type(torch.LongTensor)).cuda()

            optimizer.zero_grad()

            net_out = model(data)

            loss = criterion(net_out, target)

            # normalize loss because "criterion" has size_average=False
            loss = loss / target.size()[0]

            loss.backward()

            optimizer.step()

            if args.orth_udpate_alpha > 0:
                model.orthWeightUpdate(args.orth_udpate_alpha)

        print('Train Epoch: {} \t Last batch loss: {:.6f}'.format(epoch, loss.data[0]))

        if epoch in epoch_subsample:
        # ------------------ evaluation (deterministic, noiseless) --------
            if args.save_evaluation:
                tr, te, ctr, cte = evaluate(model, criterion, loaders, use_eval=True)
                Train.append(tr); Test.append(te); Ctrain.append(ctr); Ctest.append(cte)

        # ------------------- save layer data -----------------------------
            if args.save_layer_data:

                for replica in range(args.num_replicas):

                    # reset counter
                    model.ind_samples_start = 0

                    for _, (data, _) in enumerate(loaders['all_loader']):

                        data = Variable(data.type(torch.FloatTensor)).cuda()

                        model(data, save_layer_vals=1, ind_replica=replica)

                        if model.ind_samples_start == args.num_data_X:
                            break

                ep_ind = np.nonzero(epoch_subsample == epoch)[0][0]

                for ind in range(args.K):
                    layer_data[ind][ep_ind, :, :, :] = model.layer_values[ind]

        # -------------------- save model ---------------------------------
        if args.save_trained:
            torch.save(model.state_dict(), args.saved_model_path + 'modelTishby_' + str(args.ID) + '_ep_' + str(epoch) + '.pt')


    if args.save_evaluation:
        pickle.dump({'Train': Train, 'Test': Test, 'Ctrain': Ctrain, 'Ctest': Ctest}, open(args.saved_model_path + 'modelTishby_' + str(args.ID) + '_eval.p', 'wb'), protocol = 4)

    if args.save_layer_data:
        pickle.dump({'data': layer_data, 'y': all_dataset.y}, open(args.saved_model_path + 'modelTishby_' + str(args.ID) + '_layer_data.p', 'wb'), protocol = 4)


def main():

    args = parse_arguments()

    # ------------------- Data Loaders -----------------
    train_dataset = TishbyDataset('train', args)
    test_dataset = TishbyDataset('test', args)
    all_dataset = TishbyDataset('all', args)

    loaders = {}

    loaders['train_loader'] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                          batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=5,
                                                          pin_memory=True)

    loaders['test_loader'] = torch.utils.data.DataLoader(dataset=test_dataset,
                                                         batch_size=args.batch_size,
                                                         shuffle=False,
                                                         num_workers=5,
                                                         pin_memory=True)

    loaders['all_loader'] = torch.utils.data.DataLoader(dataset=all_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=5,
                                                        pin_memory=True)
    # -------------------------------------------------

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


    epoch_subsample = np.unique(np.round(np.logspace(np.log10(1), np.log10(args.max_epochs - 1), num=args.num_subsampled_epochs, endpoint=True)).astype(int))

    model = Net(args.n_0, args.n_i, args.n_K, args.K, args.sigma_z, nonlinearity)
    model.cuda()

    criterion = nn.CrossEntropyLoss(size_average=False).cuda()

    # create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_train(model, optimizer, criterion, args, loaders, epoch_subsample, all_dataset)

    test_loss, correct = run_test(model, criterion, loaders)

    Ntest = len(loaders['test_loader'].dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, Ntest, 100.*correct/Ntest))

if __name__ == "__main__":
    main()
