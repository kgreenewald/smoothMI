import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self, n_0, n_i, n_K, K, sigma_z, nonlinearity):
        super(Net, self).__init__()

        self.nonlinearity = nonlinearity

        # hidden layers dimensions
        self.layer_dims = n_i[:]
        self.layer_dims.insert(0, n_0)
        self.layer_dims.insert(len(self.layer_dims), n_K)

        # layers connecting 0--1, 1--2, 2--3, ..., K-1--K
        self.layers = nn.ModuleList([nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]) for i in range(K)])

        self.beta = sigma_z ** 2

        self.ind_samples_start = 0


    def prepare_layer_data_saving(self, num_data_X, num_replicas):
        self.layer_values = []
        self.num_data_X = num_data_X

        for dim in self.layer_dims[1:]:  # ignore first layer
            self.layer_values.append(np.zeros((num_data_X, dim, num_replicas)))


    def forward(self, x, save_layer_vals=0, ind_replica=0):

        batch_size = x.size(0)

        if save_layer_vals and self.ind_samples_start < self.num_data_X:
            ind_start = self.ind_samples_start
            ind_end = ind_start + batch_size

        noisy_ind = np.random.randint(0, len(self.layers))

        for i in range(len(self.layers) - 1):

            if self.nonlinearity:
                x_noiseless = self.nonlinearity(self.layers[i](x))
            else:
                x_noiseless = self.layers[i](x)

            x = self.addNoise(x_noiseless)

            # --------  save layer values --
            if save_layer_vals and self.ind_samples_start < self.num_data_X:
                self.layer_values[i][ind_start:ind_end, :, ind_replica] = x_noiseless.data.cpu().numpy()
            # ----------------------------------------

        # in the last layer don't use non-linearity and don't add noise
        out = self.layers[-1](x)

        # advance counter for the next call
        if save_layer_vals and self.ind_samples_start < self.num_data_X:

            self.layer_values[-1][ind_start:ind_end, :, ind_replica] = F.softmax(out, dim=1).data.cpu().numpy()

            self.ind_samples_start = ind_end

        return out


    def addNoise(self, layer, apply=True):
        if apply and self.beta > 0.0 and self.training:
            layer = layer + torch.normal(means=torch.zeros_like(layer), std=np.sqrt(self.beta))
        return layer


    def getWeights(self):
        weights = []
        for i in range(len(self.layers)):
            d={}
            d['A'] = self.layers[i].weight.data.cpu().numpy()
            d['b'] = self.layers[i].bias.data.cpu().numpy()
            weights.append(d)
        return weights


    def orthWeightUpdate(self, alpha=1e-4):

        # update all layers except last one
        for i in range(len(self.layers)-1):
            A = self.layers[i].weight.data
            updateA = torch.mm(torch.mm(A, A.t()), A) - A
            self.layers[i].weight.data.sub_(alpha * updateA)

