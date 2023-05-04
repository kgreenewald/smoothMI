import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker

import numpy as np
import pickle
import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--modelID', type=int, help='Model ID', default=1)
    parser.add_argument('--estID', type=int, help='Estimation ID', default=1)

    args = parser.parse_args()

    D = pickle.load(open('./saved/tishby_args_' + str(args.modelID) + '.pkl', 'rb'))
    args.max_epochs = D['max_epochs']
    args.num_subsampled_epochs = D['num_subsampled_epochs']

    return args


def main():

    args = parse_arguments()

    fig = plt.figure(figsize=(10, 5))
    grid = plt.GridSpec(4, 5, top=0.8, hspace=0.1, wspace=0.0, left=0.07, right=0.9, bottom=0.1)

    ep_sub = np.unique(np.round(np.logspace(np.log10(1), np.log10(args.max_epochs-1), num=args.num_subsampled_epochs, endpoint=True)).astype(int))

    fileName = './saved/modelTishby_' + str(args.modelID) + '_layer_data.p'
    f = open(fileName, 'rb')
    M = pickle.load(f)

    #histograms
    colors = plt.cm.Greens_r(np.linspace(0, 0.5, len(ep_sub)))
    xmin = -1.0
    xmax = 1.0
    zmax = 0.5

    bins = np.linspace(xmin, xmax, 100)

    # to make tick labels to display power of 10 in histograms
    def log_tick_formatter(val, pos=None):
        val = np.int(np.log10(np.exp(val)))
        return '$10^{' + str(val) + '}$'

    # scale histogram axis, so y-axis is elongated
    scale = np.diag([1, 2, 1, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    # ----------------- histograms ---------------------
    ax = []
    ax.append(plt.subplot(grid[0:2, 0], projection='3d'))
    ax.append(plt.subplot(grid[0:2, 1], projection='3d'))
    ax.append(plt.subplot(grid[0:2, 2], projection='3d'))
    ax.append(plt.subplot(grid[0:2, 3], projection='3d'))
    ax.append(plt.subplot(grid[0:2, 4], projection='3d'))

    ax[0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0]), scale)
    ax[1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1]), scale)
    ax[2].get_proj = lambda: np.dot(Axes3D.get_proj(ax[2]), scale)
    ax[3].get_proj = lambda: np.dot(Axes3D.get_proj(ax[3]), scale)
    ax[4].get_proj = lambda: np.dot(Axes3D.get_proj(ax[4]), scale)

    for L in range(5):
        ax[L].set_title('Layer ' + str(L + 1), fontsize=9, pad=-50, loc='left')
        ax[L].set_ylabel('Epoch', fontsize=8, labelpad=0)
        ax[L].set_xlim([xmin, xmax])
        ax[L].set_zlim([0.0, zmax])
        ax[L].set_xticks([xmin, (xmin + xmax) / 2, xmax])
        ax[L].yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax[L].yaxis.set_major_locator(mticker.FixedLocator(np.log([10, 100, 1000, 10000])))
        ax[L].tick_params(axis='both', which='major', labelsize=5, pad=-2)
        ax[L].tick_params(axis='z', which='major', labelsize=5, pad=-1)
        ax[L].zaxis.set_major_locator(mticker.LinearLocator(3))
        ax[L].zaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "%.2f" % round(x, 2)))
        ax[L].view_init(azim=-45)
        for i in range(84):
            dat = M['data'][L][i, :, :, 0].reshape(-1)
            vals, _ = np.histogram(dat, bins=bins, density=False)
            vals = vals / len(dat)
            ax[L].bar(bins[:-1], vals, zs=np.log(ep_sub[i]), zdir='y', alpha=0.8, color=colors[i], width=0.1, edgecolor='black', linewidth=0.05)

    # ----------------- MI plot -----------------------

    MIres = pickle.load(open('./saved/MIresultsTishby_' + str(args.modelID) + '_' + str(args.estID) + '.p', 'rb'))

    colors = plt.cm.tab10(np.linspace(0.3, 1, 5))

    axMI = plt.subplot(grid[2, 0:3])
    for L in range(5):
        axMI.plot(ep_sub, MIres['mi'][:len(ep_sub), L, 0], color=colors[L], label='Layer ' + str(L + 1))
    axMI.legend(loc=6, prop={'size': 7})
    axMI.set_ylim([0.0, 8.9])
    axMI.set_xscale('log')
    axMI.set_ylabel('MI(nats)')
    axMI.xaxis.set_major_formatter(mticker.NullFormatter())
    axMI.yaxis.set_major_locator(mticker.MultipleLocator(4))
    axMI.grid(linestyle=':', alpha=0.7)
    for side in ['top', 'bottom', 'right', 'left']:
        axMI.spines[side].set_alpha(0.2)

    # ----------------- Train/Test loss -----------------------

    Evals = pickle.load(open('./saved/modelTishby_' + str(args.modelID) + '_eval.p', 'rb'))

    axEv = plt.subplot(grid[3, 0:3])
    axEv.plot(ep_sub, Evals['Train'][:len(ep_sub)], color='blue', label='Train')
    axEv.plot(ep_sub, Evals['Test'][:len(ep_sub)], color='red', label='Test')
    axEv.legend(loc=6, prop={'size': 8})
    axEv.set_ylabel('Loss')
    axEv.set_xlabel('Epoch', labelpad=-2)
    axEv.set_xscale('log')
    axEv.grid(linestyle=':', alpha=0.7)
    for side in ['top', 'bottom', 'right', 'left']:
        axEv.spines[side].set_alpha(0.2)

    # ----------------- Scatter Plots -------------------------

    axSc = []
    axSc.append(plt.subplot(grid[2, 3], projection='3d'))
    axSc.append(plt.subplot(grid[2, 4], projection='3d'))
    axSc.append(plt.subplot(grid[3, 3], projection='3d'))
    axSc.append(plt.subplot(grid[3, 4], projection='3d'))

    eps = [11, 35, 356, 7389]

    vmin = xmin
    vmax = xmax

    for ind, ep in enumerate(eps):
        e = min(ep_sub, key=lambda x: abs(x - ep))
        i = np.nonzero(ep_sub == e)[0][0]

        X = M['data'][4][i, :1000, 0, 0]
        Y = M['data'][4][i, :1000, 1, 0]
        Z = M['data'][4][i, :1000, 2, 0]

        if 'y' in M:
            X0 = X[M['y'][:1000]==0]
            X1 = X[M['y'][:1000]==1]
            Y0 = Y[M['y'][:1000]==0]
            Y1 = Y[M['y'][:1000]==1]
            Z0 = Z[M['y'][:1000]==0]
            Z1 = Z[M['y'][:1000]==1]
            axSc[ind].scatter(X0, Y0, Z0, alpha=0.03, marker='.', facecolors='dodgerblue', edgecolor='dodgerblue', s=50)
            axSc[ind].scatter(X1, Y1, Z1, alpha=0.03, marker='.', facecolors='orangered', edgecolor='orangered', s=50)
        else:
            axSc[ind].scatter(X, Y, Z, alpha=0.03, marker='.', facecolors='dodgerblue', edgecolor='dodgerblue', s=50)

        axSc[ind].set_title('Ep ' + str(ep), fontsize=8, loc='right', pad=-3)

        axSc[ind].set_zlim3d(vmin, vmax)
        axSc[ind].set_ylim3d(vmin, vmax)
        axSc[ind].set_xlim3d(vmin, vmax)

        axSc[ind].xaxis.set_major_locator(mticker.LinearLocator(3))
        axSc[ind].yaxis.set_major_locator(mticker.LinearLocator(3))
        axSc[ind].zaxis.set_major_locator(mticker.LinearLocator(3))

        axSc[ind].zaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "%.1f" % x))
        axSc[ind].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "%.1f" % x))
        axSc[ind].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "%.1f" % x))

        axSc[ind].tick_params(axis='both', which='major', labelsize=5, pad=-2)
        axSc[ind].tick_params(axis='z', which='major', labelsize=5, pad=-5)

    fig.savefig('fig_modelID_' + str(args.modelID) + '_' + str(args.estID) + '.png', bbox_inches='tight', dpi=500)


if __name__ == "__main__":
    main()
