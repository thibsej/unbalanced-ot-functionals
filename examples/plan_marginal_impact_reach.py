import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from common.utils import euclidean_cost
from common.sinkhorn import BatchVanillaSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy

path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)

def template_measure(nsample):
    x1 = np.linspace(0.0, 0.2, nsample)
    a1 = np.ones(nsample)
    a1[0], a1[-1] = 0.0, 0.0
    a1 = a1 / np.sum(a1)

    x2 = np.linspace(0.9, 1.0, nsample)
    a2 = 0.05 - np.abs(x2 - 0.95)
    a2[0], a2[-1] = 0.0, 0.0
    a2 = a2 / np.sum(a2)

    x = np.concatenate((x1, x2))
    a = np.concatenate((0.65 * a1, 0.35 * a2))
    a = a / np.sum(a)

    y1 = np.linspace(0.2, 0.4, nsample)
    b1 = np.linspace(0.0, 1.0, nsample)
    b1[0], b1[-1] = 0.0, 0.0
    b1 = b1 / np.sum(b1)

    y2 = np.linspace(0.5, 0.9, nsample)
    b2 = np.sqrt( np.abs(1 - ((y2 - 0.7) / 0.2)**2) )
    b2[0], b2[-1] = 0.0, 0.0
    b2 = b2 / np.sum(b2)

    y = np.concatenate((y1, y2))
    b = np.concatenate((0.45 * b1, 0.55 * b2))
    b = b / np.sum(b)

    return a, x, b, y

# Init of measures and solvers
a, x, b, y = template_measure(300)
A, X, B, Y = torch.from_numpy(a)[None, :], torch.from_numpy(x)[None, :, None], torch.from_numpy(b)[None, :], \
             torch.from_numpy(y)[None, :, None]
blur = 1e-3
reach = np.array([10**x for x in np.linspace(-2, np.log10(0.5), 4)])
cost = euclidean_cost(2)
solver = BatchVanillaSinkhorn(nits=10000, nits_grad=2, tol=1e-8, assume_convergence=True)
list_entropy = [KullbackLeibler(blur, reach[0]), TotalVariation(blur, reach[0])]

# Init of plot
blue = (.55,.55,.95)
red = (.95,.55,.55)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(48,12))

# Plotting transport marginals for each entropy
for i in range(len(list_entropy)):
    for j in range(len(reach)):
        entropy = list_entropy[i]
        entropy.reach = reach[j]
        f, g = solver.sinkhorn_asym(A, X, B, Y, cost, entropy)
        C = cost(X, Y)
        pi = ((f[:, :, None] + g[:, None, :] - C) / blur).exp() * A[:, :, None] * B[:, None, :]

        pi_1, pi_2 = pi.sum(dim=2), pi.sum(dim=1)
        pi_1, pi_2 = pi_1[0, :].data.numpy(), pi_2[0, :].data.numpy()

        ax[i, j].plot(x, a, color='b', linestyle='--')
        ax[i, j].plot(y, b, color='r', linestyle='--')
        ax[i, j].fill_between(x, 0, pi_1, color=red)
        ax[i, j].fill_between(y, 0, pi_2, color=blue)
        ax[i, j].set_yticklabels([])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_title(f'reach={reach[j]:0.3f}', fontsize=55)
        # ax[i, j].set_title(f'{entropy.__name__}, reach={reach[j]:0.3f}', fontsize=30)
plt.tight_layout()
plt.savefig(path + '/comparison_entropy_impact_reach.eps', format='eps')
plt.show()