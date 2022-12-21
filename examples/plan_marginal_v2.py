import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from unbalancedot.utils import euclidean_cost
from unbalancedot.sinkhorn import BatchVanillaSinkhorn
from unbalancedot.entropy import (
    KullbackLeibler,
    Balanced,
    TotalVariation,
    Range,
    PowerEntropy,
)

path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)


def template_measure(nsample):
    x1 = np.linspace(0.0, 0.2, nsample)
    a1 = np.ones(nsample)
    a1[0], a1[-1] = 0.0, 0.0
    a1 = a1 / np.sum(a1)

    x2 = np.linspace(0.8, 1.0, nsample)
    a2 = 0.1 - np.abs(x2 - 0.9)
    a2[0], a2[-1] = 0.0, 0.0
    a2 = a2 / np.sum(a2)

    x = np.concatenate((x1, x2))
    a = np.concatenate((0.65 * a1, 0.35 * a2))
    a = a / np.sum(a)

    y1 = np.linspace(0.2, 0.4, nsample)
    b1 = np.linspace(0.0, 1.0, nsample)
    b1[0], b1[-1] = 0.0, 0.0
    b1 = b1 / np.sum(b1)

    y2 = np.linspace(0.5, 0.8, nsample)
    b2 = np.sqrt(np.abs(1 - ((y2 - 0.65) / 0.15) ** 2))
    b2[0], b2[-1] = 0.0, 0.0
    b2 = b2 / np.sum(b2)

    y = np.concatenate((y1, y2))
    b = np.concatenate((0.45 * b1, 0.55 * b2))
    b = b / np.sum(b)

    return a, x, b, y


# Init of measures and solvers
a, x, b, y = template_measure(1000)
A, X, B, Y = (
    torch.from_numpy(a)[None, :],
    torch.from_numpy(x)[None, :, None],
    torch.from_numpy(b)[None, :],
    torch.from_numpy(y)[None, :, None],
)
p, blur, reach = 2, 1e-3, 0.1
cost = euclidean_cost(p)
solver = BatchVanillaSinkhorn(
    nits=10000, nits_grad=1, tol=1e-5, assume_convergence=True
)
list_entropy = [
    Balanced(blur),
    KullbackLeibler(blur, reach),
    TotalVariation(blur, reach),
    Range(blur, 0.7, 1.3),
    PowerEntropy(blur, reach, 0.0),
]

# Init of plot
blue = (0.55, 0.55, 0.95)
red = (0.95, 0.55, 0.55)
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
plt.fill_between(x, 0, a, color="orange")
plt.fill_between(y, 0, b, color="purple")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(path + "/comparison_entropy_reference.pdf", bbox_inches='tight', format="pdf")


# Plotting each entropy separately
for entropy in list_entropy:
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    f, g = solver.sinkhorn_asym(A, X, B, Y, cost, entropy)
    C = cost(X, Y)
    pi = (
        ((f[:, :, None] + g[:, None, :] - C) / blur).exp()
        * A[:, :, None]
        * B[:, None, :]
    )

    pi_1, pi_2 = pi.sum(dim=2), pi.sum(dim=1)
    pi_1, pi_2 = pi_1[0, :].data.numpy(), pi_2[0, :].data.numpy()

    plt.plot(x, a, color='orange', linestyle="--")
    plt.plot(y, b, color='purple', linestyle="--")
    plt.fill_between(x, 0, pi_1, color=(178/255, 102/255,255/255))
    plt.fill_between(y, 0, pi_2, color=(255/255, 178/255, 102/255))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(path + f"/comparison_entropy_{entropy.__name__}.pdf",
                bbox_inches='tight', format="pdf")
