import numpy as np
import torch
import matplotlib.pyplot as plt
from common.utils import generate_measure
from common.functional import regularized_ot
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy


nbatch, nsample, ndim, p = 1, 100, 2, 2
_, x = generate_measure(nbatch, nsample, ndim)
a = torch.ones((1,nsample)) / nsample

x = torch.Tensor([0.5, 0.25])[None, None, :] * x
y = x + torch.Tensor([0.5, 0.75])[None, None, :]
X, Y = x[0,:,:].numpy(), y[0,:,:].numpy()
x.requires_grad = True

blur, reach = 1e-2, 0.5
L_entropy = [Balanced(blur), KullbackLeibler(blur, reach), TotalVariation(blur, reach), Range(blur, 0.9, 1.1),
             PowerEntropy(blur, reach, 0), PowerEntropy(blur, reach, -1)]


for i in range(6):
    fig, ax = plt.subplots(figsize=(8, 8))
    entropy = L_entropy[i]
    cost = regularized_ot(a, x, a, y, p, entropy, nits=10000, tol=0, assume_convergence=True)
    ax.scatter(X[:, 0], X[:, 1], c='b')
    ax.scatter(Y[:, 0], Y[:, 1], c='r')
    grad_x = torch.autograd.grad(cost, [x], retain_graph=True)[0]
    field = -(grad_x / a[0, :, None]).data.numpy()
    ax.quiver(X[:, 0], X[:, 1], field[0, :, 0], field[0, :, 1], color=(.48, .85, .37), scale=2,
                    scale_units="xy", alpha=.3)
    ax.set_title(type(entropy).__name__)
    fig.legend()
    plt.show()
