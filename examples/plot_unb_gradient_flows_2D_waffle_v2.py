import os
import time

import numpy as np
from random import choices
import imageio
from matplotlib import pyplot as plt
from functools import partial
import torch

from unbalancedot.functional import regularized_ot, hausdorff_divergence, \
    sinkhorn_divergence
from unbalancedot.sinkhorn import BatchVanillaSinkhorn
from unbalancedot.entropy import KullbackLeibler, Balanced, TotalVariation, \
    Range
from unbalancedot.utils import euclidean_cost

# Build path to save plots
path = os.getcwd() + "/output"
if not os.path.isdir(path):
    os.mkdir(path)
path = path + "/unbalanced_gradient_flow_waffle"
if not os.path.isdir(path):
    os.mkdir(path)

# Check GPU availability for computational speedup
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.set_default_tensor_type(dtype)


###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~
def load_image(fname):
    img = imageio.imread(fname, as_gray=True)  # Grayscale
    img = (img[::-1, :]) / 255.
    return 1 - img


def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]),
                         np.linspace(0, 1, A.shape[1]))

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)


def display_samples(ax, x, a, color):
    x_ = x[0, :, :].detach().cpu().numpy()
    a_ = a[0, :].detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], s=50 * 100 * a_,
               c=color, edgecolors='none')


N, M = (100, 100) if not use_cuda else (2000, 2000)

X_i = draw_samples("data/density_a.png", N, dtype)[None, :, :]
Y_j = draw_samples("data/density_b.png", M, dtype)[None, :, :]
A_i = torch.ones(N)[None, :] / N
B_j = torch.ones(M)[None, :] / M


def gradient_flow(func, entropy, solver, cost, p, lr_x=.05, lr_a=.005,
                  Nsteps=100):
    """
    Flows along the gradient of the cost function, using a simple Euler scheme.
    """

    # Parameters for the gradient descent
    display_its = [int(t * Nsteps) for t in [0, .02, .03, 0.07, 0.017, 1.]]
    loss = partial(func, cost=cost, entropy=entropy, solver=solver)
    fname = path + f"/unbalanced_flow_{func.__name__}_p{p}_" \
                   f"{entropy.__dict__}_lrx{lr_x}_lra{lr_a}_steps{Nsteps}.png"

    # Use colors to identify the particles
    colors = (10 * X_i[0, :, 0]).cos() * (10 * X_i[0, :, 1]).cos()
    colors = colors.detach().cpu().numpy()

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()
    a_i, b_j = A_i.clone(), B_j.clone()

    # We're going to perform gradient descent on Loss(al, be)
    # wrt. the positions x_i of the diracs masses that make up al:
    x_i.requires_grad = True
    a_i.requires_grad = True

    t_0 = time.time()
    plt.figure(figsize=(12, 8))
    k = 1
    for i in range(Nsteps + 1):  # Euler scheme ===============
        # Compute cost and gradient
        div = loss(a_i, x_i, b_j, y_j, cost=cost, entropy=entropy)
        [g, m] = torch.autograd.grad(div, [x_i, a_i])

        if i in display_its:  # display
            ax = plt.subplot(2, 3, k)
            k = k + 1
            plt.set_cmap("hsv")

            # shameless hack to prevent a slight change of axis...
            plt.scatter([10], [10])

            display_samples(ax, y_j, b_j, [(178/255, 102/255, 255/255)])
            display_samples(ax, x_i, a_i, colors)

            ax.set_title("t = {:1.2f}".format(i / Nsteps), fontsize=25)

            plt.axis([0, 1, 0, 1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()

        # in-place modification of the tensor's values
        x_i.data -= lr_x * g
        if isinstance(entropy, Balanced):
            a_i.data *= (- (2 * lr_a * m)).exp()
            a_i.data /= a_i.data.sum(1)
        else:
            a_i.data *= (- (2 * lr_a * m)).exp()
        print(f"At step {i} the total mass is {a_i.sum().item()}")
    plt.title(f"t = {i / (Nsteps - 1):1.2f}, elapsed time: "
              f"{(time.time() - t_0) / Nsteps} s/it",
              fontsize=30)
    plt.savefig(fname)
    plt.show()


if __name__ == '__main__':
    setting = 0
    p, cost = 2, euclidean_cost(2)
    solver = BatchVanillaSinkhorn(nits=5000, nits_grad=15, tol=1e-8,
                                  assume_convergence=True)

    if setting == 0:  # Compare KL for various mass steps
        gradient_flow(sinkhorn_divergence, entropy=KullbackLeibler(1e-3, 0.3),
                      solver=solver, cost=cost, p=p,
                      lr_x=60., lr_a=0.5, Nsteps=300)
        gradient_flow(regularized_ot, entropy=KullbackLeibler(1e-3, 0.3),
                      solver=solver, cost=cost, p=p,
                      lr_x=60., lr_a=0.5, Nsteps=300)
        gradient_flow(sinkhorn_divergence, entropy=KullbackLeibler(1e-1, 0.3),
                      solver=solver, cost=cost, p=p,
                      lr_x=60., lr_a=0.5, Nsteps=300)
