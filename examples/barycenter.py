import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

from common.sinkhorn import BatchVanillaSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.functional import sinkhorn_divergence

torch.set_default_dtype(torch.float32)

def gaussian_spot(grid, mean, var):
    dens = np.exp( - ((grid - mean) / (var))**2 )
    dens = dens / np.sum(dens)
    x = grid[dens > 1e-5]
    a = dens[dens > 1e-5]
    a[0], a[-1] = 0.0, 0.0
    a = a / np.sum(a)
    print(f"Size of tensor is {x.shape}")
    return a, x

nsample = 1000
grid = np.linspace(0.0, 1.0, nsample)
a1, x1 = gaussian_spot(grid, 0.15, 0.03)
a2, x2 = gaussian_spot(grid, 0.65, 0.03)
b1, y1 = gaussian_spot(grid, 0.35, 0.03)
b2, y2 = gaussian_spot(grid, 0.85, 0.03)
a, x = np.concatenate([0.7 * a1, 0.3 * a2]), np.concatenate([x1, x2])
b, y = np.concatenate([0.3 * b1, 0.7 * b2]), np.concatenate([y1, y2])
a, b = a / np.sum(a), b / np.sum(b)
A, X = torch.from_numpy(a).float()[None, :], torch.from_numpy(x).float()[None, :, None]
B, Y = torch.from_numpy(b).float()[None, :], torch.from_numpy(y).float()[None, :, None]

nbar = 1000
C, Z = torch.ones(nbar)[None, :] / nbar, torch.from_numpy(np.linspace(0.0, 1.0, nbar)).float()[None, :, None]
Z.requires_grad = True
C.requires_grad = True


p=2
solver = BatchVanillaSinkhorn(nits=1000, tol=1e-5, assume_convergence=True)
entropy = KullbackLeibler(1e-2, 0.6)
optimizer = torch.optim.Adam([Z, C], lr=0.1)
for it in range(3):
    print(f"Iteration {it}")
    optimizer.zero_grad()
    cost = 0.5 * sinkhorn_divergence(C, Z, A, X, p, entropy, solver) + \
           0.5 * sinkhorn_divergence(C, Z, B, Y, p, entropy, solver)
    cost.backward()
    optimizer.step()


kde = KernelDensity(kernel='gaussian', bandwidth=1/np.sqrt(nsample) ).fit(Z[0,:].data.numpy())
dens = kde.score_samples(grid[:, None])
dens = np.exp(dens) / np.sum(np.exp(dens))

plt.plot(x, a, color='r')
plt.plot(y, b, color='b')
plt.plot(grid, dens)
plt.show()
