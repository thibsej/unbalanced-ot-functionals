import torch
from .utils import dist_matrix, scal
from .entropy import Entropy
from .sinkhorn import dist_matrix, sinkhorn_asym, sinkhorn_sym


def regularized_ot(a, x, b, y, p, entropy, nits=100, tol=1e-3, assume_convergence=False):  # OT_eps
    f_x, g_y = sinkhorn_asym(a, x, b, y, p=p, entropy=entropy, nits=nits, tol=tol,
                             assume_convergence=assume_convergence)
    cost = entropy.output_regularized()(a, x, b, y, p, f_x, g_y)
    return cost


def hausdorff_divergence(a, x, b, y, p, entropy, nits=100, tol=1e-3, assume_convergence=False):  # H_eps
    g_xy, f_x = sinkhorn_sym(a, x, p=p, entropy=entropy, y_j=y, nits=nits, tol=tol,
                             assume_convergence=assume_convergence)
    f_yx, g_y = sinkhorn_sym(b, y, p=p, entropy=entropy, y_j=x, nits=nits, tol=tol,
                             assume_convergence=assume_convergence)
    cost = entropy.output_hausdorff()(a, x, b, y, p, f_yx, f_x, g_xy, g_y)
    return cost


def sinkhorn_divergence(a, x, b, y, p, entropy, nits=100, tol=1e-3, assume_convergence=False):  # S_eps
    f_xy, g_xy = sinkhorn_asym(a, x, b, y, p=p, entropy=entropy, nits=nits, tol=tol,
                               assume_convergence=assume_convergence)
    _, f_x = sinkhorn_sym(a, x, p=p, entropy=entropy, nits=nits, tol=tol, assume_convergence=assume_convergence)
    _, g_y = sinkhorn_sym(b, y, p=p, entropy=entropy, nits=nits, tol=tol, assume_convergence=assume_convergence)
    cost = entropy.output_sinkhorn()(a, x, b, y, p, f_xy, f_x, g_xy, g_y)
    return cost


def energyDistance(a, x, b, y, p=1):
    if (torch.equal(x, y)):
        Cxy = dist_matrix(x, y, p)
        return -.5 * torch.bmm((a - b)[:, None, :], torch.bmm(Cxy, (a - b)[:, :, None]))
    Cxy = dist_matrix(x, y, p)
    Cxx = dist_matrix(x, x, p)
    Cyy = dist_matrix(y, y, p)
    xy = torch.bmm(a[:, None, :], torch.bmm(Cxy, b[:, :, None]))
    xx = torch.bmm(a[:, None, :], torch.bmm(Cxx, a[:, :, None]))
    yy = torch.bmm(b[:, None, :], torch.bmm(Cyy, b[:, :, None]))
    return xy - .5 * xx - .5 * yy