import torch
from .utils import dist_matrix, scal
from .entropy import Entropy, TotalVariation, Range
from .sinkhorn import dist_matrix, BatchVanillaSinkhorn


def regularized_ot(a, x, b, y, p, entropy, solver=BatchVanillaSinkhorn(nits=100, nits_grad=5, tol=1e-3, assume_convergence=True)):
    f_x, g_y = solver.sinkhorn_asym(a, x, b, y, p=p, entropy=entropy)
    cost = entropy.output_regularized(a, x, b, y, p, f_x, g_y)
    return cost


def hausdorff_divergence(a, x, b, y, p, entropy, solver=BatchVanillaSinkhorn(nits=100, nits_grad=5, tol=1e-3,
                                                                             assume_convergence=True)):
    g_xy, f_x = solver.sinkhorn_sym(a, x, p=p, entropy=entropy, y_j=y)
    f_yx, g_y = solver.sinkhorn_sym(b, y, p=p, entropy=entropy, y_j=x)
    cost = entropy.output_hausdorff(a, x, b, y, p, f_yx, f_x, g_xy, g_y)
    return cost


def sinkhorn_divergence(a, x, b, y, p, entropy, solver=BatchVanillaSinkhorn(nits=100, nits_grad=5, tol=1e-3,
                                                                            assume_convergence=True)):
    if isinstance(entropy, TotalVariation) | isinstance(entropy, Range):
        f_xy, g_xy = solver.sinkhorn_asym(a, x, b, y, p=p, entropy=entropy)
        f_x1, f_x2 = solver.sinkhorn_asym(a, x, a, x, p=p, entropy=entropy)
        g_y1, g_y2 = solver.sinkhorn_asym(b, y, b, y, p=p, entropy=entropy)
        return entropy.output_sinkhorn(a, x, b, y, p, f_xy, f_x1, f_x2, g_xy, g_y1, g_y2)
    else:
        f_xy, g_xy = solver.sinkhorn_asym(a, x, b, y, p=p, entropy=entropy)
        _, f_x = solver.sinkhorn_sym(a, x, p=p, entropy=entropy)
        _, g_y = solver.sinkhorn_sym(b, y, p=p, entropy=entropy)
        return entropy.output_sinkhorn(a, x, b, y, p, f_xy, f_x, g_xy, g_y)



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