import torch
from .entropy import TotalVariation, Range
from .sinkhorn import BatchVanillaSinkhorn
from .utils import dist_matrix


def regularized_ot(
    a,
    x,
    b,
    y,
    cost,
    entropy,
    solver=BatchVanillaSinkhorn(
        nits=100, nits_grad=5, tol=1e-3, assume_convergence=True
    ),
):
    f_x, g_y = solver.sinkhorn_asym(a, x, b, y, cost=cost, entropy=entropy)
    return entropy.output_regularized(a, x, b, y, cost, f_x, g_y)


def hausdorff_divergence(
    a,
    x,
    b,
    y,
    cost,
    entropy,
    solver=BatchVanillaSinkhorn(
        nits=100, nits_grad=5, tol=1e-3, assume_convergence=True
    ),
):
    g_xy, f_x = solver.sinkhorn_sym(a, x, cost=cost, entropy=entropy, y_j=y)
    f_yx, g_y = solver.sinkhorn_sym(b, y, cost=cost, entropy=entropy, y_j=x)
    return entropy.output_hausdorff(a, x, b, y, cost, f_yx, f_x, g_xy, g_y)


def sinkhorn_divergence(
    a,
    x,
    b,
    y,
    cost,
    entropy,
    solver=BatchVanillaSinkhorn(
        nits=100, nits_grad=5, tol=1e-3, assume_convergence=True
    ),
):
    if isinstance(entropy, TotalVariation) | isinstance(entropy, Range):
        f_xy, g_xy = solver.sinkhorn_asym(
            a, x, b, y, cost=cost, entropy=entropy
        )
        f_x1, f_x2 = solver.sinkhorn_asym(
            a, x, a, x, cost=cost, entropy=entropy
        )
        g_y1, g_y2 = solver.sinkhorn_asym(
            b, y, b, y, cost=cost, entropy=entropy
        )
        return entropy.output_sinkhorn(
            a, x, b, y, cost, f_xy, f_x1, f_x2, g_xy, g_y1, g_y2
        )
    else:
        f_xy, g_xy = solver.sinkhorn_asym(
            a, x, b, y, cost=cost, entropy=entropy
        )
        _, f_x = solver.sinkhorn_sym(a, x, cost=cost, entropy=entropy)
        _, g_y = solver.sinkhorn_sym(b, y, cost=cost, entropy=entropy)
        return entropy.output_sinkhorn(a, x, b, y, cost, f_xy, f_x, g_xy, g_y)


def energyDistance(a, x, b, y, p=1):
    if torch.equal(x, y):
        Cxy = dist_matrix(x, y, p)
        return -0.5 * torch.bmm(
            (a - b)[:, None, :], torch.bmm(Cxy, (a - b)[:, :, None])
        )
    Cxy = dist_matrix(x, y, p)
    Cxx = dist_matrix(x, x, p)
    Cyy = dist_matrix(y, y, p)
    xy = torch.bmm(a[:, None, :], torch.bmm(Cxy, b[:, :, None]))
    xx = torch.bmm(a[:, None, :], torch.bmm(Cxx, a[:, :, None]))
    yy = torch.bmm(b[:, None, :], torch.bmm(Cyy, b[:, :, None]))
    return xy - 0.5 * xx - 0.5 * yy


def gaussianKernel(a, x, b, y, sig):
    Cxy = (-dist_matrix(x, y, 2) / sig**2).exp()
    Cxx = (-dist_matrix(x, x, 2) / sig**2).exp()
    Cyy = (-dist_matrix(y, y, 2) / sig**2).exp()
    xy = torch.bmm(a[:, None, :], torch.bmm(Cxy, b[:, :, None]))
    xx = torch.bmm(a[:, None, :], torch.bmm(Cxx, a[:, :, None]))
    yy = torch.bmm(b[:, None, :], torch.bmm(Cyy, b[:, :, None]))
    return - xy + 0.5 * xx + 0.5 * yy
