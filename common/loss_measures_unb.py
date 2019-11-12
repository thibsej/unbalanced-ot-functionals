import numpy as np
import torch
from .utils import dist_matrix, scal
from .sinkhorn_batch_unb import dist_matrix, sink, sym_sink



def legendre_entropy(mode, reach):
    if mode == 'balanced':
        def phis(x):
            return x
        def partial_phis(x):
            return 0

    if mode == 'kullback':
        def phis(x):
            return reach * ((x / reach).exp() - 1)
        def partial_phis(x):
            return (x / reach).exp()

    if mode == 'range':
        def phis(x):
            return torch.max(reach[0] * x, reach[1] * x)
        def partial_phis(x):
            out = torch.max( - reach[0] * x.sign(), reach[1] * x.sign() )
            out[x == 0] = reach[0]
            return out

    if mode == 'total-variation':
        def phis(x):
            return x
        def partial_phis(x):
            out =torch.ones_like(x)
            #out[x.abs() >= reach] = 0
            return out

    if mode == 'power-entropy':
        if reach[0] == 0:
            def phis(x):
                return - reach[1] * ( 1 - (x / reach[1]) ).log()
        else:
            def phis(x):
                return reach[1] * (1 - 1 / reach[0]) * ( (1 + (x / reach[1]) / (reach[0] - 1))**reach[0] - 1 )

        def partial_phis(x):
            return ( 1 - (x / (reach[1] * (1-reach[0]))) ) ** (reach[0] - 1)
    return phis, partial_phis



def output_potential(loss, blur, mode, reach=None):
    """
    It formats the potential to give an evaluation of the divergence of the form
    < a , F(f) > + < b , G(g) >
    """
    phis, partial_phis = legendre_entropy(mode, reach)
    if loss in ['regularized', 'sinkhorn']:
        output = lambda x: - phis(-x) - 0.5 * blur * partial_phis(-x)
    if loss in ['hausdorff']:
        output = lambda x: phis(-x) + blur * partial_phis(-x)
    return output


#######################################################################################################################
# Derived Functionals .....................................................................
#######################################################################################################################

def regularized_ot(a, x, b, y, p=2, blur=.1, mode='balanced', reach=None, nits=100, tol=1e-3, assume_convergence=False):  # OT_eps
    g_y, f_x = sink(a, x, b, y, p=p, blur=blur, mode=mode, reach=reach, nits=nits, tol=tol, assume_convergence=assume_convergence)
    output = output_potential(loss='regularized', blur=blur, mode=mode, reach=reach)
    cost = scal(a, output(f_x)) + scal(b, output(g_y)) + blur * a.sum(1)[:,None] * b.sum(1)[:,None]
    print(f"Cost =  {scal(a, output(f_x))} + {scal(b, output(g_y))} + {(blur * a.sum(1)[:,None] * b.sum(1)[:,None])}")
    # if kwargs.get("output", "cost") == "potential": return (cost, f_x, g_y)
    # if kwargs.get("output", "cost") == "derivative": return (cost, output(f_x), output(g_y))
    return cost


def hausdorff_divergence(a, x, b, y, p=2, blur=.1, mode='balanced', reach=None, nits=100, tol=1e-3, assume_convergence=False):  # H_eps
    g_xy, f_x = sym_sink(a, x, y, p=p, blur=blur, mode=mode, reach=reach, nits=nits, tol=tol, assume_convergence=assume_convergence)
    f_yx, g_y = sym_sink(b, y, x, p=p, blur=blur, mode=mode, reach=reach, nits=nits, tol=tol, assume_convergence=assume_convergence)
    output = output_potential(loss='hausdorff', blur=blur, mode=mode, reach=reach)
    cost = scal(a, output(f_x) - output(f_yx) + blur * (a.sum(1) - b.sum(1))[:,None]) +\
           scal(b, output(g_y) - output(g_xy) + blur * (b.sum(1) - a.sum(1))[:,None])
    # if kwargs.get("output", "cost") == "potential": return (cost, (f_yx, f_x), (g_xy, g_y))
    # if kwargs.get("output", "cost") == "derivative": return (cost,
                                                             # output(f_x) - output(g_xy) + blur * (a.sum(1) - b.sum(1)),
                                                             # output(f_x) - output(g_xy) + blur * (b.sum(1) - a.sum(1)))
    return cost


def sinkhorn_divergence(a, x, b, y, p=2, blur=.1, mode='balanced', reach=None, nits=100, tol=1e-3, assume_convergence=False):  # S_eps
    g_yx, f_xy = sink(a, x, b, y, p=p, blur=blur, mode=mode, reach=reach, nits=nits, tol=tol, assume_convergence=assume_convergence)
    _, f_x = sym_sink(a, x, p=p, blur=blur, mode=mode, reach=reach, nits=nits, tol=tol, assume_convergence=assume_convergence)
    _, g_y = sym_sink(b, y, p=p, blur=blur, mode=mode, reach=reach, nits=nits, tol=tol, assume_convergence=assume_convergence)
    output = output_potential(loss='sinkhorn', blur=blur, mode=mode, reach=reach)
    cost = scal(a, output(f_xy) - output(f_x)) + scal(b, output(g_yx) - output(g_y))
    # if kwargs.get("output", "cost") == "potential": return (cost, (f_xy, f_x), (g_yx, g_y))
    # if kwargs.get("output", "cost") == "derivative": return (cost, output(f_xy) - output(f_x),
                                                             # output(g_yx) - output(g_y))
    return cost


def sharp_ot(a, x, b, y, p=2, blur=.1, mode='balanced', reach=None):
    C_e = dist_matrix(x, y, p, blur)
    a_y, b_x = sink(a, x, b, y)
    pot = (b_x[:, :, None] + a_y[:, None, :]) / blur
    if mode in ['log-ref', 'exp-ref']:
        ref = a[:, :, None] * b[:, None, :]
        return (blur * ref * C_e * torch.exp((pot - C_e))).sum(2).sum(1)
    if mode in ['log-unif', 'exp-unif']:
        return (blur * C_e * torch.exp(pot - C_e)).sum(2).sum(1)


def energyDistance(a, x, b, y, p=1):
    if (torch.equal(x, y)):
        Cxy = dist_matrix(x, y, p, 1)
        return -.5 * torch.bmm((a - b)[:, None, :], torch.bmm(Cxy, (a - b)[:, :, None]))
    Cxy = dist_matrix(x, y, p, 1)
    Cxx = dist_matrix(x, x, p, 1)
    Cyy = dist_matrix(y, y, p, 1)
    xy = torch.bmm(a[:, None, :], torch.bmm(Cxy, b[:, :, None]))
    xx = torch.bmm(a[:, None, :], torch.bmm(Cxx, a[:, :, None]))
    yy = torch.bmm(b[:, None, :], torch.bmm(Cyy, b[:, :, None]))
    return xy - .5 * xx - .5 * yy