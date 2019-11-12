import torch
from .entropy import Entropy
from .utils import scal, dist_matrix, convolution, softmin, sym_softmin

def sinkhorn_asym(a_i, x_i, b_j, y_j, p, entropy, nits=100, tol=1e-3, assume_convergence=False):
    if type(nits) in [list, tuple]: nits = nits[0]
    torch.set_grad_enabled(not assume_convergence)
    f_i, g_j = entropy.init_potential()(a_i, x_i, b_j, y_j, p)
    softmin_x, softmin_y = softmin(a_i, x_i, b_j, y_j, p)
    aprox = entropy.aprox()
    err = entropy.error_sink()
    blur = entropy.blur
    i = 0
    while i < nits - 1:
        f_i_prev = f_i
        g_j = - aprox( - softmin_x(f_i, blur))
        f_i = - aprox( - softmin_y(g_j, blur))
        if err(f_i, f_i_prev) < tol: break
        i += 1

    torch.set_grad_enabled(True)
    if not assume_convergence:
        g_j = - aprox(- softmin_x(f_i, blur))
        f_i = - aprox(- softmin_y(g_j, blur))
    else:
        softmin_x, _ = softmin(a_i, x_i.detach(), b_j, y_j, p)
        _, softmin_y = softmin(a_i, x_i, b_j, y_j.detach(), p)
        g_j = - aprox(- softmin_x(f_i.detach(), blur))
        f_i = - aprox(- softmin_y(g_j.detach(), blur))

    return f_i, g_j


def sinkhorn_sym(a_i, x_i, p, entropy, y_j=None, nits=100, tol=1e-3, assume_convergence=False):
    if type(nits) in [list, tuple]: nits = nits[1]
    torch.set_grad_enabled(not assume_convergence)
    f_i, _ = entropy.init_potential()(a_i, x_i, a_i, x_i, p)
    softmin_xx = sym_softmin(a_i, x_i, x_i, p)
    aprox = entropy.aprox()
    err = entropy.error_sink()
    blur = entropy.blur
    i = 0
    while i < (nits - 1):
        f_i_prev = f_i
        f_i = 0.5 * ( f_i - aprox(- softmin_xx(f_i, blur)) )
        if err(f_i, f_i_prev).item() < tol: break
        i += 1

    torch.set_grad_enabled(True)
    if not assume_convergence:
        S_x = sym_softmin(a_i, x_i.detach(), x_i, p)
        if y_j is not None:
            S2_x = sym_softmin(a_i, x_i, y_j, p)
            f_x, f_xy = -aprox( - S_x(f_i, blur) ), -aprox( - S2_x(f_i, blur) )
        else:
            f_x = -aprox( - S_x(f_i, blur) )
    else:
        S_x = sym_softmin(a_i, x_i.detach(), x_i, p)
        if y_j is not None:
            S2_x = sym_softmin(a_i, x_i.detach(), y_j, p)
            f_x, f_xy = -aprox( - S_x(f_i.detach(), blur) ), -aprox( - S2_x(f_i.detach(), blur) )
        else:
            f_x = -aprox( - S_x(f_i.detach(), blur) )
    if y_j is None:
        return None, f_x
    else:
        return f_xy, f_x