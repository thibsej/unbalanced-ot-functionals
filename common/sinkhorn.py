import torch
import numpy as np
from .entropy import Entropy
from .utils import scal, dist_matrix, convolution, softmin, sym_softmin


class SinkhornSolver(object):

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, p, entropy):
        """Computes the Sinkhorn algorithn for two different measures"""
        raise NotImplementedError

    def sinkhorn_sym(self, a_i, x_i, p, entropy, y_j=None):
        """Computes the symmetric potential for a diven measure, and its extrapolation if required"""
        raise NotImplementedError


class BatchVanillaSinkhorn(SinkhornSolver):

    def __init__(self, nits, tol, assume_convergence):
        self.nits = nits
        self.tol = tol
        self.assume_convergence = assume_convergence

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, p, entropy):
        if type(self.nits) in [list, tuple]: nits = self.nits[0]
        torch.set_grad_enabled(not self.assume_convergence)
        f_i, g_j = entropy.init_potential()(a_i, x_i, b_j, y_j, p)
        softmin_x, softmin_y = softmin(a_i, x_i, b_j, y_j, p)
        aprox = entropy.aprox()
        err = entropy.error_sink()
        blur = entropy.blur
        i = 0
        while i < self.nits - 1:
            f_i_prev = f_i
            g_j = - aprox( - softmin_x(f_i, blur))
            f_i = - aprox( - softmin_y(g_j, blur))
            if err(f_i, f_i_prev) < self.tol: break
            i += 1

        torch.set_grad_enabled(True)
        if not self.assume_convergence:
            g_j = - aprox(- softmin_x(f_i, blur))
            f_i = - aprox(- softmin_y(g_j, blur))
        else:
            softmin_x, _ = softmin(a_i, x_i.detach(), b_j, y_j, p)
            _, softmin_y = softmin(a_i, x_i, b_j, y_j.detach(), p)
            g_j = - aprox(- softmin_x(f_i.detach(), blur))
            f_i = - aprox(- softmin_y(g_j.detach(), blur))

        return f_i, g_j


    def sinkhorn_sym(self, a_i, x_i, p, entropy, y_j=None):
        if type(self.nits) in [list, tuple]: nits = self.nits[1]
        torch.set_grad_enabled(not self.assume_convergence)
        f_i, _ = entropy.init_potential()(a_i, x_i, a_i, x_i, p)
        softmin_xx = sym_softmin(a_i, x_i, x_i, p)
        aprox = entropy.aprox()
        err = entropy.error_sink()
        blur = entropy.blur
        i = 0
        while i < (self.nits - 1):
            f_i_prev = f_i
            f_i = 0.5 * ( f_i - aprox(- softmin_xx(f_i, blur)) )
            if err(f_i, f_i_prev).item() < self.tol: break
            i += 1

        torch.set_grad_enabled(True)
        if not self.assume_convergence:
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


class BatchScalingSinkhorn(SinkhornSolver):

    def __init__(self, budget, assume_convergence):
        self.budget = budget
        self.assume_convergence = assume_convergence

    @staticmethod
    def epsilon_schedule(diameter, blur, budget):
        eps_s = [diameter] \
                + [np.exp(e) for e in np.linspace(np.log(diameter), np.log(blur), budget)] \
                + [blur]
        return eps_s

    @staticmethod
    def cost_softmin(a_i, b_j, C):
        a_i_log, b_j_log = a_i.log(), b_j.log()
        softmin_x = lambda f_i, ep: - ep * ((f_i / ep + a_i_log)[:, None, :] - C.transpose(1, 2) / ep).logsumexp(dim=2)
        softmin_y = lambda f_j, ep: - ep * ((f_j / ep + b_j_log)[:, None, :] - C / ep).logsumexp(dim=2)
        return softmin_x, softmin_y

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, p, entropy):
        torch.set_grad_enabled(not self.assume_convergence)
        f_i, g_j = entropy.init_potential()(a_i, x_i, b_j, y_j, p)
        C = dist_matrix(x_i, y_j, p)
        softmin_x, softmin_y = self.cost_softmin(a_i, b_j, C)
        scales = self.epsilon_schedule(C.max(), entropy.blur, self.budget)
        aprox = entropy.aprox()
        for scale in scales:
            g_j = - aprox(- softmin_x(f_i, scale))
            f_i = - aprox(- softmin_y(g_j, scale))

        torch.set_grad_enabled(True)
        if not self.assume_convergence:
            g_j = - aprox(- softmin_x(f_i, entropy.blur))
            f_i = - aprox(- softmin_y(g_j, entropy.blur))
        else:
            softmin_x, _ = softmin(a_i, x_i.detach(), b_j, y_j, p)
            _, softmin_y = softmin(a_i, x_i, b_j, y_j.detach(), p)
            g_j = - aprox(- softmin_x(f_i.detach(), entropy.blur))
            f_i = - aprox(- softmin_y(g_j.detach(), entropy.blur))

        return f_i, g_j

    def sinkhorn_sym(self, a_i, x_i, p, entropy, y_j=None):
        f_i, _ = entropy.init_potential()(a_i, x_i, a_i, x_i, p)
        C = dist_matrix(x_i, y_j, p)
        softmin_xx, _ = self.cost_softmin(a_i, a_i, C)
        scales = self.epsilon_schedule(C.max(), entropy.blur, self.budget)
        aprox = entropy.aprox()
        torch.set_grad_enabled(not self.assume_convergence)
        for scale in scales:
            f_i = 0.5 * (f_i - aprox(- softmin_xx(f_i, scale)))

        torch.set_grad_enabled(True)
        if not self.assume_convergence:
            S_x = sym_softmin(a_i, x_i.detach(), x_i, p)
            if y_j is not None:
                S2_x = sym_softmin(a_i, x_i, y_j, p)
                f_x, f_xy = -aprox(- S_x(f_i, entropy.blur)), -aprox(- S2_x(f_i, entropy.blur))
            else:
                f_x = -aprox(- S_x(f_i, entropy.blur))
        else:
            S_x = sym_softmin(a_i, x_i.detach(), x_i, p)
            if y_j is not None:
                S2_x = sym_softmin(a_i, x_i.detach(), y_j, p)
                f_x, f_xy = -aprox(- S_x(f_i.detach(), entropy.blur)), -aprox(- S2_x(f_i.detach(), entropy.blur))
            else:
                f_x = -aprox(- S_x(f_i.detach(), entropy.blur))
        if y_j is None:
            return None, f_x
        else:
            return f_xy, f_x
