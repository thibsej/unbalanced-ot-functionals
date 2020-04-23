import torch
import numpy as np
from .entropy import Entropy
from .utils import dist_matrix, softmin, exp_softmin

#TODO: Make a robust sinkhorn that calls exp-sinkhorn if stable and log-sinkhorn otherwise

class SinkhornSolver(object):

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, p, entropy, f_i=None, g_j=None):
        """Computes the Sinkhorn algorithm for two different measures"""
        raise NotImplementedError

    def sinkhorn_sym(self, a_i, x_i, p, entropy, y_j=None, f_i=None):
        """Computes the symmetric potential for a given measure, and its extrapolation if required"""
        raise NotImplementedError


class BatchVanillaSinkhorn(SinkhornSolver):

    def __init__(self, nits, nits_grad, tol, assume_convergence):
        self.nits = nits
        self.nits_grad = nits_grad
        self.tol = tol
        self.assume_convergence = assume_convergence

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, cost, entropy, f_i=None, g_j=None):
        if type(self.nits) in [list, tuple]: nits = self.nits[0]
        torch.set_grad_enabled(not self.assume_convergence)
        if f_i is None or g_j is None:
            f_i, g_j = entropy.init_potential(a_i, x_i, b_j, y_j, cost)
        C = cost(x_i, y_j)
        softmin_x, softmin_y = softmin(a_i, C, b_j)
        aprox = entropy.aprox
        err = entropy.error_sink
        blur = entropy.blur
        for i in range(self.nits - self.nits_grad):
            f_i_prev = f_i
            g_j = - aprox( - softmin_x(f_i, blur))
            f_i = - aprox( - softmin_y(g_j, blur))
            if err(f_i, f_i_prev) < self.tol: break

        torch.set_grad_enabled(True)
        if self.assume_convergence:
            C = cost(x_i, y_j)
            softmin_x, softmin_y = softmin(a_i, C, b_j)
        for i in range(self.nits_grad):
            g_j = - aprox(- softmin_x(f_i, blur))
            f_i = - aprox(- softmin_y(g_j, blur))
        return f_i, g_j

    def sinkhorn_sym(self, a_i, x_i, cost, entropy, y_j=None, f_i=None):
        if type(self.nits) in [list, tuple]: nits = self.nits[1]
        torch.set_grad_enabled(not self.assume_convergence)
        if f_i is None:
            f_i, _ = entropy.init_potential(a_i, x_i, a_i, x_i, cost)
        C = cost(x_i, x_i)
        softmin_xx, _ = softmin(a_i, C)
        aprox = entropy.aprox
        err = entropy.error_sink
        blur = entropy.blur
        for i in range(self.nits - self.nits_grad):
            f_i_prev = f_i
            f_i = 0.5 * ( f_i - aprox(- softmin_xx(f_i, blur)) )
            if err(f_i, f_i_prev).item() < self.tol: break

        torch.set_grad_enabled(True)
        if self.assume_convergence:
            C = cost(x_i, x_i)
            softmin_xx, _ = softmin(a_i, C)
        for i in range(self.nits_grad):
            f_i = 0.5 * ( f_i - aprox(- softmin_xx(f_i, blur)) )

        if y_j is not None:
            C = cost(x_i, y_j)
            softmin_xy, _ = softmin(a_i, C)
            f_xy = - aprox(- softmin_xy(f_i, blur))
            return f_xy, f_i
        else:
            return None, f_i


class BatchScalingSinkhorn(SinkhornSolver):

    def __init__(self, budget, nits_grad, assume_convergence):
        self.budget = budget
        self.nits_grad = nits_grad
        self.assume_convergence = assume_convergence

    @staticmethod
    def epsilon_schedule(diameter, blur, budget):
        eps_s = [torch.tensor([diameter])] + \
                [torch.tensor([np.exp(e)]) for e in np.linspace(np.log(diameter), np.log(blur), budget)] \
                + [torch.tensor([blur])]
        return eps_s

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, cost, entropy, f_i=None, g_j=None):
        torch.set_grad_enabled(not self.assume_convergence)
        if f_i is None or g_j is None:
            f_i, g_j = entropy.init_potential(a_i, x_i, b_j, y_j, cost)
        C = cost(x_i, y_j)
        softmin_x, softmin_y = softmin(a_i, C, b_j)
        scales = self.epsilon_schedule(C.max().cpu().item(), entropy.blur, self.budget)
        aprox = entropy.aprox
        for scale in scales:
            g_j = - aprox(- softmin_x(f_i, scale))
            f_i = - aprox(- softmin_y(g_j, scale))

        torch.set_grad_enabled(True)
        if self.assume_convergence:
            C = cost(x_i, y_j)
            softmin_x, softmin_y = softmin(a_i, C, b_j)
        for i in range(self.nits_grad):
            g_j = - aprox(- softmin_x(f_i, entropy.blur))
            f_i = - aprox(- softmin_y(g_j, entropy.blur))
        return f_i, g_j

    def sinkhorn_sym(self, a_i, x_i, cost, entropy, y_j=None, f_i=None):
        if f_i is None:
            f_i, _ = entropy.init_potential(a_i, x_i, a_i, x_i, cost)
        C = cost(x_i, x_i)
        softmin_xx, _ = softmin(a_i, C)
        scales = self.epsilon_schedule(C.max(), entropy.blur, self.budget)
        aprox = entropy.aprox
        torch.set_grad_enabled(not self.assume_convergence)
        for scale in scales:
            f_i = 0.5 * (f_i - aprox(- softmin_xx(f_i, scale)))

        torch.set_grad_enabled(True)
        if self.assume_convergence:
            C = cost(x_i, x_i)
            softmin_xx, _ = softmin(a_i, C)
        for i in range(self.nits_grad):
            f_i = 0.5 * (f_i - aprox(- softmin_xx(f_i, entropy.blur)))

        if y_j is not None:
            C = cost(x_i, y_j)
            softmin_xy, _ = softmin(a_i, C)
            f_xy = - aprox(- softmin_xy(f_i, entropy.blur))
            return f_xy, f_i
        else:
            return None, f_i


class BatchExpSinkhorn(SinkhornSolver):

    def __init__(self, nits, nits_grad, tol, assume_convergence):
        self.nits = nits
        self.nits_grad = nits_grad
        self.tol = tol
        self.assume_convergence = assume_convergence

    def sinkhorn_asym(self, a_i, x_i, b_j, y_j, cost, entropy, f_i=None, g_j=None):
        if type(self.nits) in [list, tuple]: nits = self.nits[0]
        torch.set_grad_enabled(not self.assume_convergence)
        if f_i is None or g_j is None:
            f_i, g_j = entropy.init_potential(a_i, x_i, b_j, y_j, cost)
        u_i, v_j = (f_i / entropy.blur).exp(), (g_j / entropy.blur).exp()

        K = (- cost(x_i, y_j) / entropy.blur).exp()
        if ~K.gt(torch.zeros_like(K)).all():  # exits if K has zeros to avoid underflow
            return None, None

        softmin_x, softmin_y = exp_softmin(a_i, K, b_j)
        prox = entropy.kl_prox
        err = entropy.error_sink
        for i in range(self.nits - self.nits_grad):
            u_i_prev = u_i
            v_j = prox(softmin_x(u_i))
            u_i = prox(softmin_y(v_j))
            if err(entropy.blur * u_i.log(), entropy.blur * u_i_prev.log()) < self.tol: break

        torch.set_grad_enabled(True)
        if self.assume_convergence:
            K = (- cost(x_i, y_j) / entropy.blur).exp()
            softmin_x, softmin_y = exp_softmin(a_i, K, b_j)
        for i in range(self.nits_grad):
            v_j = prox(softmin_x(u_i))
            u_i = prox(softmin_y(v_j))
        return entropy.blur * u_i.log(), entropy.blur * v_j.log()

    def sinkhorn_sym(self, a_i, x_i, cost, entropy, y_j=None, f_i=None):
        if type(self.nits) in [list, tuple]: nits = self.nits[1]
        torch.set_grad_enabled(not self.assume_convergence)
        if f_i is None:
            f_i, _ = entropy.init_potential(a_i, x_i, a_i, x_i, cost)
        u_i = (f_i / entropy.blur).exp()

        K = (- cost(x_i, x_i) / entropy.blur).exp()
        if ~K.gt(torch.zeros_like(K)).all():  # exits if K has zeros to avoid underflow
            return None, None

        softmin_xx, _ = exp_softmin(a_i, K)
        prox = entropy.kl_prox
        err = entropy.error_sink
        for i in range(self.nits - self.nits_grad):
            u_i_prev = u_i
            u_i = (u_i * prox(softmin_xx(u_i))).sqrt()
            if err(entropy.blur * u_i.log(), entropy.blur * u_i_prev.log()) < self.tol: break

        torch.set_grad_enabled(True)
        if self.assume_convergence:
            K = (- cost(x_i, x_i) / entropy.blur).exp()
            softmin_xx, _ = exp_softmin(a_i, K)
        for i in range(self.nits_grad):
            u_i = prox(softmin_xx(u_i))

        if y_j is not None:
            Ky = (- cost(x_i, y_j) / entropy.blur).exp()
            softmin_xy, _ = exp_softmin(a_i, Ky)
            u_xy = prox(softmin_xy(u_i))
            return entropy.blur * u_xy.log(), entropy.blur * u_i.log()
        else:
            return None, entropy.blur * u_i.log()
