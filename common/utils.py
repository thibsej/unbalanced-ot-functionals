import numpy as np
import torch


def scal(a, f):
    return torch.sum(a * f, dim=1)


def dist_matrix(x_i, y_j, p):
    if p == 1:
        return (x_i[:, :, None, :] - y_j[:, None, :, :]).norm(p=2, dim=3)
    elif p == 2:
        return (x_i[:, :, None, :] - y_j[:, None, :, :]).norm(p=2, dim=3) ** 2
    else:
        C_e = (x_i[:, :, None, :] - y_j[:, None, :, :]).norm(p=2, dim=3)
        return C_e ** (p)


def convolution(a, x, b, y, p):
    C = dist_matrix(x, y, p)
    return torch.bmm(C, b[:, :, None]).squeeze(), torch.bmm(C.transpose(1, 2), a[:, :, None]).squeeze()


def softmin(a_i, x_i, b_j, y_j, p):
    """
    Outputs the fixed point mapping (S_x, S_y) of Sinkhorn iterations, i.e.
    mappings such that at convergence, f = S_y(g) and g = S_x(f).
    """
    C_e = dist_matrix(x_i, y_j, p)
    a_i_log, b_j_log = a_i.log(), b_j.log()
    softmin_x = lambda f_i, ep: - ep * ((f_i / ep + a_i_log)[:, None, :] - C_e.transpose(1, 2) / ep).logsumexp(dim=2)
    softmin_y = lambda f_j, ep: - ep * ((f_j / ep + b_j_log)[:, None, :] - C_e / ep).logsumexp(dim=2)
    return softmin_x, softmin_y


def sym_softmin(a_i, x_i, y_j, p):
    """
    Outputs the fixed point mapping (S_x, S_y) of Sinkhorn iterations, i.e.
    mappings such that at convergence, f = S_y(g) and g = S_x(f).
    """
    C_e = dist_matrix(x_i, y_j, p)
    a_i_log = a_i.log()
    softmin_x = lambda f_j, ep: - ep * ((f_j / ep + a_i_log)[:, None, :] - C_e.transpose(1, 2) / ep).logsumexp(dim=2)
    return softmin_x


def generate_measure(n_batch, n_sample, n_dim):
    """
    Generate a batch of probability measures in R^d sampled over the unit square
    :param n_batch: Number of batches
    :param n_sample: Number of sampling points in R^d
    :param n_dim: Dimension of the feature space
    :return: A (Nbatch, Nsample, Ndim) torch.Tensor
    """
    m = torch.distributions.exponential.Exponential(1.0)
    a = m.sample(torch.Size([n_batch, n_sample]))
    a = a / a.sum(dim=1)[:,None]
    m = torch.distributions.uniform.Uniform(0.0, 1.0)
    x = m.sample(torch.Size([n_batch, n_sample, n_dim]))
    return a, x


def generate_gaussian_measure(n_batch, n_sample, n_dim):
    a = torch.ones(n_batch, n_sample)
    a = a / a.sum(dim=1)[:, None]
    m = torch.distributions.normal.Normal(0.0, 1.0)
    x = m.sample(torch.Size([n_batch, n_sample, n_dim]))
    return a, x
