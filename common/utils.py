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


def softmin(a_i, C, b_j=None):
    """
    Outputs the fixed point mapping (S_x, S_y) of Sinkhorn iterations, i.e.
    mappings such that at convergence, f = S_y(g) and g = S_x(f).
    """
    a_i_log = a_i.log()
    softmin_x = lambda f_i, ep: - ep * ((f_i / ep + a_i_log)[:, None, :] - C.transpose(1, 2) / ep).logsumexp(dim=2)
    if b_j is not None:
        b_j_log = b_j.log()
        softmin_y = lambda f_j, ep: - ep * ((f_j / ep + b_j_log)[:, None, :] - C / ep).logsumexp(dim=2)
        return softmin_x, softmin_y
    else:
        return softmin_x, None


def exp_softmin(a_i, K, b_j=None):
    """
    Outputs the fixed point mapping (S_x, S_y) of Sinkhorn iterations, i.e.
    mappings such that at convergence, f = S_y(g) and g = S_x(f).
    Exponential form which is not stabilized.
    """
    softmin_x = lambda f_i: torch.einsum('ijk,ij->ik', K, f_i * a_i)
    if b_j is not None:
        softmin_y = lambda f_j: torch.einsum('ijk,ik->ij', K, f_j * b_j)
        return softmin_x, softmin_y
    else:
        return softmin_x, None


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
