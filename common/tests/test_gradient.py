import pytest

import torch

from common.functional import regularized_ot, hausdorff_divergence, sinkhorn_divergence, energyDistance
from common.sinkhorn import BatchVanillaSinkhorn, BatchExpSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure, convolution, scal, dist_matrix

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=10)


@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [regularized_ot])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True),
                                  BatchExpSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True)])
def test_gradient_unbalanced_weight_and_position_asym(solv, div, entropy, reach, p, m, n):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    x.requires_grad = True
    b, y = generate_measure(1, 6, 2)
    f, g = solv.sinkhorn_asym(a, x, n * b, y, p, entropy)
    cost = entropy.output_regularized(a, x, n * b, y, p, f, g)
    [grad_num_x, grad_num_a] = torch.autograd.grad(cost, [x, a])
    grad_th_a = - entropy.legendre_entropy(-f) + entropy.blur * n - entropy.blur * \
              ((f[:,:,None ] + g[:,None,:] - dist_matrix(x, y, p)).exp() * n * b[:,None,:]).sum(dim=2)
    pi = n * a[:, :, None] * b[:, None, :] * (
                (f[:, :, None] + g[:, None, :] - dist_matrix(x, y, p)) / entropy.blur).exp()
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum('ijk, ikl->ijl', pi, y)
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)


@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('m', [1., 0.7, 1.5])
@pytest.mark.parametrize('entropy', [Balanced(1e0)])
@pytest.mark.parametrize('div', [regularized_ot])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True),
                                  BatchExpSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True)])
def test_gradient_unbalanced_weight_and_position_asym(solv, div, entropy, p, m):
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    a, b = m * a, m * b
    a.requires_grad = True
    x.requires_grad = True
    f, g = solv.sinkhorn_asym(a, x, b, y, p, entropy)
    cost = entropy.output_regularized(a, x, b, y, p, f, g)
    [grad_num_x, grad_num_a] = torch.autograd.grad(cost, [x, a])
    grad_th_a = f
    pi = a[:, :, None] * b[:, None, :] * (
                (f[:, :, None] + g[:, None, :] - dist_matrix(x, y, p)) / entropy.blur).exp()
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum('ijk, ikl->ijl', pi, y)
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)

# TODO Debug symmetric gradient
@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m', [1., 0.7, 1.5])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [regularized_ot])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True),
                                  BatchExpSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True)])
def test_gradient_unbalanced_weight_and_position_asym(solv, div, entropy, reach, p, m):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    x.requires_grad = True
    _, f = solv.sinkhorn_asym(a, x, a, x, p, entropy)
    cost = entropy.output_regularized(a, x, a, x, p, f, f)
    [grad_num_x, grad_num_a] = torch.autograd.grad(cost, [x, a])
    grad_th_a = - entropy.legendre_entropy(-f) + entropy.blur * m - entropy.blur * \
              ((f[:,:,None ] + f[:,None,:] - dist_matrix(x, x, p)).exp() * a[:,None,:]).sum(dim=2)
    pi = a[:, :, None] * a[:, None, :] * (
                (f[:, :, None] + f[:, None, :] - dist_matrix(x, x, p)) / entropy.blur).exp()
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum('ijk, ikl->ijl', pi, x)
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)


@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('m', [1., 0.7, 1.5])
@pytest.mark.parametrize('entropy', [Balanced(1e0)])
@pytest.mark.parametrize('div', [regularized_ot])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True),
                                  BatchExpSinkhorn(nits=5000, nits_grad=20,  tol=1e-14, assume_convergence=True)])
def test_gradient_unbalanced_weight_and_position_sym(solv, div, entropy, p, m):
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    x.requires_grad = True
    _, f = solv.sinkhorn_sym(a, x, p, entropy)
    cost = entropy.output_regularized(a, x, a, x, p, f, f)
    [grad_num_x, grad_num_a] = torch.autograd.grad(cost, [x, a])
    grad_th_a = f
    pi = a[:, :, None] * a[:, None, :] * (
                (f[:, :, None] + f[:, None, :] - dist_matrix(x, x, p)) / entropy.blur).exp()
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum('ijk, ikl->ijl', pi, x)
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)