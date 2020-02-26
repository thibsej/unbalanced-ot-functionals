import pytest

import torch

from common.functional import regularized_ot, hausdorff_divergence, sinkhorn_divergence, energyDistance
from common.sinkhorn import BatchVanillaSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure, convolution, scal, dist_matrix

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=10)
solver = BatchVanillaSinkhorn(nits=5000, tol=1e-16, assume_convergence=True)


@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, tol=1e-14, assume_convergence=True),
                                    BatchVanillaSinkhorn(nits=5000, tol=1e-14, assume_convergence=False)])
def test_gradient_unbalanced_position(solv, entropy, reach, p, m, n):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 1)
    x.requires_grad = True
    b, y = generate_measure(1, 6, 1)
    f, g = solv.sinkhorn_asym(m * a, x, n * b, y, p, entropy)
    cost = entropy.output_regularized(m * a, x, n * b, y, p, f, g)
    [grad_num] = torch.autograd.grad(cost, [x])
    pi = m * n * a[:, :, None] * b[:, None, :] * ((f[:, :, None] + g[:, None, :] - dist_matrix(x, y, p)) / entropy.blur).exp()
    grad_th = 2 * x * pi.sum(dim=2)[:,:,None] - 2 * torch.einsum('ijk, ikl->ijl', pi, y)
    print(f"Theoretical gradient = {grad_th}")
    print(f"Autograd gradient = {grad_num}")
    assert torch.allclose(grad_num, grad_th, rtol=1e-5)

#TODO: Modify and debug tests that follows from here


@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [regularized_ot])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, tol=1e-14, assume_convergence=True),
                                    BatchVanillaSinkhorn(nits=5000, tol=1e-14, assume_convergence=False)])
def test_gradient_unbalanced_weight(solv, div, entropy, reach, p, m, n):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    b, y = generate_measure(1, 6, 2)
    f, g = solv.sinkhorn_asym(a, x, n * b, y, p, entropy)
    cost = entropy.output_regularized(a, x, n * b, y, p, f, g)
    [grad_num] = torch.autograd.grad(cost, [a])
    grad_th = - entropy.legendre_entropy(-f) + entropy.blur * n - entropy.blur * \
              ((f[:,:,None ] + g[:,None,:] - dist_matrix(x, y, p)).exp() * n * b[:,None,:]).sum(dim=2)
    print(f"Theoretical gradient = {grad_th}")
    print(f"Autograd gradient = {grad_num}")
    print(f"gradient ratio = {grad_num / grad_th}")
    assert torch.allclose(m * grad_th, grad_num, rtol=1e-5)


@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [regularized_ot])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=5000, tol=1e-14, assume_convergence=True),
                                    BatchVanillaSinkhorn(nits=5000, tol=1e-14, assume_convergence=False)])
def test_gradient_unbalanced_weight_and_position(solv, div, entropy, reach, p, m, n):
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


@pytest.mark.parametrize('p', [1.5, 2])
@pytest.mark.parametrize('entropy', [Balanced(1e0)])
@pytest.mark.parametrize('div', [regularized_ot])
def test_gradient_balanced_position(div, entropy, p):
    a, x = generate_measure(1, 5, 2)
    x.requires_grad = True
    delta_a, delta_x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    t = 1e-5
    cost_0 = div(a, x, b, y, p, entropy, solver=solver)
    cost_delta = div(a, x + t * delta_x, b, y, p, entropy, solver=solver)
    [g] = torch.autograd.grad(cost_delta, [x])
    num_grad = torch.sum(g * delta_x)
    discrete_grad = (cost_delta.item() - cost_0) / t
    assert torch.allclose(num_grad, discrete_grad, rtol=1e-1)


@pytest.mark.parametrize('p', [1.5, 2])
@pytest.mark.parametrize('entropy', [Balanced(1e0)])
@pytest.mark.parametrize('div', [sinkhorn_divergence, hausdorff_divergence, regularized_ot])
def test_gradient_balanced_weight(div, entropy, p):
    a, x = generate_measure(1, 5, 2)
    a.requires_grad = True
    delta_a, delta_x = generate_measure(1, 5, 2)
    delta_a = delta_a - delta_a.mean()
    b, y = generate_measure(1, 6, 2)
    t = 1e-5
    cost_0 = div(a, x, b, y, p, entropy, solver=solver)
    cost_delta = div(a + t * delta_a, x, b, y, p, entropy, solver=solver)
    [g] = torch.autograd.grad(cost_delta, [a])
    num_grad = torch.sum(g * delta_x)
    discrete_grad = (cost_delta.item() - cost_0) / t
    assert torch.allclose(num_grad, discrete_grad, rtol=1e-1)