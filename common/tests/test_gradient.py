import pytest

import torch

from common.functional import regularized_ot, hausdorff_divergence, sinkhorn_divergence, energyDistance
from common.sinkhorn import BatchVanillaSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure, convolution, scal

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=10)
solver = BatchVanillaSinkhorn(nits=5000, tol=1e-12, assume_convergence=True)

# TODO: Code a test on autograd
# TODO: test with finite differences
# TODO: test for both symmetric and non-symmetric sinkhorn
@pytest.mark.parametrize('p', [1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [sinkhorn_divergence, hausdorff_divergence, regularized_ot])
def test_gradient_unbalanced_position(div, entropy, reach, p, m, n):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    x.requires_grad = True
    delta_a, delta_x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    t = 1e-5
    cost_0 = div(m * a, x, n * b, y, p, entropy, solver=solver)
    cost_delta = div(m * a, x + t * delta_x, n * b, y, p, entropy, solver=solver)
    [g] = torch.autograd.grad(cost_delta, [x])
    assert torch.allclose(torch.sum(g * delta_x), (cost_delta.item() - cost_0) / t, atol=1e-0)


@pytest.mark.parametrize('p', [1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), TotalVariation(1e0, 1e0), Range(1e0, 0.3, 2),
                                     PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [sinkhorn_divergence, hausdorff_divergence, regularized_ot])
def test_gradient_unbalanced_weight(div, entropy, reach, p, m, n):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    a.requires_grad = True
    delta_a, delta_x = generate_measure(1, 5, 2)
    delta_a = delta_a - delta_a.mean()
    b, y = generate_measure(1, 6, 2)
    t = 1e-5
    cost_0 = div(m * a, x, n * b, y, p, entropy, solver=solver)
    cost_delta = div(m * a + t * delta_a, x, n * b, y, p, entropy, solver=solver)
    [g] = torch.autograd.grad(cost_delta, [a])
    assert torch.allclose(torch.sum(g * delta_a), (cost_delta.item() - cost_0) / t, atol=1e-0)


@pytest.mark.parametrize('p', [1.5, 2])
@pytest.mark.parametrize('entropy', [Balanced(1e0)])
@pytest.mark.parametrize('div', [sinkhorn_divergence, hausdorff_divergence, regularized_ot])
def test_gradient_balanced_position(div, entropy, p):
    a, x = generate_measure(1, 5, 2)
    x.requires_grad = True
    delta_a, delta_x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    t = 1e-5
    cost_0 = div(a, x, b, y, p, entropy, solver=solver)
    cost_delta = div(a, x + t * delta_x, b, y, p, entropy, solver=solver)
    [g] = torch.autograd.grad(cost_delta, [x])
    assert torch.allclose(torch.sum(g * delta_x), (cost_delta.item() - cost_0) / t, atol=1e-0)


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
    assert torch.allclose(torch.sum(g * delta_a), (cost_delta.item() - cost_0) / t, atol=1e-0)