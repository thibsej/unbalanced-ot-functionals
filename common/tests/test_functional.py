import pytest

import torch
from common.functional import regularized_ot, hausdorff_divergence, sinkhorn_divergence, energyDistance
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure, convolution, scal

torch.set_printoptions(precision=10)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('m', [1., 0.7, 2.])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [hausdorff_divergence, sinkhorn_divergence])
def test_divergence_zero(div, entropy, p, m):
    a, x = generate_measure(1, 5, 2)
    cost = div(m * a, x, m * a, x, p, entropy, nits=10000, tol=0)
    assert torch.allclose(cost, torch.Tensor([0.0]), atol=1e-6)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
@pytest.mark.parametrize('div', [hausdorff_divergence, sinkhorn_divergence])
def test_divergence_positivity(div, entropy, p, m, n):
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    cost = div(m * a, x, n * b, y, p, entropy, nits=10000, tol=0)
    assert torch.ge(cost, 0.0).all()

# TODO: Need to debug the Power entropy (why is there a negative value of the loss ?)
@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e4, 1e0), TotalVariation(1e4, 1e0), Range(1e4, 0.3, 2),
                                     PowerEntropy(1e4, 1e0, 0), PowerEntropy(1e4, 1e0, -1)])
def test_consistency_infinite_blur_regularized_ot_unbalanced(entropy, p, m, n):
    torch.set_default_dtype(torch.float64)
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    phi = entropy.entropy()
    f, g = convolution(a, x, b, y, p)
    control = scal(a, f) + m * phi(torch.Tensor([n])) + n * phi(torch.Tensor([m]))
    cost = regularized_ot(m * a, x, n * b, y, p, entropy, nits=10000, tol=0)
    assert torch.allclose(cost, control, atol=1e-0)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('entropy', [Balanced(1e5)])
def test_consistency_infinite_blur_regularized_ot_balanced(entropy, p):
    """Control consistency in OT_eps when eps goes to infinity, especially for balanced OT"""
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    f, g = convolution(a, x, b, y, p)
    control = scal(a, f)
    cost = regularized_ot(a, x, b, y, p, entropy, nits=10000, tol=0)
    assert torch.allclose(cost, control, atol=1e-0)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e6, 1e0), Balanced(1e6), TotalVariation(1e6, 1e0),
                                     Range(1e6, 0.3, 2), PowerEntropy(1e6, 1e0, 0), PowerEntropy(1e6, 1e0, -1)])
def test_consistency_infinite_blur_sinkhorn_div(entropy, p):
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    control = energyDistance(a, x, b, y, p)
    cost = sinkhorn_divergence(a, x, b, y, p, entropy, nits=10000, tol=0)
    assert torch.allclose(cost, control, atol=1e-0)
