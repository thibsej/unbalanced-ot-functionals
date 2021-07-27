import pytest

import torch

from unbalancedot.functional import (
    regularized_ot,
    hausdorff_divergence,
    sinkhorn_divergence,
    energyDistance,
)
from unbalancedot.sinkhorn import BatchVanillaSinkhorn
from unbalancedot.entropy import (
    KullbackLeibler,
    Balanced,
    TotalVariation,
    Range,
    PowerEntropy,
)
from unbalancedot.utils import (
    generate_measure,
    convolution,
    scal,
    euclidean_cost,
)

torch.set_printoptions(precision=10)
torch.set_default_tensor_type(torch.DoubleTensor)
solver = BatchVanillaSinkhorn(
    nits=5000, nits_grad=5, tol=1e-15, assume_convergence=True
)


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [1.0, 0.7, 2.0])
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e0, 1e0),
        Balanced(1e0),
        TotalVariation(1e0, 1e0),
        Range(1e0, 0.3, 2),
        PowerEntropy(1e0, 1e0, 0),
        PowerEntropy(1e0, 1e0, -1),
    ],
)
@pytest.mark.parametrize("div", [sinkhorn_divergence, hausdorff_divergence])
def test_divergence_zero(div, entropy, reach, p, m):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    func = div(m * a, x, m * a, x, cost, entropy, solver=solver)
    assert torch.allclose(func, torch.Tensor([0.0]), rtol=1e-6)


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [1.0, 0.7, 2.0])
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e0, 1e0),
        Balanced(1e0),
        TotalVariation(1e0, 1e0),
        Range(1e0, 0.3, 2),
        PowerEntropy(1e0, 1e0, 0),
        PowerEntropy(1e0, 1e0, -1),
    ],
)
def test_consistency_regularized_sym_asym(entropy, reach, p, m):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    f_xy, g_xy = solver.sinkhorn_asym(a, x, a, x, cost, entropy)
    _, f_xx = solver.sinkhorn_sym(a, x, cost, entropy)
    func_asym = entropy.output_regularized(a, x, a, x, cost, f_xy, g_xy)
    func_sym = entropy.output_regularized(a, x, a, x, cost, f_xx, f_xx)
    assert torch.allclose(func_asym, func_sym, rtol=1e-6)


# TODO: Sinkhorn and Hausdorff negative for Range
@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "m,n", [(1.0, 1.0), (0.7, 2.0), (0.5, 0.7), (1.5, 2.0)]
)
@pytest.mark.parametrize(
    "entropy",
    [
        Balanced(1e0),
        KullbackLeibler(1e0, 1e0),
        TotalVariation(1e0, 1e0),
        Range(1e0, 0.3, 2),
        PowerEntropy(1e0, 1e0, 0),
        PowerEntropy(1e0, 1e0, -1),
    ],
)
@pytest.mark.parametrize("div", [sinkhorn_divergence, hausdorff_divergence])
def test_divergence_positivity(div, entropy, reach, p, m, n):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    func = div(m * a, x, n * b, y, cost, entropy, solver=solver)
    assert torch.ge(func, 0.0).all()


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "m,n", [(1.0, 1.0), (0.7, 2.0), (0.5, 0.7), (1.5, 2.0)]
)
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e4, 1e0),
        TotalVariation(1e4, 1e0),
        Range(1e4, 0.3, 2),
        PowerEntropy(1e4, 1e0, 0),
        PowerEntropy(1e4, 1e0, -1),
    ],
)
def test_consistency_infinite_blur_regularized_ot_unbalanced(
    entropy, reach, p, m, n
):
    entropy.reach = reach
    cost = euclidean_cost(p)
    torch.set_default_dtype(torch.float64)
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    phi = entropy.entropy
    f, g = convolution(a, x, b, y, cost)
    control = (
        scal(a, f) + m * phi(torch.Tensor([n])) + n * phi(torch.Tensor([m]))
    )
    func = regularized_ot(m * a, x, n * b, y, cost, entropy, solver=solver)
    assert torch.allclose(func, control, rtol=1e-0)


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("entropy", [Balanced(1e5)])
def test_consistency_infinite_blur_regularized_ot_balanced(entropy, p):
    """Control consistency in OT_eps when eps goes to infinity,
    especially for balanced OT"""
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    f, g = convolution(a, x, b, y, cost)
    control = scal(a, f)
    func = regularized_ot(a, x, b, y, cost, entropy, solver)
    assert torch.allclose(func, control, rtol=1e-0)


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e6, 1e0),
        Balanced(1e6),
        TotalVariation(1e6, 1e0),
        Range(1e6, 0.3, 2),
        PowerEntropy(1e6, 1e0, 0),
        PowerEntropy(1e6, 1e0, -1),
    ],
)
def test_consistency_infinite_blur_sinkhorn_div(entropy, reach, p):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    control = energyDistance(a, x, b, y, p)
    func = sinkhorn_divergence(a, x, b, y, cost, entropy, solver)
    assert torch.allclose(func, control, rtol=1e-0)
