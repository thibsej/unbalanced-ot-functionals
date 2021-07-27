import pytest

import torch

from unbalancedot.functional import regularized_ot, sinkhorn_divergence
from unbalancedot.sinkhorn import BatchVanillaSinkhorn, BatchExpSinkhorn
from unbalancedot.entropy import (
    KullbackLeibler,
    Balanced,
    TotalVariation,
    Range,
    PowerEntropy,
)
from unbalancedot.utils import generate_measure, dist_matrix, euclidean_cost

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)
torch.set_printoptions(precision=10)


@pytest.mark.parametrize("p", [2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "m,n", [(1.0, 1.0), (0.7, 2.0), (0.5, 0.7), (1.5, 2.0)]
)
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e0, 1e0),
        TotalVariation(1e0, 1e0),
        Range(1e0, 0.3, 2),
        PowerEntropy(1e0, 1e0, 0),
        PowerEntropy(1e0, 1e0, -1),
    ],
)
@pytest.mark.parametrize("div", [regularized_ot])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=5000, nits_grad=20, tol=1e-14, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=5000, nits_grad=20, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_gradient_unbalanced_weight_and_position_asym(
    solv, div, entropy, reach, p, m, n
):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    x.requires_grad = True
    b, y = generate_measure(1, 6, 2)
    f, g = solv.sinkhorn_asym(a, x, n * b, y, cost, entropy)
    func = entropy.output_regularized(a, x, n * b, y, cost, f, g)
    [grad_num_x, grad_num_a] = torch.autograd.grad(func, [x, a])
    grad_th_a = (
        -entropy.legendre_entropy(-f)
        + entropy.blur * n
        - entropy.blur
        * (
            (f[:, :, None] + g[:, None, :] - dist_matrix(x, y, p)).exp()
            * n
            * b[:, None, :]
        ).sum(dim=2)
    )
    pi = (
        n
        * a[:, :, None]
        * b[:, None, :]
        * (
            (f[:, :, None] + g[:, None, :] - dist_matrix(x, y, p))
            / entropy.blur
        ).exp()
    )
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum(
        "ijk, ikl->ijl", pi, y
    )
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)


@pytest.mark.parametrize("p", [2])
@pytest.mark.parametrize("m", [1.0, 0.7, 1.5])
@pytest.mark.parametrize("entropy", [Balanced(1e0)])
@pytest.mark.parametrize("div", [regularized_ot])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=5000, nits_grad=1, tol=1e-14, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=5000, nits_grad=1, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_gradient_balanced_weight_and_position_asym(solv, div, entropy, p, m):
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    a, b = m * a, m * b
    a.requires_grad = True
    x.requires_grad = True
    f, g = solv.sinkhorn_asym(a, x, b, y, cost, entropy)
    func = entropy.output_regularized(a, x, b, y, cost, f, g)
    [grad_num_x, grad_num_a] = torch.autograd.grad(func, [x, a])
    grad_th_a = f
    pi = (
        a[:, :, None]
        * b[:, None, :]
        * (
            (f[:, :, None] + g[:, None, :] - dist_matrix(x, y, p))
            / entropy.blur
        ).exp()
    )
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum(
        "ijk, ikl->ijl", pi, y
    )
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)


# TV and range are not tested because they do not have symmetric potentials
@pytest.mark.parametrize("p", [2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [1.0, 0.7, 1.5])
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e0, 1e0),
        PowerEntropy(1e0, 1e0, 0),
        PowerEntropy(1e0, 1e0, -1),
    ],
)
@pytest.mark.parametrize("div", [regularized_ot])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=5000, nits_grad=200, tol=1e-18, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=5000, nits_grad=200, tol=1e-18, assume_convergence=True
        ),
    ],
)
def test_gradient_unbalanced_weight_and_position_sym(
    solv, div, entropy, reach, p, m
):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    x.requires_grad = True
    _, f = solv.sinkhorn_asym(a, x, a, x, cost, entropy)
    func = entropy.output_regularized(a, x, a, x, cost, f, f)
    [grad_num_x, grad_num_a] = torch.autograd.grad(func, [x, a])
    grad_th_a = (
        -2 * entropy.legendre_entropy(-f)
        + 2 * entropy.blur * m
        - 2
        * entropy.blur
        * (
            (f[:, :, None] + f[:, None, :] - dist_matrix(x, x, p)).exp()
            * a[:, None, :]
        ).sum(dim=2)
    )
    pi = (
        a[:, :, None]
        * a[:, None, :]
        * (
            (f[:, :, None] + f[:, None, :] - dist_matrix(x, x, p))
            / entropy.blur
        ).exp()
    )
    grad_th_x = 4 * x * pi.sum(dim=2)[:, :, None] - 4 * torch.einsum(
        "ijk, ikl->ijl", pi, x
    )
    print(f"Symmetric potential = {f}")
    print(f"gradient ratio = {grad_num_a / grad_th_a}")
    print(f"gradient ratio = {grad_num_x / grad_th_x}")
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)


@pytest.mark.parametrize("p", [2])
@pytest.mark.parametrize("m", [1.0, 0.7, 1.5])
@pytest.mark.parametrize("entropy", [Balanced(1e0)])
@pytest.mark.parametrize("div", [regularized_ot])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=5000, nits_grad=20, tol=1e-14, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=5000, nits_grad=20, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_gradient_balanced_weight_and_position_sym(solv, div, entropy, p, m):
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    a = m * a
    a.requires_grad = True
    x.requires_grad = True
    _, f = solv.sinkhorn_sym(a, x, cost, entropy)
    func = entropy.output_regularized(a, x, a, x, cost, f, f)
    [grad_num_x, grad_num_a] = torch.autograd.grad(func, [x, a])
    grad_th_a = 2 * f
    pi = (
        a[:, :, None]
        * a[:, None, :]
        * (
            (f[:, :, None] + f[:, None, :] - dist_matrix(x, x, p))
            / entropy.blur
        ).exp()
    )
    grad_th_x = 2 * x * pi.sum(dim=2)[:, :, None] - 2 * torch.einsum(
        "ijk, ikl->ijl", pi, x
    )
    assert torch.allclose(grad_th_a, grad_num_a, rtol=1e-5)
    assert torch.allclose(grad_th_x, grad_num_x, rtol=1e-5)


# TODO: Debug TV gradient for Sinkhorn
@pytest.mark.parametrize("p", [2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "m,n", [(1.0, 1.0), (0.7, 2.0), (0.5, 0.7), (1.5, 2.0)]
)
@pytest.mark.parametrize(
    "entropy",
    [
        KullbackLeibler(1e0, 1e0),
        TotalVariation(1e0, 1e0),
        Range(1e0, 0.3, 2),
        PowerEntropy(1e0, 1e0, 0),
        PowerEntropy(1e0, 1e0, -1),
    ],
)
@pytest.mark.parametrize("div", [regularized_ot])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=5000, nits_grad=20, tol=1e-14, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=5000, nits_grad=20, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_gradient_unbalanced_zero_grad_sinkhorn(
    solv, div, entropy, reach, p, m, n
):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    a = m * a
    b, y = torch.zeros_like(a), torch.zeros_like(x)
    b.copy_(a)
    y.copy_(x)
    a.requires_grad = True
    x.requires_grad = True
    func = sinkhorn_divergence(a, x, b, y, cost, entropy, solver=solv)
    [grad_num_x, grad_num_a] = torch.autograd.grad(func, [x, a])
    assert torch.allclose(torch.zeros_like(grad_num_a), grad_num_a, atol=1e-5)
    assert torch.allclose(torch.zeros_like(grad_num_x), grad_num_x, atol=1e-5)


# TODO: Debug gradient
@pytest.mark.parametrize("p", [2])
@pytest.mark.parametrize("m", [1.0, 0.7, 1.5])
@pytest.mark.parametrize("entropy", [Balanced(1e0)])
@pytest.mark.parametrize("div", [regularized_ot])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=5000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=5000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_gradient_balanced_zero_grad_sinkhorn(solv, div, entropy, p, m):
    cost = euclidean_cost(p)
    a, x = generate_measure(1, 5, 2)
    a = m * a
    b, y = torch.zeros_like(a), torch.zeros_like(x)
    b.copy_(a)
    y.copy_(x)
    a.requires_grad = True
    x.requires_grad = True
    func = sinkhorn_divergence(a, x, b, y, cost, entropy, solver=solv)
    [grad_num_x, grad_num_a] = torch.autograd.grad(func, [x, a])
    assert torch.allclose(torch.zeros_like(grad_num_a), grad_num_a, atol=1e-5)
    assert torch.allclose(torch.zeros_like(grad_num_x), grad_num_x, atol=1e-5)
