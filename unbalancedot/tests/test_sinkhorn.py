import pytest

import torch

from unbalancedot.sinkhorn import (
    BatchVanillaSinkhorn,
    BatchScalingSinkhorn,
    BatchExpSinkhorn,
)
from unbalancedot.entropy import (
    KullbackLeibler,
    Balanced,
    TotalVariation,
    Range,
    PowerEntropy,
)
from unbalancedot.utils import generate_measure, euclidean_cost

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)


@pytest.mark.parametrize(
    "entropy",
    [
        Balanced(1e1),
        KullbackLeibler(1e1, 1e0),
        TotalVariation(1e1, 1e0),
        Range(1e1, 0.3, 2),
        PowerEntropy(1e1, 1e0, 0),
        PowerEntropy(1e1, 1e0, -1),
    ],
)
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=10, nits_grad=10, tol=1e-5, assume_convergence=True
        ),
        BatchVanillaSinkhorn(
            nits=10, nits_grad=10, tol=1e-5, assume_convergence=False
        ),
        BatchScalingSinkhorn(budget=10, nits_grad=10, assume_convergence=True),
        BatchScalingSinkhorn(
            budget=10, nits_grad=10, assume_convergence=False
        ),
        BatchExpSinkhorn(
            nits=10, nits_grad=10, tol=1e-5, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=10, nits_grad=10, tol=1e-5, assume_convergence=False
        ),
    ],
)
def test_sinkhorn_no_bug(entropy, solv):
    a, x = generate_measure(2, 5, 3)
    b, y = generate_measure(2, 6, 3)
    solv.sinkhorn_asym(a, x, b, y, cost=euclidean_cost(1), entropy=entropy)
    solv.sinkhorn_sym(a, x, cost=euclidean_cost(1), entropy=entropy, y_j=y)


# TODO: Adapt the error function for TV due to translation
#  invariance when masses are both 1
@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "m,n", [(1.0, 1.0), (0.7, 2.0), (0.5, 0.7), (1.5, 2.0)]
)
@pytest.mark.parametrize(
    "entropy,atol",
    [
        (KullbackLeibler(1e6, 1e0), 1e-4),
        (Range(1e6, 0.3, 2), 1e-4),
        (TotalVariation(1e6, 1e0), 1e-0),
        (PowerEntropy(1e6, 1e0, 0), 1e-4),
        (PowerEntropy(1e6, 1e0, -1), 1e-4),
    ],
)
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
        BatchScalingSinkhorn(
            budget=1000, nits_grad=10, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_sinkhorn_asym_infinite_blur_unbalanced(
    solv, entropy, atol, p, m, n, reach
):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(2, 5, 3)
    b, y = generate_measure(2, 6, 3)
    f_c, g_c = entropy.init_potential(m * a, x, n * b, y, p)
    f, g = solv.sinkhorn_asym(m * a, x, n * b, y, cost=cost, entropy=entropy)
    assert torch.allclose(
        torch.tensor([0.0]), entropy.error_sink(f, f_c), atol=atol
    )
    assert torch.allclose(
        torch.tensor([0.0]), entropy.error_sink(g, g_c), atol=atol
    )


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("entropy,atol", [(Balanced(1e6), 1e-0)])
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
        BatchScalingSinkhorn(
            budget=1000, nits_grad=10, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_sinkhorn_asym_infinite_blur_balanced(
    solv, entropy, atol, p, m, reach
):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(2, 5, 3)
    b, y = generate_measure(2, 6, 3)
    f_c, g_c = entropy.init_potential(m * a, x, m * b, y, cost=cost)
    f, g = solv.sinkhorn_asym(m * a, x, m * b, y, cost=cost, entropy=entropy)
    assert torch.allclose(
        entropy.error_sink(f, f_c), torch.tensor([0.0]), atol=atol
    )
    assert torch.allclose(
        entropy.error_sink(g, g_c), torch.tensor([0.0]), atol=atol
    )


# TODO: Adapt the error function for TV due to translation
#  invariance when masses are both 1
@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [0.7, 1.0, 2.0])
@pytest.mark.parametrize(
    "entropy,atol",
    [
        (KullbackLeibler(1e6, 1e0), 1e-4),
        (Range(1e6, 0.3, 2), 1e-4),
        (Balanced(1e6), 1e-0),
        (TotalVariation(1e6, 1e0), 1e-0),
        (PowerEntropy(1e6, 1e0, 0), 1e-4),
        (PowerEntropy(1e6, 1e0, -1), 1e-4),
    ],
)
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
        BatchScalingSinkhorn(
            budget=1000, nits_grad=10, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_sinkhorn_sym_infinite_blur(solv, entropy, atol, p, m, reach):
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(2, 5, 3)
    f_c, _ = entropy.init_potential(m * a, x, m * a, x, cost=cost)
    _, f = solv.sinkhorn_sym(m * a, x, cost=cost, entropy=entropy)
    assert torch.allclose(
        entropy.error_sink(f, f_c), torch.tensor([0.0]), atol=atol
    )


# TODO: Adapt the error function for TV due to translation
#  invariance when masses are both 1
@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [0.7, 1.0, 2.0])
@pytest.mark.parametrize(
    "entropy,atol",
    [
        (KullbackLeibler(1e0, 1e0), 1e-6),
        (Balanced(1e0), 1e-6),
        (TotalVariation(1e0, 1e0), 1e-4),
        (Range(1e0, 0.3, 2), 1e-6),
        (PowerEntropy(1e0, 1e0, 0), 1e-6),
        (PowerEntropy(1e0, 1e0, -1), 1e-6),
    ],
)
@pytest.mark.parametrize(
    "solv",
    [
        BatchVanillaSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
        BatchScalingSinkhorn(
            budget=1000, nits_grad=10, assume_convergence=True
        ),
        BatchExpSinkhorn(
            nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True
        ),
    ],
)
def test_sinkhorn_consistency_sym_asym(solv, entropy, atol, p, m, reach):
    """Test if the symmetric and assymetric Sinkhorn
    output the same results when (a,x)=(b,y)"""
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(2, 5, 3)
    f_a, g_a = solv.sinkhorn_asym(
        m * a, x, m * a, x, cost=cost, entropy=entropy
    )
    _, f_s = solv.sinkhorn_sym(m * a, x, cost=cost, entropy=entropy)
    assert torch.allclose(
        entropy.error_sink(f_a, f_s), torch.tensor([0.0]), atol=atol
    )
    assert torch.allclose(
        entropy.error_sink(g_a, f_s), torch.tensor([0.0]), atol=atol
    )


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [0.7, 1.0, 2.0])
@pytest.mark.parametrize(
    "entropy,rtol",
    [
        (Balanced(1e1), 1e-6),
        (KullbackLeibler(1e1, 1e0), 1e-6),
        (TotalVariation(1e1, 1e0), 1e-4),
        (Range(1e1, 0.3, 2), 1e-6),
        (PowerEntropy(1e0, 1e0, 0), 1e-6),
        (PowerEntropy(1e0, 1e0, -1), 1e-6),
    ],
)
def test_sinkhorn_consistency_exp_log_asym(entropy, rtol, p, m, reach):
    """Test if the exp sinkhorn is consistent with its log form"""
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(2, 5, 3)
    b, y = generate_measure(2, 6, 3)
    solver1 = BatchVanillaSinkhorn(
        nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True
    )
    solver2 = BatchExpSinkhorn(
        nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True
    )
    f_a, g_a = solver1.sinkhorn_asym(
        m * a, x, m * b, y, cost=cost, entropy=entropy
    )
    u_a, v_a = solver2.sinkhorn_asym(
        m * a, x, m * b, y, cost=cost, entropy=entropy
    )
    assert torch.allclose(f_a, u_a, rtol=rtol)
    assert torch.allclose(g_a, v_a, rtol=rtol)


@pytest.mark.parametrize("p", [1, 1.5, 2])
@pytest.mark.parametrize("reach", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m", [0.7, 1.0, 2.0])
@pytest.mark.parametrize(
    "entropy,rtol",
    [
        (Balanced(1e1), 1e-6),
        (KullbackLeibler(1e1, 1e0), 1e-6),
        (TotalVariation(1e1, 1e0), 1e-4),
        (Range(1e1, 0.3, 2), 1e-6),
        (PowerEntropy(1e0, 1e0, 0), 1e-6),
        (PowerEntropy(1e0, 1e0, -1), 1e-6),
    ],
)
def test_sinkhorn_consistency_exp_log_sym(entropy, rtol, p, m, reach):
    """Test if the exp sinkhorn is consistent with its log form"""
    entropy.reach = reach
    cost = euclidean_cost(p)
    a, x = generate_measure(2, 5, 3)
    solver1 = BatchVanillaSinkhorn(
        nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True
    )
    solver2 = BatchExpSinkhorn(
        nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True
    )
    _, g_a = solver1.sinkhorn_sym(m * a, x, cost=cost, entropy=entropy)
    _, v_a = solver2.sinkhorn_sym(m * a, x, cost=cost, entropy=entropy)
    assert torch.allclose(g_a, v_a, rtol=rtol)


@pytest.mark.parametrize(
    "entropy",
    [
        Balanced(1e-5),
        KullbackLeibler(1e-5, 1e0),
        TotalVariation(1e-5, 1e0),
        Range(1e-5, 0.3, 2),
        PowerEntropy(1e-5, 1e0, 0),
        PowerEntropy(1e-5, 1e0, -1),
    ],
)
def test_sanity_control_exp_sinkhorn_small(entropy):
    a, x = generate_measure(2, 5, 3)
    b, y = generate_measure(2, 6, 3)
    solver = BatchExpSinkhorn(
        nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True
    )
    f, g = solver.sinkhorn_asym(
        a, x, b, y, cost=euclidean_cost(1), entropy=entropy
    )
    _, h = solver.sinkhorn_sym(a, x, cost=euclidean_cost(1), entropy=entropy)
    assert f is None
    assert g is None
    assert h is None
