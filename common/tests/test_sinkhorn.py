import pytest

import torch

from common.sinkhorn import BatchVanillaSinkhorn, BatchScalingSinkhorn, BatchExpSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure

torch.set_default_tensor_type(torch.DoubleTensor)

# TODO: Check correctness of exp-sinkhorn for small blur
# TODO: Check correctness for at least two batches, not only one


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m,n', [(1., 1.), (0.7, 2.), (0.5, 0.7), (1.5, 2.)])
@pytest.mark.parametrize('entropy,rtol', [(KullbackLeibler(1e6, 1e0), 1e-4), (Range(1e6, 0.3, 2), 1e-4),
                                          (TotalVariation(1e6, 1e0), 1e-0), (PowerEntropy(1e6, 1e0, 0), 1e-4),
                                          (PowerEntropy(1e6, 1e0, -1), 1e-4)])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True),
                                  BatchScalingSinkhorn(budget=1000, nits_grad=10, assume_convergence=True),
                                  BatchExpSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True)])
def test_sinkhorn_asym_infinite_blur_unbalanced(solv, entropy, rtol, p, m, n, reach):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    f_c, g_c = entropy.init_potential(m * a, x, n * b, y, p)
    f, g = solv.sinkhorn_asym(m * a, x, n * b, y, p=p, entropy=entropy)
    assert torch.allclose(entropy.error_sink(f, f_c), torch.tensor([0]), rtol=rtol)
    assert torch.allclose(entropy.error_sink(g, g_c), torch.tensor([0]), rtol=rtol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m', [0.7, 1., 2.])
@pytest.mark.parametrize('entropy,rtol', [(Balanced(1e6), 1e-0)])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True),
                                  BatchScalingSinkhorn(budget=1000, nits_grad=10, assume_convergence=True),
                                  BatchExpSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True)])
def test_sinkhorn_asym_infinite_blur_balanced(solv, entropy, rtol, p, m, reach):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    f_c, g_c = entropy.init_potential(m * a, x, m * b, y, p)
    f, g = solv.sinkhorn_asym(m * a, x, m * b, y, p=p, entropy=entropy)
    assert torch.allclose(entropy.error_sink(f, f_c), torch.tensor([0]), rtol=rtol)
    assert torch.allclose(entropy.error_sink(g, g_c), torch.tensor([0]), rtol=rtol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m', [0.7, 1., 2.])
@pytest.mark.parametrize('entropy,rtol', [(KullbackLeibler(1e6, 1e0), 1e-4), (Range(1e6, 0.3, 2), 1e-4),
                                          (Balanced(1e6), 1e-0), (TotalVariation(1e6, 1e0), 1e-0),
                                          (PowerEntropy(1e6, 1e0, 0), 1e-4), (PowerEntropy(1e6, 1e0, -1), 1e-4)])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True),
                                  BatchScalingSinkhorn(budget=1000, nits_grad=10, assume_convergence=True),
                                  BatchExpSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True)])
def test_sinkhorn_sym_infinite_blur(solv, entropy, rtol, p, m, reach):
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    f_c, _ = entropy.init_potential(m * a, x, m * a, x, p)
    _, f = solv.sinkhorn_sym(m * a, x, p=p, entropy=entropy)
    assert torch.allclose(entropy.error_sink(f, f_c), torch.tensor([0]), rtol=rtol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m', [0.7, 1., 2.])
@pytest.mark.parametrize('entropy,rtol', [(KullbackLeibler(1e0, 1e0), 1e-6), (Balanced(1e0), 1e-6),
                                          (TotalVariation(1e0, 1e0), 1e-4), (Range(1e0, 0.3, 2), 1e-6),
                                          (PowerEntropy(1e0, 1e0, 0), 1e-6), (PowerEntropy(1e0, 1e0, -1), 1e-6)])
@pytest.mark.parametrize('solv', [BatchVanillaSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True),
                                  BatchScalingSinkhorn(budget=1000, nits_grad=10, assume_convergence=True),
                                  BatchExpSinkhorn(nits=10000, nits_grad=10, tol=1e-14, assume_convergence=True)])
def test_sinkhorn_consistency_sym_asym(solv, entropy, rtol, p, m, reach):
    """Test if the symmetric and assymetric Sinkhorn output the same results when (a,x)=(b,y)"""
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    f_a, g_a = solv.sinkhorn_asym(m * a, x, m * a, x, p=p, entropy=entropy)
    _, f_s = solv.sinkhorn_sym(m * a, x, p=p, entropy=entropy)
    assert torch.allclose(entropy.error_sink(f_a, f_s), torch.tensor([0]), rtol=rtol)
    assert torch.allclose(entropy.error_sink(g_a, f_s), torch.tensor([0]), rtol=rtol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m', [0.7, 1., 2.])
@pytest.mark.parametrize('entropy,rtol', [(Balanced(1e1), 1e-6), (KullbackLeibler(1e1, 1e0), 1e-6),
                                          (TotalVariation(1e1, 1e0), 1e-4), (Range(1e1, 0.3, 2), 1e-6),
                                          (PowerEntropy(1e0, 1e0, 0), 1e-6), (PowerEntropy(1e0, 1e0, -1), 1e-6)])
def test_sinkhorn_consistency_exp_log_asym(entropy, rtol, p, m, reach):
    """Test if the exp sinkhorn is consistent with its log form"""
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    solver1 = BatchVanillaSinkhorn(nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True)
    solver2 = BatchExpSinkhorn(nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True)
    f_a, g_a = solver1.sinkhorn_asym(m * a, x, m * b, y, p=p, entropy=entropy)
    u_a, v_a = solver2.sinkhorn_asym(m * a, x, m * b, y, p=p, entropy=entropy)
    assert torch.allclose(f_a, u_a, rtol=rtol)
    assert torch.allclose(g_a, v_a, rtol=rtol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('reach', [0.5, 1., 2.])
@pytest.mark.parametrize('m', [0.7, 1., 2.])
@pytest.mark.parametrize('entropy,atol', [(Balanced(1e1), 1e-6), (KullbackLeibler(1e1, 1e0), 1e-6),
                                          (TotalVariation(1e1, 1e0), 1e-4), (Range(1e1, 0.3, 2), 1e-6),
                                          (PowerEntropy(1e0, 1e0, 0), 1e-6), (PowerEntropy(1e0, 1e0, -1), 1e-6)])
def test_sinkhorn_consistency_exp_log_sym(entropy, atol, p, m, reach):
    """Test if the exp sinkhorn is consistent with its log form"""
    entropy.reach = reach
    a, x = generate_measure(1, 5, 2)
    solver1 = BatchVanillaSinkhorn(nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True)
    solver2 = BatchExpSinkhorn(nits=10000, nits_grad=10, tol=1e-12, assume_convergence=True)
    _, g_a = solver1.sinkhorn_sym(m * a, x, p=p, entropy=entropy)
    _, v_a = solver2.sinkhorn_sym(m * a, x, p=p, entropy=entropy)
    assert torch.allclose(g_a, v_a, rtol=atol)
