import pytest

import torch
from common.sinkhorn import sinkhorn_asym, sinkhorn_sym
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure

@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
def test_sinkhorn_dimensional_consistency(entropy, p):
    """Preliminary test to ensure no bug is raised."""
    a, x = generate_measure(3, 5, 2)
    b, y = generate_measure(3, 6, 2)
    sinkhorn_asym(a, x, b, y, p=p, entropy=entropy)
    sinkhorn_sym(a, x, p=p, entropy=entropy)  # Symmetric Sinkhorn without extrapolation
    sinkhorn_sym(a, x, p=p, entropy=entropy, y_j=y)  # with extrapolation


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('entropy,atol', [(KullbackLeibler(1e6, 1e0), 1e-4), (Range(1e6, 0.3, 2), 1e-4),
                                          (PowerEntropy(1e6, 1e0, 0), 1e-4), (Balanced(1e6), 1e-0),
                                          (TotalVariation(1e6, 1e0), 1e-0), (PowerEntropy(1e6, 1e0, -1), 1e-4)])
def test_sinkhorn_asym_infinite_blur(entropy, atol, p):
    a, x = generate_measure(1, 5, 2)
    b, y = generate_measure(1, 6, 2)
    err = entropy.error_sink()
    f_c, g_c = entropy.init_potential()(a, x, b, y, p)
    f, g = sinkhorn_asym(a, x, b, y, p=p, entropy=entropy, nits=10000, tol=0)
    assert torch.allclose(err(f, f_c), torch.Tensor([0.0]), atol=atol)
    assert torch.allclose(err(g, g_c), torch.Tensor([0.0]), atol=atol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('entropy,atol', [(KullbackLeibler(1e6, 1e0), 1e-4), (Range(1e6, 0.3, 2), 1e-4),
                                          (PowerEntropy(1e6, 1e0, 0), 1e-4), (Balanced(1e6), 1e-0),
                                          (TotalVariation(1e6, 1e0), 1e-0), (PowerEntropy(1e6, 1e0, -1), 1e-4)])
def test_sinkhorn_sym_infinite_blur(entropy, atol, p):
    a, x = generate_measure(1, 5, 2)
    f_c, _ = entropy.init_potential()(a, x, a, x, p)
    _, f = sinkhorn_sym(a, x, p=p, entropy=entropy, nits=10000, tol=0)
    assert torch.allclose(f, f_c, atol=atol)


@pytest.mark.parametrize('p', [1, 1.5, 2])
@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
def test_sinkhorn_consistency_sym_asym(entropy, p):
    """Test if the symmetric and assymetric Sinkhorn output the same results when (a,x)=(b,y)"""
    a, x = generate_measure(1, 5, 2)
    err = entropy.error_sink()
    f_a, g_a = sinkhorn_asym(a, x, a, x, p=p, entropy=entropy, tol=0)
    _, f_s = sinkhorn_sym(a, x, p=p, entropy=entropy, nits=10000, tol=0)
    assert torch.allclose(err(f_a, f_s), torch.Tensor([0.0]), atol=1e-6)
    assert torch.allclose(err(g_a, f_s), torch.Tensor([0.0]), atol=1e-6)
