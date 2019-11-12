import pytest

import torch
from common.sinkhorn import sinkhorn_asym, sinkhorn_sym
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure


@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
def test_sinkhorn_dimensional_consistency(entropy):
    """Preliminary test to ensure no bug is raised."""
    a, x = generate_measure(3, 10, 2)
    b, y = generate_measure(3, 15, 2)
    sinkhorn_asym(a, x, y, b, p=2, entropy=entropy)
    sinkhorn_sym(a, x, p=2, entropy=entropy) # Symmetric Sinkhorn without extrapolation
    sinkhorn_sym(a, x, p=2, entropy=entropy, y_j=y) # with extrapolation


@pytest.mark.parametrize('entropy', [KullbackLeibler(1e5, 1e0), Balanced(1e5), TotalVariation(1e5, 1e0),
                                     Range(1e5, 0.3, 2), PowerEntropy(1e5, 1e0, 0), PowerEntropy(1e5, 1e0, -1)])
def test_sinkhorn_asym_infinite_blur(entropy):
    p=2
    a, x = generate_measure(3, 10, 2)
    b, y = generate_measure(3, 15, 2)
    f_c, g_c = entropy.init_potential()(a, x, b, y, p)
    f, g = sinkhorn_asym(a, x, y, b, p=2, entropy=entropy, nits=10000, tol=1e-7)
    assert torch.allclose(f, f_c, 1e-6)
    assert torch.allclose(g, g_c, 1e-6)


@pytest.mark.parametrize('entropy', [KullbackLeibler(1e5, 1e0), Balanced(1e5), TotalVariation(1e5, 1e0),
                                     Range(1e5, 0.3, 2), PowerEntropy(1e5, 1e0, 0), PowerEntropy(1e5, 1e0, -1)])
def test_sinkhorn_sym_infinite_blur(entropy):
    p=2
    a, x = generate_measure(3, 10, 2)
    f_c, _ = entropy.init_potential()(a, x, a, x, p)
    _, f = sinkhorn_sym(a, x, p=p, entropy=entropy, nits=10000, tol=1e-7)
    assert torch.allclose(f, f_c, 1e-6)


@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0,1e0, -1)])
def test_sinkhorn_sym_infinite_blur(entropy):
    p=2
    a, x = generate_measure(3, 10, 2)
    f_a, g_a = sinkhorn_asym(a, x, a, x, p=p, entropy=entropy)
    f_s = sinkhorn_sym(a, x, p=p, entropy=entropy, nits=10000, tol=1e-7)
    assert torch.allclose(f_a, f_s, 1e-6)
    assert torch.allclose(g_a, f_s, 1e-6)

