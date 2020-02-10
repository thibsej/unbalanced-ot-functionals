import pytest

import torch
from common.functional import regularized_ot, hausdorff_divergence, sinkhorn_divergence, energyDistance
from common.sinkhorn import BatchVanillaSinkhorn
from common.entropy import KullbackLeibler, Balanced, TotalVariation, Range, PowerEntropy
from common.utils import generate_measure

torch.set_default_tensor_type(torch.cuda.FloatTensor)
solver = BatchVanillaSinkhorn(nits=10, tol=0, assume_convergence=True)

@pytest.mark.parametrize('entropy', [KullbackLeibler(1e0, 1e0), Balanced(1e0), TotalVariation(1e0, 1e0),
                                     Range(1e0, 0.3, 2), PowerEntropy(1e0, 1e0, 0), PowerEntropy(1e0, 1e0, -1)])
def test_divergence_zero(entropy):
    a, x = generate_measure(1, 5, 2)
    a, x = a.float().cuda(), x.float().cuda()
    b, y = generate_measure(1, 6, 2)
    b, y = b.float().cuda(), y.float().cuda()
    cost = sinkhorn_divergence(a, x, b, y, p=2, entropy=entropy, solver=solver)