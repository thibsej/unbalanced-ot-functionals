# import pytest
#
# import torch
# from unbalancedot.functional import sinkhorn_divergence
# from unbalancedot.sinkhorn import BatchVanillaSinkhorn
# from unbalancedot.entropy import (
#     KullbackLeibler,
#     Balanced,
#     TotalVariation,
#     Range,
#     PowerEntropy,
# )
# from unbalancedot.utils import generate_measure, euclidean_cost
#
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(0)
# solver = BatchVanillaSinkhorn(nits=10, nits_grad=2, tol=0,
#                               assume_convergence=True)
#
#
# @pytest.mark.parametrize(
#     "entropy",
#     [
#         KullbackLeibler(1e0, 1e0),
#         Balanced(1e0),
#         TotalVariation(1e0, 1e0),
#         Range(1e0, 0.3, 2),
#         PowerEntropy(1e0, 1e0, 0),
#         PowerEntropy(1e0, 1e0, -1),
#     ],
# )
# def test_divergence_zero(entropy):
#     a, x = generate_measure(1, 5, 2)
#     a, x = a.float().cuda(), x.float().cuda()
#     b, y = generate_measure(1, 6, 2)
#     b, y = b.float().cuda(), y.float().cuda()
#     sinkhorn_divergence(
#         a, x, b, y, cost=euclidean_cost(2), entropy=entropy, solver=solver
#     )
