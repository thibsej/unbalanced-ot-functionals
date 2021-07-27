"""
All the code on functionals derived fron the theory of entropically
regularized unbalanced optimal transport.
"""

__version__ = 0.1

from .entropy import Balanced, KullbackLeibler, TotalVariation, Range, \
    PowerEntropy # noqa
from .sinkhorn import BatchVanillaSinkhorn, BatchScalingSinkhorn # noqa
from .utils import dist_matrix, convolution, softmin, generate_measure # noqa
from .functional import regularized_ot, hausdorff_divergence, \
    sinkhorn_divergence, energyDistance # noqa
