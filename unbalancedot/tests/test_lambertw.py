import torch
import numpy as np
import scipy.special.lambertw as lambertw
from unbalancedot.torch_lambertw import log_lambertw


def test_log_lambertw():
    input = np.array(
        [-10000, -500.0, -100.0, -5.0, -2.0, 0.0, 1.0, 5.0, 500.0]
    )
    control = np.real(lambertw(np.exp(input)))
    output = log_lambertw(torch.from_numpy(input))
    assert torch.allclose(output, torch.from_numpy(control), atol=1e-7)
