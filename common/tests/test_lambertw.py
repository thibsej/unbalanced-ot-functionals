import pytest

import torch
import numpy as np
import scipy.special.lambertw as lambertw
from common.torch_lambertw import log_lambertw

def test_log_lambertw():
    input = np.array([-500., -100., -5., -2., 0., 1., 5., 500.])
    control = np.real(lambertw(np.exp(input)))
    output = log_lambertw(torch.from_numpy(input))
    assert torch.allclose(output, torch.from_numpy(control), atol=1e-7)