import torch

def log_lambertw(x):
    """
    Computes the Lambert function via Halley algorithm which converges cubically.
    The initialization is performed with a local approximation.
    """
    z = init_lambertw(x)
    a = lambda w: (w * (w.log() + w - x)) / (1 + w)
    b = lambda w: -1 / (w * (1 + w))
    for i in range(4):
        c = a(z)
        z = z - c / (1 - 0.5 * c * b(z))
    return z

def init_lambertw(x):
    z0 = torch.zeros_like(x)
    z0[x > 1.] = x[x > 1.]
    z0[x < -2.] = x[x < -2.].exp()*(1. - x[x < -2.].exp())
    pade = lambda x: x * (3. + 6. * x + x ** 2.) / (3. + 9. * x + 5. * x ** 2)
    z0[(x <= 1.) & (x >= -2.)] = pade(x[(x <= 1.) & (x >= -2.)].exp())
    z0[z0 == 0.] = 1e-6
    return z0
