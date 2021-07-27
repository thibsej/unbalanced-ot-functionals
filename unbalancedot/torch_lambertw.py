import torch


def log_lambertw(x):
    """
    Computes the Lambert function via Halley algorithm which converges
    cubically.
    The initialization is performed with a local approximation.
    """
    z = init_lambertw(x)
    eps = torch.finfo(x.dtype).eps

    def a(w):
        return (w * ((w + eps).log() + w - x)) / (1 + w)

    def b(w):
        return -1 / (w * (1 + w))

    for i in range(4):
        c = a(z)
        z = torch.max(
            z - c / (1 - 0.5 * c * b(z)),
            torch.tensor([eps], dtype=x.dtype)[:, None],
        )
    return z


def init_lambertw(x):
    z0 = torch.zeros_like(x)
    z0[x > 1.0] = x[x > 1.0]
    z0[x < -2.0] = x[x < -2.0].exp() * (1.0 - x[x < -2.0].exp())

    def pade(x):
        return x * (3.0 + 6.0 * x + x ** 2.0) / (3.0 + 9.0 * x + 5.0 * x ** 2)

    z0[(x <= 1.0) & (x >= -2.0)] = pade(x[(x <= 1.0) & (x >= -2.0)].exp())
    z0[z0 == 0.0] = 1e-6
    return z0
