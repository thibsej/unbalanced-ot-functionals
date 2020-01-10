import torch
from .torch_lambertw import log_lambertw
from .utils import scal, dist_matrix, convolution


class Entropy(object):
    """
    Object that defines the required modules for entropy functions.
    """

    def entropy(self):
        """Pointwise entropy used in the definition of Csiszar-divergence."""
        raise NotImplementedError

    def legendre_entropy(self):
        """Pointwise Legendre transforme of entropy used in the dual of Csiszar-divergence."""
        raise NotImplementedError

    def grad_legendre(self):
        """Gradient of the Legendre transform."""
        raise NotImplementedError

    def aprox(self):
        """
        Anisotropic Proximity operator. The function returned is $x mapsto -Aprox(-x)$.
        """
        raise NotImplementedError

    def init_potential(self):
        """
        Computes the initialization of the sinkhorn algorithm based on the asymptotic epsilon going to infinity.
        :return: two torch.Tensor (f,g)
        """
        raise NotImplementedError

    def error_sink(self, f, g):
        """
        returns the function that controls the error for Sinkhorn iteration
        (Hilbert norm for balanced OT, uniform norm otherwise).
        :return: function
        """
        return (f - g).abs().max()

    def output_regularized(self, a, x, b, y, p, f, g):
        """Outputs the cost of the regularized OT"""
        phis, partial_phis = self.legendre_entropy, self.grad_legendre
        output_pot = lambda x: - phis(-x) - 0.5 * self.blur * partial_phis(-x)
        return scal(a, output_pot(f)) + scal(b, output_pot(g)) + self.blur * a.sum(1)[:, None] * b.sum(1)[:, None]

    def output_sinkhorn(self, a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
        """Outputs the cost of the Sinkhorn divergence"""
        phis, partial_phis = self.legendre_entropy, self.grad_legendre
        output_pot = lambda x: - phis(-x) - 0.5 * self.blur * partial_phis(-x)
        return scal(a, output_pot(f_xy) - output_pot(f_xx)) + scal(b, output_pot(g_xy) - output_pot(g_yy))

    def output_hausdorff(self, a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
        """Outputs the cost of the Hausdorff divergence"""
        phis, partial_phis = self.legendre_entropy, self.grad_legendre
        output_pot = lambda x: phis(-x) + self.blur * partial_phis(-x)
        return scal(a, output_pot(f_xx) - output_pot(f_xy)) + scal(b, output_pot(g_yy) - output_pot(g_xy))


class KullbackLeibler(Entropy):

    def __init__(self, blur, reach):
        super(KullbackLeibler, self).__init__()

        self.blur = blur
        self.reach = reach
        self.__name__ = 'KullbackLeibler'

    def entropy(self, x):
        return self.reach * (x * x.log() - x + 1)

    def legendre_entropy(self, x):
        return self.reach * ( (x / self.reach).exp() - 1 )

    def grad_legendre(self, x):
        return (x / self.reach).exp()

    def aprox(self, x):
        z = self.blur / self.reach
        return (1 / (1 + z)) * x

    def init_potential(self, a, x, b, y, p):
        f = - self.reach * b.sum(dim=1).log()[:,None]
        g = - self.reach * a.sum(dim=1).log()[:,None]
        return f, g


class Balanced(Entropy):

    def __init__(self, blur):
        super(Balanced, self).__init__()

        self.blur = blur
        self.__name__ = 'Balanced'

    def entropy(self, x):
        if x == 1:
            return 0
        else:
            return float('inf')

    def legendre_entropy(self, x):
        return x

    def grad_legendre(self, x):
        return 1

    def aprox(self, x):
        return x

    def init_potential(self, a, x, b, y, p):
        f, g = convolution(a, x, b, y, p)
        scal_prod = scal(b, g)
        f = f - 0.5 * scal_prod[:, None]
        g = g - 0.5 * scal_prod[:, None]
        return f, g

    def error_sink(self, f, g):
        return (torch.max((f - g), dim=1)[0] - torch.min((f - g), dim=1)[0]).max()

    def output_regularized(self, a, x, b, y, p, f, g):
        return scal(a, f) + scal(b, g)

    def output_sinkhorn(self, a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
        return scal(a, f_xy - f_xx) + scal(b, g_xy - g_yy)

    def output_hausdorff(self, a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
        return scal(a, f_xy - f_xx) + scal(b, g_xy - g_yy)


class Range(Entropy):

    def __init__(self, blur, reach_low, reach_up):
        super(Range, self).__init__()

        self.blur = blur
        self.reach_low = reach_low
        self.reach_up = reach_up
        self.__name__ = 'Range'

    def entropy(self, x):
        if (x >= self.reach_low) & (x <= self.reach_up):
            return 0
        else:
            return float('inf')

    def legendre_entropy(self, x):
        return torch.max(self.reach_low * x, self.reach_up * x)

    def grad_legendre(self, x):
        return torch.max( - self.reach_low * x.sign(), self.reach_up * x.sign() )

    def aprox(self, x):
        r0, r1 = torch.tensor([self.reach_low], dtype=x.dtype), torch.tensor([self.reach_up], dtype=x.dtype)
        return torch.min(torch.max(torch.tensor([0.0], dtype=x.dtype), x - self.blur * r1.log()), x - self.blur * r0.log())

    def init_potential(self, a, x, b, y, p):
        f, g = torch.zeros_like(a), torch.zeros_like(b)
        return f, g

    def output_regularized(self, a, x, b, y, p, f, g):
        phis = self.legendre_entropy()
        output_pot = lambda x: - phis(-x)
        cost = scal(a, output_pot(f)) + scal(b, output_pot(g))
        C = dist_matrix(x, y, p)
        expC = a[:,:,None] * b[:,None,:] * (1 - ((f[:,:,None] + g[:,None,:] - C) / self.blur).exp())
        cost = cost + torch.sum(self.blur * expC, dim=(1,2))
        print(type(f))
        print(f"Each output potential is equal to {f} // {g}")
        print(f"Each output potential is equal to {output_pot(f)} // {output_pot(g)}")
        print(f"Each term of the cost has values {scal(a, output_pot(f))} // {scal(b, output_pot(g))} // {torch.sum(self.blur * expC, dim=(1,2))}")
        return  cost

    def output_sinkhorn(self, a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
        phis = self.legendre_entropy
        output_pot = lambda x: - phis(-x)
        cost = scal(a, output_pot(f_xy) - output_pot(f_xx)) + scal(b, output_pot(g_xy) - output_pot(g_yy))
        Cxy, Cxx, Cyy = dist_matrix(x, y, p), dist_matrix(x, x, p), dist_matrix(y, y, p)
        expC = lambda a, b, f, g, C: a[:, :, None] * b[:, None, :] * (1 - ((f[:, :, None] + g[:, None, :] - C) / self.blur).exp())
        cost = cost + torch.sum(self.blur * expC(a, b, f_xy, g_xy, Cxy), dim=(1,2)) \
               - 0.5 * torch.sum(self.blur * expC(a, a, f_xx, f_xx, Cxx), dim=(1,2)) \
               - 0.5 * torch.sum(self.blur * expC(b, b, g_yy, g_yy, Cyy), dim=(1,2))
        return cost


class TotalVariation(Entropy):
    def __init__(self, blur, reach):
        super(TotalVariation, self).__init__()

        self.blur = blur
        self.reach = reach
        self.__name__ = 'TotalVariation'

    def entropy(self, x):
        return self.reach * (x - 1).abs()

    def legendre_entropy(self, x):
        return x

    def grad_legendre(self, x):
        return 1

    def aprox(self, x):
        return torch.min(torch.max(-self.reach * torch.ones_like(x), x), self.reach * torch.ones_like(x))

    def init_potential(self, a, x, b, y, p):
        aprox = self.aprox
        mask_a, mask_b = torch.eq(a.sum(1), torch.ones(a.size(0), dtype=a.dtype)), \
                         torch.eq(b.sum(1), torch.ones(b.size(0), dtype=b.dtype))
        f, g = torch.ones_like(a), torch.ones_like(b)
        if mask_a.all() or mask_b.all():
            f, g = convolution(a, x, b, y, p)
            scal_prod = scal(b, g)
            f = f - 0.5 * scal_prod[:, None]
            g = g - 0.5 * scal_prod[:, None]
            f, g = -aprox(-f), -aprox(-g)
        f[~mask_a, :] = - self.reach * (a[~mask_a, :].sum(1)).log().sign()[:,None]
        g[~mask_b, :] = - self.reach * (b[~mask_b, :].sum(1)).log().sign()[:,None]
        return f, g

    def error_sink(self, f, g):
        err1 = (torch.max((f - g), dim=1)[0] - torch.min((f - g), dim=1)[0]).max()
        err2 = (f-g).abs().max()
        return torch.min(err1, err2)

    def output_regularized(self, a, x, b, y, p, f, g):
        phis = self.legendre_entropy
        output_pot = lambda x: - phis(-x)
        cost = scal(a, output_pot(f)) + scal(b, output_pot(g))
        C = dist_matrix(x, y, p)
        expC = a[:,:,None] * b[:,None,:] * (1 - ((f[:,:,None] + g[:,None,:] - C) / self.blur).exp())
        cost = cost + torch.sum(self.blur * expC, dim=(1,2))
        print(type(f))
        print(f"Each output potential is equal to {f} // {g}")
        print(f"Each output potential is equal to {output_pot(f)} // {output_pot(g)}")
        print(
            f"Each term of the cost has values {scal(a, output_pot(f))} // {scal(b, output_pot(g))} // {torch.sum(self.blur * expC, dim=(1,2))}")
        return cost

    def output_sinkhorn(self, a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
        phis = self.legendre_entropy
        output_pot = lambda x: - phis(-x)
        cost = scal(a, output_pot(f_xy) - output_pot(f_xx)) + scal(b, output_pot(g_xy) - output_pot(g_yy))
        Cxy, Cxx, Cyy = dist_matrix(x, y, p), dist_matrix(x, x, p), dist_matrix(y, y, p)
        expC = lambda a, b, f, g, C: a[:, :, None] * b[:, None, :] * (1 - ((f[:, :, None] + g[:, None, :] - C) / self.blur).exp())
        cost = cost + torch.sum(self.blur * expC(a, b, f_xy, g_xy, Cxy), dim=(1,2)) \
               - 0.5 * torch.sum(self.blur * expC(a, a, f_xx, f_xx, Cxx), dim=(1,2)) \
               - 0.5 * torch.sum(self.blur * expC(b, b, g_yy, g_yy, Cyy), dim=(1,2))
        return cost


class OtherPowerEntropy(Entropy):
    def __init__(self, blur, reach, power):
        super(OtherPowerEntropy, self).__init__()
        assert power < 1, "The entropy exponent is not admissible (should be <1)."

        self.blur = blur
        self.reach = reach
        self.power = power
        self.__name__ = 'PowerEntropy'

    def entropy(self, x):
        s = self.power / ( self.power - 1 )
        return (self.reach / (s * (s - 1))) * (x**s - s*(x-1) - 1)

    def legendre_entropy(self, x):
        return self.reach * (1 - 1 / self.power) * ((1 + x / (self.reach * (self.power - 1))) ** self.power - 1)


class BergEntropy(Entropy):
    def __init__(self, blur, reach, power):
        super(BergEntropy, self).__init__()
        assert power < 1, "The entropy exponent is not admissible (should be <1)."

        self.blur = blur
        self.reach = reach
        self.power = power
        self.__name__ = 'PowerEntropy'

    def entropy(self, x):
        return self.reach * (x - 1 - x.log())

    def legendre_entropy(self, x):
        return - self.reach * (1 - (x / self.reach)).log()


class PowerEntropy(Entropy):
    def __init__(self, blur, reach, power):
        super(PowerEntropy, self).__init__()
        assert power < 1, "The entropy exponent is not admissible (should be <1)."

        self.blur = blur
        self.reach = reach
        self.power = power
        self.__name__ = 'PowerEntropy'

    def grad_legendre(self, x):
        return (1 - (x / (self.reach * (1-self.power)))) ** (self.power - 1)

    def aprox(self, x):
        delta = -(x / (self.blur * (1-self.power))) + (self.reach / self.blur) + \
                torch.tensor([self.reach / self.blur], dtype=x.dtype).log()
        return (1 - self.power) * (self.reach - self.blur * log_lambertw(delta))

    def init_potential(self, a, x, b, y, p):
        f = self.reach * (1 - self.power) * (b.sum(dim=1) ** (1 / (self.power - 1)) - 1)[:,None]
        g = self.reach * (1 - self.power) * (a.sum(dim=1) ** (1 / (self.power - 1)) - 1)[:,None]
        return f, g