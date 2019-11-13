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

    def error_sink(self):
        """
        returns the function that controls the error for Sinkhorn iteration
        (Hilbert norm for balanced OT, uniform norm otherwise).
        :return: function
        """
        def err(f, g):
            return (f - g).abs().max()
        return err

    def output_regularized(self):
        """Outputs the cost of the regularized OT"""
        def output_cost(a, x, b, y, p, f, g):
            phis, partial_phis = self.legendre_entropy(), self.grad_legendre()
            output_pot = lambda x: - phis(-x) - 0.5 * self.blur * partial_phis(-x)
            return scal(a, output_pot(f)) + scal(b, output_pot(g)) + self.blur * a.sum(1)[:, None] * b.sum(1)[:, None]
        return output_cost

    def output_sinkhorn(self):
        """Outputs the cost of the Sinkhorn divergence"""
        def output_cost(a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
            phis, partial_phis = self.legendre_entropy(), self.grad_legendre()
            output_pot = lambda x: - phis(-x) - 0.5 * self.blur * partial_phis(-x)
            return scal(a, output_pot(f_xx) - output_pot(f_xy)) + scal(b, output_pot(g_yy) - output_pot(g_xy))
        return output_cost

    def output_hausdorff(self):
        """Outputs the cost of the Hausdorff divergence"""
        def output_cost(a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
            phis, partial_phis = self.legendre_entropy(), self.grad_legendre()
            output_pot = lambda x: phis(-x) + self.blur * partial_phis(-x)
            return scal(a, output_pot(f_xx) - output_pot(f_xy)) + scal(b, output_pot(g_yy) - output_pot(g_xy))
        return output_cost


class KullbackLeibler(Entropy):

    def __init__(self, blur, reach):
        super(KullbackLeibler, self).__init__()

        self.blur = blur
        self.reach = reach

    def entropy(self):
        def phi(x):
            return self.reach * (x * x.log() - x + 1)
        return phi

    def legendre_entropy(self):
        def phis(x):
            return self.reach * ( (x / self.reach).exp() - 1 )
        return phis

    def grad_legendre(self):
        def partial_phis(x):
            return (x / self.reach).exp()
        return partial_phis

    def aprox(self):
        z = self.blur / self.reach
        def aprox(x):
            return (1 / (1 + z)) * x
        return aprox

    def init_potential(self):
        def init_pot(a,x,b,y,p):
            f = - self.reach * b.sum(dim=1).log()[:,None]
            g = - self.reach * a.sum(dim=1).log()[:,None]
            return f, g
        return init_pot



class Balanced(Entropy):

    def __init__(self, blur):
        super(Balanced, self).__init__()

        self.blur = blur

    def entropy(self):
        def phi(x):
            if x == 1:
                return 0
            else:
                return float('inf')
        return phi

    def legendre_entropy(self):
        def phis(x):
            return x
        return phis

    def grad_legendre(self):
        def partial_phis(x):
            return 1
        return partial_phis

    def aprox(self):
        def aprox(x):
            return x
        return aprox

    def init_potential(self):
        def init_pot(a,x,b,y,p):
            f, g = convolution(a, x, b, y, p)
            scal_prod = scal(b, g)
            f = f - 0.5 * scal_prod[:, None]
            g = g - 0.5 * scal_prod[:, None]
            return f, g
        return init_pot

    def error_sink(self):
        def err(f, g):
            return (torch.max((f - g), dim=1)[0] - torch.min((f - g), dim=1)[0]).max()
        return err

    def output_regularized(self):
        def output_cost(a, x, b, y, p, f, g):
            return scal(a, f) + scal(b, g)
        return output_cost

    def output_sinkhorn(self):
        def output_cost(a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
            return scal(a, f_xy - f_xx) + scal(b, g_xy - g_yy)
        return output_cost

    def output_hausdorff(self):
        def output_cost(a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
            return scal(a, f_xx - f_xy) + scal(b, g_yy - g_xy)
        return output_cost


class Range(Entropy):

    def __init__(self, blur, reach_low, reach_up):
        super(Range, self).__init__()

        self.blur = blur
        self.reach_low = reach_low
        self.reach_up = reach_up

    def entropy(self):
        def phi(x):
            if (x >= self.reach_low) & (x <= self.reach_up):
                return 0
            else:
                return float('inf')
        return phi

    def legendre_entropy(self):
        def phis(x):
            return torch.max(self.reach_low * x, self.reach_up * x)
        return phis

    def grad_legendre(self):
        def partial_phis(x):
            return torch.max( - self.reach_low * x.sign(), self.reach_up * x.sign() )
        return partial_phis

    def aprox(self):
        def aprox(x):
            r0, r1 = torch.Tensor([self.reach_low]), torch.Tensor([self.reach_up])
            return torch.min(torch.max(torch.Tensor([0]), x - self.blur * r1.log()), x - self.blur * r0.log())
        return aprox

    def init_potential(self):
        def init_pot(a,x,b,y,p):
            f, g = torch.zeros_like(a), torch.zeros_like(b)
            return f, g
        return init_pot

    def output_regularized(self):
        def output_cost(a, x, b, y, p, f, g):
            phis, partial_phis = self.legendre_entropy(), self.grad_legendre()
            output_pot = lambda x: - phis(-x) - 0.5 * self.blur * partial_phis(-x)
            return scal(a, output_pot(f)) + scal(b, output_pot(g)) + self.blur * a.sum(1)[:, None] * b.sum(1)[:, None]
        return output_cost

    def output_sinkhorn(self):
        def output_cost(a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
            phis, partial_phis = self.legendre_entropy(), self.grad_legendre()
            output_pot = lambda x: - phis(-x) - 0.5 * self.blur * partial_phis(-x)
            return scal(a, output_pot(f_xx) - output_pot(f_xy)) + scal(b, output_pot(g_yy) - output_pot(g_xy))
        return output_cost

    def output_hausdorff(self):
        def output_cost(a, x, b, y, p, f_xy, f_xx, g_xy, g_yy):
            phis, partial_phis = self.legendre_entropy(), self.grad_legendre()
            output_pot = lambda x: phis(-x) + self.blur * partial_phis(-x)
            return scal(a, output_pot(f_xx) - output_pot(f_xy)) + scal(b, output_pot(g_yy) - output_pot(g_xy))
        return output_cost



class TotalVariation(Entropy):

    def __init__(self, blur, reach):
        super(TotalVariation, self).__init__()

        self.blur = blur
        self.reach = reach

    def entropy(self):
        def phi(x):
            return self.reach * (x - 1).abs()
        return phi

    def legendre_entropy(self):
        def phis(x):
            return x
        return phis

    def grad_legendre(self):
        def partial_phis(x):
            return 1
        return partial_phis

    def aprox(self):
        def aprox(x):
            torch.min(torch.max(-self.reach * torch.ones_like(x), x), self.reach * torch.ones_like(x))
            return x
        return aprox

    def init_potential(self):
        def init_pot(a,x,b,y,p):
            f, g = convolution(a, x, b, y, p)
            aprox = self.aprox()
            scal_prod = scal(b, g)
            f = f - 0.5 * scal_prod[:, None]
            g = g - 0.5 * scal_prod[:, None]
            return -aprox(-f), -aprox(-g)
        return init_pot

    def error_sink(self):
        def err(f, g):
            err1 = (torch.max((f - g), dim=1)[0] - torch.min((f - g), dim=1)[0]).max()
            err2 = (f-g).abs().max()
            return torch.min(err1, err2)
        return err


class PowerEntropy(Entropy):

    def __init__(self, blur, reach, power):
        super(PowerEntropy, self).__init__()
        assert power < 1, "The entropy exponent is not admissible (>1)."

        self.blur = blur
        self.reach = reach
        self.power = power

    def entropy(self):
        s = self.power / ( self.power - 1 )
        if s == 0:
            def phi(x):
                return self.reach * (x - 1 - x.log())
        else:
            def phi(x):
                return (self.reach / (s * (s - 1))) * (x**s - s*(x-1) - 1)
        return phi

    def legendre_entropy(self):
        if self.power == 0:
            def phis(x):
                return - self.reach * (1 - (x / self.reach)).log()
        else:
            def phis(x):
                return self.reach * (1 - 1 / self.power) * ((1 + (x / self.reach) / (self.power - 1)) ** self.power - 1)
        return phis

    def grad_legendre(self):
        def partial_phis(x):
            return ( 1 - (x / (self.reach * (1-self.power))) ) ** (self.power - 1)
        return partial_phis

    def aprox(self):
        def aprox(x):
            delta = -(x / (self.blur * (1-self.power))) - (self.reach / self.blur) + \
                    torch.Tensor([self.reach / self.blur]).log()
            return (1-self.power) * (self.reach- self.blur * log_lambertw(delta))
        return aprox

    def init_potential(self):
        def init_pot(a,x,b,y,p):
            f = self.reach * (1-self.power) * (b.sum(dim=1) ** (1 / (1-self.power) ) - 1)[:,None]
            g = self.reach * (1-self.power) * (a.sum(dim=1) ** (1 / (1-self.power) ) - 1)[:,None]
            return f, g
        return init_pot