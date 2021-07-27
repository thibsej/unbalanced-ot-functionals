import torch
from .torch_lambertw import log_lambertw
from .utils import scal, convolution


# TODO: Add  multiscale exponential version of the KL projections
class Entropy(object):
    """
    Object that defines the required modules for entropy functions.
    """

    def entropy(self, x):
        """Pointwise entropy used in the definition of Csiszar-divergence.

        Parameters
        ----------
        x: torch.Tensor of size [Nsample] or [Batch, Nsample]

        Returns
        -------
        torch.Tensor of size [Nsample] or [Batch, Nsample]
        """
        raise NotImplementedError

    def legendre_entropy(self, x):
        """Pointwise Legendre transform of entropy used in the dual of
        Csiszar-divergence.

        Parameters
        ----------
        x: torch.Tensor of size [Nsample] or [Batch, Nsample]

        Returns
        -------
        torch.Tensor of size [Nsample] or [Batch, Nsample]
        """
        raise NotImplementedError

    def grad_legendre(self, x):
        """Gradient of the Legendre transform.

        Parameters
        ----------
        x: torch.Tensor of size [Nsample] or [Batch, Nsample]

        Returns
        -------
        torch.Tensor of size [Nsample] or [Batch, Nsample]
        """
        raise NotImplementedError

    def aprox(self, x):
        """
        Anisotropic Proximity operator.
        The function returned is $x mapsto -Aprox(-x)$.

        Parameters
        ----------
        x: torch.Tensor of size [Nsample] or [Batch, Nsample]

        Returns
        -------
        torch.Tensor of size [Nsample] or [Batch, Nsample]
        """
        raise NotImplementedError

    def kl_prox(self, x):
        """
        Kullback Proximity operator of $phi^*$. Takes as input
        <beta, exp(g-C / blur)> and returns the update e^(f / blur).

        Parameters
        ----------
        x: torch.Tensor of size [Nsample] or [Batch, Nsample]

        Returns
        -------
        torch.Tensor of size [Nsample] or [Batch, Nsample]
        """
        raise NotImplementedError

    def init_potential(self, a, x, b, y, cost):
        """
        Computes the initialization of the sinkhorn algorithm based on the
        asymptotic epsilon going to infinity.

        Parameters
        ----------
        a: torch.Tensor of size [Batch, Nsample]
        Mass located at each position x of the first measure.

        x: torch.Tensor of size [Batch, Nsample, Dim]
        Support of the first measure

        b: torch.Tensor of size [Batch, Msample]
        Mass located at each position x of the second measure.

        y: torch.Tensor of size [Batch, Msample, Dim]
        Support of the second measure

        cost: function
        Function which computes the cost matrix C(x,y).

        Returns
        -------
        f: torch.Tensor of size [Batch, Nsample]
        Dual potential w.r.t first measure.

        g: torch.Tensor of size [Batch, Msample]
        Dual potential w.r.t second measure.
        """
        raise NotImplementedError

    def error_sink(self, f, g):
        """
        Function which computes the error between two iterates of the
        Sinkhorn algorithm to estimate its convergence.
        Default is uniform norm (or Hilbert distance for Balanced OT).

        Parameters
        ----------
        f: torch.Tensor of size [Batch, Nsample]
        Dual potential w.r.t first measure.

        g: torch.Tensor of size [Batch, Msample]
        Dual potential w.r.t second measure.

        Returns
        -------
        torch.Tensor of size [Batch]
        Estimation of the convergence
        """
        return (f - g).abs().max()

    def output_regularized(self, a, x, b, y, cost, f, g):
        """Outputs the cost of the regularized OT

        Parameters
        ----------
        a: torch.Tensor of size [Batch, Nsample]
        Mass located at each position x of the first measure.

        x: torch.Tensor of size [Batch, Nsample, Dim]
        Support of the first measure

        b: torch.Tensor of size [Batch, Msample]
        Mass located at each position x of the second measure.

        y: torch.Tensor of size [Batch, Msample, Dim]
        Support of the second measure

        cost: function
        Function which computes the cost matrix C(x,y).

        f: torch.Tensor of size [Batch, Nsample]
        Dual potential w.r.t first measure.

        g: torch.Tensor of size [Batch, Msample]
        Dual potential w.r.t second measure.

        Returns
        -------
        torch.Tensor of size [Batch]
        """
        phis, partial_phis = self.legendre_entropy, self.grad_legendre

        def output_pot(x):
            return -phis(-x) - 0.5 * self.blur * partial_phis(-x)

        return (
            scal(a, output_pot(f))
            + scal(b, output_pot(g))
            + self.blur * a.sum(1)[:, None] * b.sum(1)[:, None]
        )

    def output_sinkhorn(self, a, x, b, y, cost, f_xy, f_xx, g_xy, g_yy):
        """Outputs the cost of the Sinkhorn divergence

        Parameters
        ----------
        a: torch.Tensor of size [Batch, Nsample]
        Mass located at each position x of the first measure.

        x: torch.Tensor of size [Batch, Nsample, Dim]
        Support of the first measure.

        b: torch.Tensor of size [Batch, Msample]
        Mass located at each position x of the second measure.

        y: torch.Tensor of size [Batch, Msample, Dim]
        Support of the second measure.

        cost: function
        Function which computes the cost matrix C(x,y).

        f_xy: torch.Tensor of size [Batch, Nsample]
        Dual potential w.r.t first measure compared to the second measure.

        f_xx: torch.Tensor of size [Batch, Nsample]
        Symmetric dual potential of first measure.

        g_xy: torch.Tensor of size [Batch, Msample]
        Dual potential w.r.t second measure compared to the first measure.

        g_yy: torch.Tensor of size [Batch, Msample]
        Symmetric dual potential of second measure.

        Returns
        -------
        torch.Tensor of size [Batch]
        """
        phis, partial_phis = self.legendre_entropy, self.grad_legendre

        def output_pot(x):
            return -phis(-x) - 0.5 * self.blur * partial_phis(-x)

        return scal(a, output_pot(f_xy) - output_pot(f_xx)) + scal(
            b, output_pot(g_xy) - output_pot(g_yy)
        )

    def output_hausdorff(self, a, x, b, y, cost, f_xy, f_xx, g_xy, g_yy):
        """Outputs the cost of the Hausdorff divergence

        Parameters
        ----------
        a: torch.Tensor of size [Batch, Nsample]
        Mass located at each position x of the first measure.

        x: torch.Tensor of size [Batch, Nsample, Dim]
        Support of the first measure.

        b: torch.Tensor of size [Batch, Msample]
        Mass located at each position x of the second measure.

        y: torch.Tensor of size [Batch, Msample, Dim]
        Support of the second measure.

        cost: function
        Function which computes the cost matrix C(x,y).

        f_xy: torch.Tensor of size [Batch, Nsample]
        Extrapolated symmetric dual potential w.r.t first measure.

        f_xx: torch.Tensor of size [Batch, Nsample]
        Symmetric dual potential of first measure.

        g_xy: torch.Tensor of size [Batch, Msample]
        Extrapolated symmetric dual potential w.r.t second measure.

        g_yy: torch.Tensor of size [Batch, Msample]
        Symmetric dual potential of second measure.

        Returns
        -------
        torch.Tensor of size [Batch]
        """
        phis, partial_phis = self.legendre_entropy, self.grad_legendre

        def output_pot(x):
            return phis(-x) + self.blur * partial_phis(-x)

        return scal(a, output_pot(f_xx) - output_pot(f_xy)) + scal(
            b, output_pot(g_yy) - output_pot(g_xy)
        )


class KullbackLeibler(Entropy):
    def __init__(self, blur, reach):
        super(KullbackLeibler, self).__init__()

        self.__name__ = "KullbackLeibler"
        self.blur = blur
        self.reach = reach

    def entropy(self, x):
        return self.reach * (x * x.log() - x + 1)

    def legendre_entropy(self, x):
        return self.reach * ((x / self.reach).exp() - 1)

    def grad_legendre(self, x):
        return (x / self.reach).exp()

    def aprox(self, x):
        z = self.blur / self.reach
        return (1 / (1 + z)) * x

    def kl_prox(self, x):
        z = self.blur / self.reach
        return x ** (-1 / (1 + z))

    def init_potential(self, a, x, b, y, cost):
        f = -self.reach * b.sum(dim=1).log()[:, None]
        g = -self.reach * a.sum(dim=1).log()[:, None]
        return f, g


class Balanced(Entropy):
    def __init__(self, blur):
        super(Balanced, self).__init__()

        self.__name__ = "Balanced"
        self.blur = blur

    def entropy(self, x):
        if x == 1:
            return 0
        else:
            return float("inf")

    def legendre_entropy(self, x):
        return x

    def grad_legendre(self, x):
        return 1

    def aprox(self, x):
        return x

    def kl_prox(self, x):
        return x ** (-1)

    def init_potential(self, a, x, b, y, cost):
        f, g = convolution(a, x, b, y, cost)
        scal_prod = scal(b, g)
        f = f - 0.5 * scal_prod[:, None]
        g = g - 0.5 * scal_prod[:, None]
        return f, g

    def error_sink(self, f, g):
        return (
            torch.max((f - g), dim=1)[0] - torch.min((f - g), dim=1)[0]
        ).max()

    def output_regularized(self, a, x, b, y, cost, f, g):
        return scal(a, f) + scal(b, g)

    def output_sinkhorn(self, a, x, b, y, cost, f_xy, f_xx, g_xy, g_yy):
        return scal(a, f_xy - f_xx) + scal(b, g_xy - g_yy)

    def output_hausdorff(self, a, x, b, y, cost, f_xy, f_xx, g_xy, g_yy):
        return scal(a, f_xy - f_xx) + scal(b, g_xy - g_yy)


class Range(Entropy):
    def __init__(self, blur, reach_low, reach_up):
        super(Range, self).__init__()

        self.__name__ = "Range"
        self.blur = blur
        self.reach_low = reach_low
        self.reach_up = reach_up

    def entropy(self, x):
        if (x >= self.reach_low) & (x <= self.reach_up):
            return 0
        else:
            return float("inf")

    def legendre_entropy(self, x):
        return torch.max(self.reach_low * x, self.reach_up * x)

    def grad_legendre(self, x):
        return torch.max(-self.reach_low * x.sign(), self.reach_up * x.sign())

    def aprox(self, x):
        r0 = torch.tensor([self.reach_low], dtype=x.dtype)
        r1 = torch.tensor([self.reach_up], dtype=x.dtype)
        return torch.min(
            torch.max(
                torch.tensor([0.0], dtype=x.dtype), x - self.blur * r1.log()
            ),
            x - self.blur * r0.log(),
        )

    def kl_prox(self, x):
        r0 = torch.tensor([self.reach_low], dtype=x.dtype)
        r1 = torch.tensor([self.reach_up], dtype=x.dtype)
        return torch.min(
            torch.max(torch.tensor([1.0], dtype=x.dtype), r0 * x ** (-1)),
            r1 * x ** (-1),
        )

    def init_potential(self, a, x, b, y, cost):
        f, g = torch.zeros_like(a), torch.zeros_like(b)
        return f, g

    def output_regularized(self, a, x, b, y, cost, f, g):
        phis = self.legendre_entropy

        def output_pot(x):
            return -phis(-x)

        func = scal(a, output_pot(f)) + scal(b, output_pot(g))
        C = cost(x, y)
        expC = (
            a[:, :, None]
            * b[:, None, :]
            * (1 - ((f[:, :, None] + g[:, None, :] - C) / self.blur).exp())
        )
        func = func + torch.sum(self.blur * expC, dim=(1, 2))
        return func

    def output_sinkhorn(
        self, a, x, b, y, cost, f_xy, f_x1, f_x2, g_xy, g_y1, g_y2
    ):
        phis = self.legendre_entropy

        def output_pot(x):
            return -phis(-x)

        func = scal(
            a,
            output_pot(f_xy) - 0.5 * output_pot(f_x1) - 0.5 * output_pot(f_x2),
        ) + scal(
            b,
            output_pot(g_xy) - 0.5 * output_pot(g_y1) - 0.5 * output_pot(g_y2),
        )
        Cxy, Cxx, Cyy = cost(x, y), cost(x, x), cost(y, y)

        def expC(a, b, f, g, C):
            return (
                a[:, :, None]
                * b[:, None, :]
                * (1 - ((f[:, :, None] + g[:, None, :] - C) / self.blur).exp())
            )

        func = (
            func
            + torch.sum(self.blur * expC(a, b, f_xy, g_xy, Cxy), dim=(1, 2))
            - 0.5
            * torch.sum(self.blur * expC(a, a, f_x1, f_x2, Cxx), dim=(1, 2))
            - 0.5
            * torch.sum(self.blur * expC(b, b, g_y1, g_y2, Cyy), dim=(1, 2))
        )
        return func


class TotalVariation(Entropy):
    def __init__(self, blur, reach):
        super(TotalVariation, self).__init__()

        self.__name__ = "TotalVariation"
        self.blur = blur
        self.reach = reach

    def entropy(self, x):
        return self.reach * (x - 1).abs()

    def legendre_entropy(self, x):
        return x

    def grad_legendre(self, x):
        return 1

    def aprox(self, x):
        return torch.min(
            torch.max(-self.reach * torch.ones_like(x), x),
            self.reach * torch.ones_like(x),
        )

    def kl_prox(self, x):
        return torch.min(
            torch.max(
                torch.tensor([-self.reach / self.blur]).exp()
                * torch.ones_like(x),
                x ** (-1),
            ),
            torch.tensor([self.reach / self.blur]).exp() * torch.ones_like(x),
        )

    def init_potential(self, a, x, b, y, cost):
        aprox = self.aprox
        mask_a, mask_b = (
            torch.eq(a.sum(1), torch.ones(a.size(0), dtype=a.dtype)),
            torch.eq(b.sum(1), torch.ones(b.size(0), dtype=b.dtype)),
        )
        f, g = torch.ones_like(a), torch.ones_like(b)
        if mask_a.all() or mask_b.all():
            f, g = convolution(a, x, b, y, cost)
            scal_prod = scal(b, g)
            f = f - 0.5 * scal_prod[:, None]
            g = g - 0.5 * scal_prod[:, None]
            f, g = -aprox(-f), -aprox(-g)
        f[~mask_a, :] = (
            -self.reach * (a[~mask_a, :].sum(1)).log().sign()[:, None]
        )
        g[~mask_b, :] = (
            -self.reach * (b[~mask_b, :].sum(1)).log().sign()[:, None]
        )
        return f, g

    def error_sink(self, f, g):
        err2 = (f - g).abs().max()
        return err2

    def output_regularized(self, a, x, b, y, cost, f, g):
        phis = self.legendre_entropy

        def output_pot(x):
            return -phis(-x)

        func = scal(a, output_pot(f)) + scal(b, output_pot(g))
        C = cost(x, y)
        expC = (
            a[:, :, None]
            * b[:, None, :]
            * (1 - ((f[:, :, None] + g[:, None, :] - C) / self.blur).exp())
        )
        func = func + torch.sum(self.blur * expC, dim=(1, 2))
        return func

    def output_sinkhorn(
        self, a, x, b, y, cost, f_xy, f_x1, f_x2, g_xy, g_y1, g_y2
    ):
        phis = self.legendre_entropy

        def output_pot(x):
            return -phis(-x)

        func = scal(
            a,
            output_pot(f_xy) - 0.5 * output_pot(f_x1) - 0.5 * output_pot(f_x2),
        ) + scal(
            b,
            output_pot(g_xy) - 0.5 * output_pot(g_y1) - 0.5 * output_pot(g_y2),
        )
        Cxy, Cxx, Cyy = cost(x, y), cost(x, x), cost(y, y)

        def expC(a, b, f, g, C):
            return (
                a[:, :, None]
                * b[:, None, :]
                * (1 - ((f[:, :, None] + g[:, None, :] - C) / self.blur).exp())
            )

        func = (
            func
            + torch.sum(self.blur * expC(a, b, f_xy, g_xy, Cxy), dim=(1, 2))
            - 0.5
            * torch.sum(self.blur * expC(a, a, f_x1, f_x2, Cxx), dim=(1, 2))
            - 0.5
            * torch.sum(self.blur * expC(b, b, g_y1, g_y2, Cyy), dim=(1, 2))
        )
        return func


class PowerEntropy(Entropy):
    def __init__(self, blur, reach, power):
        super(PowerEntropy, self).__init__()
        assert (
            power < 1
        ), "The entropy exponent is not admissible (should be <1)."

        self.__name__ = "PowerEntropy"
        self.blur = blur
        self.reach = reach
        self.power = power

    def entropy(self, x):
        if self.power == 0:
            return self.reach * (x - 1 - x.log())
        else:
            s = self.power / (self.power - 1)
            return (self.reach / (s * (s - 1))) * (x ** s - s * (x - 1) - 1)

    def legendre_entropy(self, x):
        if self.power == 0:
            return -self.reach * (1 - (x / self.reach)).log()
        else:
            return (
                self.reach
                * (1 - 1 / self.power)
                * ((1 + x / (self.reach * (self.power - 1))) ** self.power - 1)
            )

    def grad_legendre(self, x):
        return (1 - (x / (self.reach * (1 - self.power)))) ** (self.power - 1)

    def aprox(self, x):
        delta = (
            -(x / (self.blur * (1 - self.power)))
            + (self.reach / self.blur)
            + torch.tensor([self.reach / self.blur], dtype=x.dtype).log()
        )
        return (1 - self.power) * (
            self.reach - self.blur * log_lambertw(delta)
        )

    def kl_prox(self, x):
        z = torch.tensor([self.reach / self.blur])
        delta = z + z.log() - x.log() / (1 - self.power)
        return ((1 - self.power) * (log_lambertw(delta) - z)).exp()

    def init_potential(self, a, x, b, y, cost):
        f = (
            self.reach
            * (1 - self.power)
            * (b.sum(dim=1) ** (1 / (self.power - 1)) - 1)[:, None]
        )
        g = (
            self.reach
            * (1 - self.power)
            * (a.sum(dim=1) ** (1 / (self.power - 1)) - 1)[:, None]
        )
        return f, g
