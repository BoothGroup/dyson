"""
Kernel polynomial method (moment-conserving Chebyshev eigensolver),
conserving Chebyshev moments of the Green's function.
"""

import numpy as np
import scipy.integrate

from dyson import util
from dyson.solvers import BaseSolver


class KPMGF(BaseSolver):
    """

    Input
    -----
    moments : numpy.ndarray
        Chebyshev moments of the Green's function.
    grid : numpy.ndarray
        Real-valued frequency grid to plot the spectral function upon.
    scale : tuple of int
        Scaling parameters used to scale the spectrum to [-1, 1],
        given as `(a, b)` where

            a = (ωmax - ωmin) / (2 - ε)
            b = (ωmax + ωmin) / 2

        where ωmax and ωmin are the maximum and minimum energies in
        the spectrum, respectively, and ε is a small number shifting
        the spectrum values away from the boundaries.

    Parameters
    ----------
    max_cycle : int, optional
        Maximum number of iterations. If `None`, perform as many as
        the inputted number of moments permits. Default value is
        `None`.
    kernel_type : str, optional
        Kernel to apply to regularise the Chebyshev representation.
        Can be one of `{None, "lorentz", "lanczos", "jackson"}, or a
        callable whose arguments are the solver object and the
        iteration number. Default value is `None`.
    lorentz_parameter : float or callable, optional
        Lambda parameter for the Lorentz kernel, a float value which
        is then scaled by the number of Chebyshev moments. Default
        value is 0.1.
    lanczos_order : int
        Order parameter for the Lanczos kernel. Default value is 2.

    Returns
    -------
    spectral_function : numpy.ndarray
        Spectral function expressed on the input grid.
    """

    def __init__(self, moments, grid, scale, **kwargs):
        # Input:
        self.moments = moments
        self.grid = grid
        self.scale = scale

        # Parameters
        self.max_cycle = kwargs.pop("max_cycle", None)
        # self.hermitian = True
        self.kernel_type = kwargs.pop("kernel_type", None)
        self.lorentz_parameter = kwargs.pop("lorentz_parameter", 0.1)
        self.lanczos_order = kwargs.pop("lanczos_order", 2)

        max_cycle_limit = len(moments) - 1
        if self.max_cycle is None:
            self.max_cycle = max_cycle_limit
        if self.max_cycle > max_cycle_limit:
            raise ValueError(
                "`max_cycle` cannot be more than the number of " "inputted moments minus one."
            )

        # Base class:
        super().__init__(moments, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        # self.log.info(" > hermitian:  %s", self.hermitian)
        self.log.info(" > grid:  %s[%d]", type(self.grid), len(self.grid))
        self.log.info(" > scale: %s", scale)
        self.log.info(" > kernel_type:  %s", self.kernel_type)
        self.log.info(" > lorentz_parameter:  %s", self.lorentz_parameter)
        self.log.info(" > lanczos_order:  %s", self.lanczos_order)

        # Caching:
        self.iteration = None
        self.polynomial = None

    def get_expansion_coefficients(self, iteration=None):
        """
        Compute the expansion coefficients to modify the moments,
        thereby damping the Gibbs oscillations.
        """

        if iteration is None:
            iteration = self.iteration

        n = iteration
        x = np.arange(1, iteration + 1)

        if self.kernel_type is None or self.kernel_type.lower() == "dirichlet":
            coefficients = np.ones((n,))

        elif callable(self.kernel_type):
            coefficients = self.kernel_type(n)

        elif self.kernel_type.lower() == "lorentz":
            if callable(self.lorentz_parameter):
                λ = self.lorentz_parameter(n)
            else:
                λ = self.lorentz_parameter
            coefficients = np.sinh(λ * (1 - x / n))
            coefficients /= np.sinh(λ)

        elif self.kernel_type.lower() == "fejer":
            coefficients = 1 - x / (n + 1)

        elif self.kernel_type.lower() == "lanczos":
            xp = np.pi * x / n
            m = self.lanczos_order
            coefficients = (np.sign(xp) / xp) ** m

        elif self.kernel_type.lower() == "jackson":
            norm = 1 / (n + 1)
            coefficients = (n - x + 1).astype(float)
            coefficients *= np.cos(np.pi * x * norm)
            coefficients += np.sin(np.pi * x * norm)
            coefficients /= np.tan(np.pi * norm)
            coefficients *= norm

        else:
            raise ValueError("Invalid self.kernel_type `%s`" % self.kernel_type)

        return coefficients

    def initialise_recurrence(self):
        self.log.info("-" * 21)
        self.log.info("{:^4s} {:^16s}".format("Iter", "Integral"))
        self.log.info("{:^4s} {:^16s}".format("-" * 4, "-" * 16))

        self.iteration = 0
        self.polynomial = np.concatenate((self.moments[0],) * self.grid.size)
        self.polynomial = self.polynomial.reshape(self.grid.size, self.nphys, self.nphys)

    def _kernel(self, iteration=None, trace=True):
        if self.iteration is None:
            self.initialise_recurrence()

        if iteration is None:
            iteration = self.max_cycle
        if iteration < self.iteration:
            raise ValueError(
                "Cannot compute spectral function for an " "iteration number already passed."
            )

        coefficients = self.get_expansion_coefficients(iteration + 1)
        moments = np.einsum("n,npq->npq", coefficients, self.moments[: iteration + 1])

        # Skip forward polynomial values to starting iteration
        a, b = self.scale
        scaled_grid = (self.grid - b) / a
        grids = (np.ones_like(scaled_grid), scaled_grid)
        for n in range(1, self.iteration):
            grids = (grids[-1], 2 * scaled_grid * grids[-1] - grids[-2])

        f = self._get_spectral_function(trace=True)
        integral = scipy.integrate.simps(f, self.grid)
        self.log.info("%4d %16.8g", self.iteration, integral)

        while self.iteration < iteration:
            self.iteration += 1
            self.polynomial += np.einsum("pq,w->wpq", moments[self.iteration], grids[-1]) * 2
            grids = (grids[-1], 2 * scaled_grid * grids[-1] - grids[-2])

            if self.iteration in (1, 2, 3, 4, 5, 10, iteration) or self.iteration % 100 == 0:
                f = self._get_spectral_function(trace=True)
                integral = scipy.integrate.simps(f, self.grid)
                self.log.info("%4d %16.8g", self.iteration, integral)

        f = self._get_spectral_function(trace=trace)

        if self.iteration == self.max_cycle:
            self.log.info("-" * 89)

        return f

    def _get_spectral_function(self, trace=True):
        """
        Get the spectral function corresponding to the current
        iteration.
        """

        a, b = self.scale
        grid = (self.grid - b) / a

        f = self.polynomial.copy()
        f /= np.pi
        f /= np.sqrt(1 - grid ** 2)[:, None, None]
        # FIXME should this be here?
        # f /= np.pi
        f /= np.sqrt(a ** 2 - (self.grid - b ** 2))[:, None, None]

        if trace:
            # FIXME do this sooner?
            return np.trace(f, axis1=1, axis2=2)

        return f

    @property
    def nphys(self):
        return self.moments[0].shape[0]
