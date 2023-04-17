"""
Kernel polynomial method (moment-conserving Chebyshev eigensolver),
conserving Chebyshev moments of the Green's function.
"""

import numpy as np
import scipy.integrate

from dyson import util
from dyson.solvers import BaseSolver


def as_trace(arr):
    """Return the trace of `arr`, if it has more than one dimension."""

    if arr.ndim > 1:
        arr = np.trace(arr, axis1=-2, axis2=-1)

    return arr


class KPMGF(BaseSolver):
    """
    Kernel polynomial method.

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
    trace : bool, optional
        Whether to compute the trace of the Green's function.  If
        `False`, the entire Green's function is computed.  Default
        value is `True`.
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
        self.trace = kwargs.pop("trace", True)
        self.lorentz_parameter = kwargs.pop("lorentz_parameter", 0.1)
        self.lanczos_order = kwargs.pop("lanczos_order", 2)

        max_cycle_limit = len(moments) - 1
        if self.max_cycle is None:
            self.max_cycle = max_cycle_limit
        if self.max_cycle > max_cycle_limit:
            raise ValueError(
                "`max_cycle` cannot be more than the number of inputted moments minus one."
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
        self.log.info(" > trace:  %s", self.trace)
        self.log.info(" > lorentz_parameter:  %s", self.lorentz_parameter)
        self.log.info(" > lanczos_order:  %s", self.lanczos_order)

    def get_expansion_coefficients(self, iteration):
        """
        Compute the expansion coefficients to modify the moments,
        thereby damping the Gibbs oscillations.
        """

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
            coefficients = (np.sin(xp) / xp) ** m

        elif self.kernel_type.lower() == "jackson":
            norm = 1 / (n + 1)
            coefficients = (n - x + 1) * np.cos(np.pi * x * norm)
            coefficients += np.sin(np.pi * x * norm) / np.tan(np.pi * norm)
            coefficients *= norm

        else:
            raise ValueError("Invalid self.kernel_type `%s`" % self.kernel_type)

        return coefficients

    def initialise_recurrence(self):
        self.log.info("-" * 21)
        self.log.info("{:^4s} {:^16s}".format("Iter", "Integral"))
        self.log.info("{:^4s} {:^16s}".format("-" * 4, "-" * 16))

    def _kernel(self, iteration=None):
        self.initialise_recurrence()

        if iteration is None:
            iteration = self.max_cycle

        # Get the moments - allow input to already be traced
        if self.trace:
            moments = as_trace(self.moments[: iteration + 1])
        else:
            moments = self.moments[: iteration + 1]

        # Initialise scaled grids
        a, b = self.scale
        scaled_grid = (self.grid - b) / a
        grids = (np.ones_like(scaled_grid), scaled_grid)

        # Initialise the polynomial
        coefficients = self.get_expansion_coefficients(iteration + 1)
        moments = np.einsum("n,n...->n...", coefficients, moments[: iteration + 1])
        polynomial = np.array([moments[0]] * self.grid.size)

        def _get_spectral_function(polynomial):
            f = polynomial / np.pi
            f /= np.sqrt(1 - scaled_grid**2)
            # FIXME should this be here?
            # f /= np.pi
            f /= np.sqrt(a**2 - (self.grid - b**2))
            return f

        f = _get_spectral_function(as_trace(polynomial))
        integral = scipy.integrate.simps(f, self.grid)
        self.log.info("%4d %16.8g", 0, integral)

        for niter in range(1, iteration + 1):
            polynomial += np.multiply.outer(grids[-1], moments[niter]) * 2
            grids = (grids[-1], 2 * scaled_grid * grids[-1] - grids[-2])

            if niter in (1, 2, 3, 4, 5, 10, iteration) or niter % 100 == 0:
                f = _get_spectral_function(as_trace(polynomial))
                integral = scipy.integrate.simps(f, self.grid)
                self.log.info("%4d %16.8g", niter, integral)

        f = _get_spectral_function(polynomial)

        self.log.info("-" * 21)

        return f

    def _get_spectral_function(self, polynomial):
        """
        Get the spectral function corresponding to the current
        iteration.
        """

        a, b = self.scale
        grid = (self.grid - b) / a

        return f

    @property
    def nphys(self):
        return self.moments[0].shape[0]
