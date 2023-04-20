"""
Chebyshev polynomial Green's function method, similar to the KPMGF
and also conserves Chebyshev moments of the Green's function.

Ref: https://doi.org/10.1103/PhysRevLett.115.106601
"""

import numpy as np
import scipy.integrate

from dyson import util
from dyson.solvers import BaseSolver
from dyson.solvers.kpmgf import as_trace


class CPGF(BaseSolver):
    """
    Chebyshev polynomial Green's function method.

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
    trace : bool, optional
        Whether to compute the trace of the Green's function.  If
        `False`, the entire Green's function is computed.  Default
        value is `True`.
    include_real : bool, optional
        Whether to include the real part of the Green's function in
        the computation.  Default value is `False`.

    Parameters
    ----------
    max_cycle : int, optional
        Maximum number of iterations. If `None`, perform as many as
        the inputted number of moments permits.  Default value is
        `None`.
    eta : float, optional
        Regularisation parameter.  Default value is 0.1.
    """

    def __init__(self, moments, grid, scale, **kwargs):
        # Input:
        self.moments = moments
        self.grid = grid
        self.scale = scale

        # Parameters
        self.max_cycle = kwargs.pop("max_cycle", None)
        # self.hermitian = True
        self.eta = kwargs.pop("eta", 0.1)
        self.trace = kwargs.pop("trace", True)
        self.include_real = kwargs.pop("include_real", False)

        max_cycle_limit = len(moments) - 1
        if self.max_cycle is None:
            self.max_cycle = max_cycle_limit
        if self.max_cycle > max_cycle_limit:
            raise ValueError(
                "`max_cycle` cannot be more than the number of inputted moments minus one."
            )

        # Base class:
        super().__init__(**kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        # self.log.info(" > hermitian:  %s", self.hermitian)
        self.log.info(" > grid:  %s[%d]", type(self.grid), len(self.grid))
        self.log.info(" > scale: %s", scale)
        self.log.info(" > eta: %s", self.eta)
        self.log.info(" > trace: %s", self.trace)
        self.log.info(" > include_real: %s", self.include_real)

    def initialise_recurrence(self):
        self.log.info("-" * 21)
        self.log.info("{:^4s} {:^16s}".format("Iter", "Integral"))
        self.log.info("{:^4s} {:^16s}".format("-" * 4, "-" * 16))

    def _kernel(self, iteration=None, trace=True):
        self.initialise_recurrence()

        if iteration is None:
            iteration = self.max_cycle

        filter_type = lambda arr: arr.imag if not self.include_real else arr

        # Get the moments - allow input to already be traced
        if self.trace:
            moments = as_trace(self.moments[: iteration + 1]).astype(complex)
        else:
            moments = self.moments[: iteration + 1].astype(complex)

        # Initialise scaled grids
        a, b = self.scale
        scaled_grid = (self.grid - b) / a
        scaled_eta = self.eta / a
        z = scaled_grid - 1.0j * scaled_eta

        # Initialise the Green's function
        fac = lambda n: (2.0 / 1.0j) / (1.0 + int(n == 0))
        num = z - 1.0j * np.sqrt(1.0 - z**2)
        den = np.sqrt(1.0 - z**2)
        gn = lambda n: fac(n) * num**n / den
        gf = -np.einsum("z,...->z...", gn(0), moments[0])
        #gf /= a * np.pi

        integral = scipy.integrate.simps(as_trace(gf.imag), self.grid)
        self.log.info("%4d %16.8g", 0, integral)

        for niter in range(1, iteration + 1):
            part = -np.einsum("z,...->z...", gn(niter), moments[niter])
            #part /= a * np.pi
            gf += part

            if niter in (1, 2, 3, 4, 5, 10, iteration) or niter % 100 == 0:
                integral = scipy.integrate.simps(as_trace(gf.imag), self.grid)
                self.log.info("%4d %16.8g", niter, integral)

        self.log.info("-" * 21)

        return filter_type(gf)
