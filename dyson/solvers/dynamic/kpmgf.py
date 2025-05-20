"""Kernel polynomial method Green's function solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.solvers.solver import DynamicSolver

if TYPE_CHECKING:
    from typing import Literal

    from dyson.grids.frequency import RealFrequencyGrid
    from dyson.typing import Array


def _infer_max_cycle(moments: Array) -> int:
    """Infer the maximum number of cycles from the moments."""
    return moments.shape[0] - 1


class KPMGF(DynamicSolver):
    """Kernel polynomial method Green's function solver [1]_.

    Args:
        moments: Chebyshev moments of the Green's function.
        grid: Real frequency grid upon which to evaluate the Green's function.
        scaling: Scaling factors to ensure the energy scale of the Lehmann representation is in
            `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`.

    References:
        [1] A. Weiβe, G. Wellein, A. Alvermann, and H. Fehske, Rev. Mod. Phys. 78, 275 (2006).
    """

    def __init__(
        self,
        moments: Array,
        grid: RealFrequencyGrid,
        scaling: tuple[float, float],
        kernel_type: Literal["dirichlet", "lorentz", "fejer", "lanczos", "jackson"] | None = None,
        trace: bool = False,
        include_real: bool = True,
        max_cycle: int | None = None,
        lorentz_parameter: float = 0.1,
        lanczos_order: int = 2,
    ):
        """Initialise the solver.

        Args:
            moments: Chebyshev moments of the Green's function.
            grid: Real frequency grid upon which to evaluate the Green's function.
            scaling: Scaling factors to ensure the energy scale of the Lehmann representation is in
                `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`.
            kernel_type: Kernel to apply to regularise the Chebyshev representation.
            trace: Whether to return only the trace.
            include_real: Whether to include the real part of the Green's function.
            max_cycle: Maximum number of iterations.
            lorentz_parameter: Lambda parameter for the Lorentz kernel.
            lanczos_order: Order of the Lanczos kernel.
        """
        self._moments = moments
        self._grid = grid
        self._scaling = scaling
        self.kernel_type = kernel_type if kernel_type is not None else "dirichlet"
        self.trace = trace
        self.include_real = include_real
        self.max_cycle = max_cycle if max_cycle is not None else _infer_max_cycle(moments)
        self.lorentz_parameter = lorentz_parameter
        self.lanczos_order = lanczos_order

    def _coefficients_dirichlet(self, iteration: int) -> Array:
        """Get the expansion coefficients for the Dirichlet kernel."""
        return np.ones(iteration)

    def _coefficients_lorentz(self, iteration: int) -> Array:
        """Get the expansion coefficients for the Lorentz kernel."""
        iters = np.arange(1, iteration + 1)
        coefficients = np.sinh(self.lorentz_parameter * (1 - iters / iteration))
        coefficients /= np.sinh(self.lorentz_parameter)
        return coefficients

    def _coefficients_fejer(self, iteration: int) -> Array:
        """Get the expansion coefficients for the Fejér kernel."""
        iters = np.arange(1, iteration + 1)
        return 1 - iters / (iteration + 1)

    def _coefficients_lanczos(self, iteration: int) -> Array:
        """Get the expansion coefficients for the Lanczos kernel."""
        iters = np.arange(1, iteration + 1)
        factor = np.pi * iters / iteration
        return (np.sin(factor) / factor) ** self.lanczos_order

    def _coefficients_jackson(self, iteration: int) -> Array:
        """Get the expansion coefficients for the Jackson kernel."""
        iters = np.arange(1, iteration + 1)
        norm = 1 / (iteration + 1)
        coefficients = (iteration - iters + 1) * np.cos(np.pi * iters * norm)
        coefficients += np.sign(np.pi * iters * norm) / np.tan(np.pi * norm)
        coefficients *= norm
        return coefficients

    def kernel(self, iteration: int | None = None) -> Array:
        """Run the solver.

        Args:
            iteration: The iteration number.

        Returns:
            The Green's function on the real frequency grid.
        """
        if iteration is None:
            iteration = self.max_cycle

        # Get the moments -- allow input to already be traced
        moments = util.as_trace(self.moments[: iteration + 1], 3).astype(complex)

        # Scale the grid
        scaled_grid = (self.grid - self.scaling[1]) / self.scaling[0]
        grids = (np.ones_like(scaled_grid), scaled_grid)

        # Initialise the polynomial
        coefficients = getattr(self, f"_coefficients_{self.kernel_type}")(iteration + 1)
        polynomial = np.array([moments[0] * coefficients[0]] * self.grid.size)

        # Iteratively compute the Green's function
        for cycle in range(1, iteration + 1):
            polynomial += (
                util.einsum("z,...->z...", grids[-1], moments[cycle]) * coefficients[cycle]
            )
            grids = (grids[-1], 2 * scaled_grid * grids[-1] - grids[-2])

        # Get the Green's function
        polynomial /= np.sqrt(1 - scaled_grid**2)
        polynomial /= np.sqrt(self.scaling[0] ** 2 - (self.grid - self.scaling[1]) ** 2)
        greens_function = -polynomial

        return greens_function if self.include_real else greens_function.imag

    @property
    def moments(self) -> Array:
        """Get the moments of the self-energy."""
        return self._moments

    @property
    def grid(self) -> RealFrequencyGrid:
        """Get the real frequency grid."""
        return self._grid

    @property
    def scaling(self) -> tuple[float, float]:
        """Get the scaling factors."""
        return self._scaling

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.moments.shape[-1]
