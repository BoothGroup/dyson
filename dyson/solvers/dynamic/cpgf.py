"""Chebyshev polynomial Green's function solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.solvers.solver import DynamicSolver

if TYPE_CHECKING:
    from typing import Any, Literal

    from dyson.expression.expression import BaseExpression
    from dyson.grids.frequency import RealFrequencyGrid
    from dyson.typing import Array
    from dyson.lehmann import Lehmann


def _infer_max_cycle(moments: Array) -> int:
    """Infer the maximum number of cycles from the moments."""
    return moments.shape[0] - 1


class CPGF(DynamicSolver):
    """Chebyshev polynomial Green's function solver [1]_.

    Args:
        moments: Chebyshev moments of the Green's function.
        grid: Real frequency grid upon which to evaluate the Green's function.
        scaling: Scaling factors to ensure the energy scale of the Lehmann representation is in
            `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`.

    References:
        [1] A. Ferreira, and E. R. Mucciolo, Phys. Rev. Lett. 115, 106601 (2015).
    """

    trace: bool = False
    include_real: bool = True
    ordering: Literal["time-ordered", "advanced", "retarded"] = "time-ordered"
    _options: set[str] = {"trace", "include_real", "ordering"}

    def __init__(  # noqa: D417
        self,
        moments: Array,
        grid: RealFrequencyGrid,
        scaling: tuple[float, float],
        max_cycle: int | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            moments: Chebyshev moments of the Green's function.
            grid: Real frequency grid upon which to evaluate the Green's function.
            scaling: Scaling factors to ensure the energy scale of the Lehmann representation is in
                `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`.
            max_cycle: Maximum number of iterations.
            trace: Whether to return only the trace.
            include_real: Whether to include the real part of the Green's function.
            ordering: Time ordering of the resolvent.
        """
        self._moments = moments
        self._grid = grid
        self._scaling = scaling
        self.max_cycle = max_cycle if max_cycle is not None else _infer_max_cycle(moments)
        self.set_options(**kwargs)

        if self.ordering == "time-ordered":
            raise NotImplementedError("ordering='time-ordered' is not implemented for CPGF.")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> CPGF:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        if "grid" not in kwargs:
            raise ValueError("Missing required argument grid.")
        max_cycle = kwargs.pop("max_cycle", 16)
        energies, couplings = self_energy.diagonalise_matrix_with_projection(static, overlap=overlap)
        emin = np.min(energies)
        emax = np.max(energies)
        scaling = ((emax - emin) / (2.0 - 1e-3), (emax + emin) / 2.0)
        greens_function = self_energy.__class__(energies, couplings, chempot=self_energy.chempot)
        moments = greens_function.chebyshev_moments(range(max_cycle + 1), scaling=scaling)
        return cls(moments, kwargs.pop("grid"), scaling, max_cycle=max_cycle, **kwargs)

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> CPGF:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        if "grid" not in kwargs:
            raise ValueError("Missing required argument grid.")
        max_cycle = kwargs.pop("max_cycle", 16)
        diag = expression.diagonal()
        emin = np.min(diag)
        emax = np.max(diag)
        scaling = ((emax - emin) / (2.0 - 1e-3), (emax + emin) / 2.0)
        moments = expression.build_gf_chebyshev_moments(max_cycle + 1, scaling=scaling)
        return cls(moments, kwargs.pop("grid"), scaling, max_cycle=max_cycle, **kwargs)

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
        scaled_eta = self.grid.eta / self.scaling[0]
        shifted_grid = scaled_grid + 1j * scaled_eta

        # Initialise factors
        numerator = shifted_grid - 1j * np.sqrt(1 - shifted_grid**2)
        denominator = np.sqrt(1 - shifted_grid**2)

        # Iteratively compute the Green's function
        shape = (self.grid.size,) if self.trace else (self.grid.size, self.nphys, self.nphys)
        greens_function = np.zeros(shape, dtype=complex)
        kernel = 1.0 / denominator
        for cycle in range(iteration + 1):
            factor = 1.0j * (2.0 - int(cycle == 0)) / self.scaling[0]
            greens_function -= util.einsum("z,...->z...", kernel, moments[cycle]) * factor
            kernel *= numerator

        if self.ordering == "advanced":
            greens_function = greens_function.conj()

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
