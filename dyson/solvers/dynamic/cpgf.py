"""Chebyshev polynomial Green's function solver [ferreira2015]_.

.. [ferreira2015] Ferreira, A., & Mucciolo, E. R. (2015). Critical delocalization of Chiral zero
   energy modes in graphene. Physical Review Letters, 115(10).
   https://doi.org/10.1103/physrevlett.115.106601
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import console, printing, util
from dyson import numpy as np
from dyson.representations.dynamic import Dynamic
from dyson.representations.enums import Component, Ordering, Reduction
from dyson.solvers.solver import DynamicSolver

if TYPE_CHECKING:
    from typing import Any

    from dyson.expressions.expression import BaseExpression
    from dyson.grids.frequency import RealFrequencyGrid
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array


def _infer_max_cycle(moments: Array) -> int:
    """Infer the maximum number of cycles from the moments."""
    return moments.shape[0] - 1


class CPGF(DynamicSolver):
    """Chebyshev polynomial Green's function solver.

    Args:
        moments: Chebyshev moments of the Green's function.
        grid: Real frequency grid upon which to evaluate the Green's function.
        scaling: Scaling factors to ensure the energy scale of the Lehmann representation is in
            `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`.
    """

    reduction: Reduction = Reduction.NONE
    component: Component = Component.FULL
    ordering: Ordering = Ordering.ORDERED
    _options: set[str] = {"reduction", "component", "ordering"}

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
            component: The component of the dynamic representation to solve for.
            reduction: The reduction of the dynamic representation to solve for.
            ordering: Time ordering of the resolvent.
        """
        self._moments = moments
        self._grid = grid
        self._scaling = scaling
        self.max_cycle = max_cycle if max_cycle is not None else _infer_max_cycle(moments)
        self.set_options(**kwargs)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.moments.ndim != 3 or self.moments.shape[1] != self.moments.shape[2]:
            raise ValueError(
                "moments must be a 3D array with the second and third dimensions equal."
            )
        if _infer_max_cycle(self.moments) < self.max_cycle:
            raise ValueError("not enough moments provided for the specified max_cycle.")
        if self.ordering == Ordering.ORDERED:
            raise NotImplementedError(f"{self.ordering} is not implemented for CPGF.")

        # Print the input information
        cond = printing.format_float(
            np.linalg.cond(self.moments[0]), threshold=1e10, scientific=True, precision=4
        )
        scaling = (
            printing.format_float(self.scaling[0]),
            printing.format_float(self.scaling[1]),
        )
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        console.print(f"Number of moments: [input]{self.moments.shape[0]}[/input]")
        console.print(f"Scaling parameters: [input]({scaling[0]}, {scaling[1]})[/input]")
        console.print(f"Overlap condition number: {cond}")

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
        energies, couplings = self_energy.diagonalise_matrix_with_projection(
            static, overlap=overlap
        )
        if "scaling" not in kwargs:
            scaling = util.get_chebyshev_scaling_parameters(energies.min(), energies.max())
        else:
            scaling = kwargs.pop("scaling")
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
        if "scaling" not in kwargs:
            diag = expression.diagonal()
            emin = np.min(diag)
            emax = np.max(diag)
            emin -= 0.5 * (emax - emin)
            emax += 0.5 * (emax - emin)
            scaling = util.get_chebyshev_scaling_parameters(emin, emax)
        else:
            scaling = kwargs.pop("scaling")
        moments = expression.build_gf_chebyshev_moments(max_cycle + 1, scaling=scaling)
        return cls(moments, kwargs.pop("grid"), scaling, max_cycle=max_cycle, **kwargs)

    def kernel(self, iteration: int | None = None) -> Dynamic[RealFrequencyGrid]:
        """Run the solver.

        Args:
            iteration: The iteration number.

        Returns:
            The Green's function on the real frequency grid.
        """
        if iteration is None:
            iteration = self.max_cycle

        # Get the printing helpers
        progress = printing.IterationsPrinter(iteration + 1, description="Polynomial order")
        progress.start()

        # Get the moments -- allow input to already be traced or diagonal
        if self.reduction == Reduction.NONE:
            moments = self.moments[: iteration + 1].astype(complex)
        elif self.reduction == Reduction.DIAG:
            moments = util.as_diagonal(self.moments[: iteration + 1], 1).astype(complex)
        elif self.reduction == Reduction.TRACE:
            moments = util.as_trace(self.moments[: iteration + 1], 1).astype(complex)
        else:
            self.reduction.raise_invalid_representation()
        if (moments.ndim - 1) != self.reduction.ndim:
            raise ValueError(
                f"moments must be {self.reduction.ndim + 1}D for reduction {self.reduction}, got "
                f"{moments.ndim}D."
            )

        # Scale the grid
        scaled_grid = (self.grid - self.scaling[1]) / self.scaling[0]
        scaled_eta = self.grid.eta / self.scaling[0]
        shifted_grid = scaled_grid + 1j * scaled_eta

        # Initialise factors
        numerator = shifted_grid - 1j * np.sqrt(1 - shifted_grid**2)
        denominator = np.sqrt(1 - shifted_grid**2)
        kernel = 1.0 / denominator

        # Iteratively compute the Green's function
        greens_function = util.einsum("z,...->z...", kernel, moments[0])
        for cycle in range(1, iteration + 1):
            progress.update(cycle)
            kernel *= numerator
            greens_function += util.einsum("z,...->z...", kernel, moments[cycle]) * 2

        # Apply factors
        greens_function /= self.scaling[0]
        greens_function *= -1.0j
        if self.ordering == Ordering.ADVANCED:
            greens_function = greens_function.conj()

        # Post-process the Green's function component
        # TODO: Can we do this earlier to avoid computing unnecessary components?
        if self.component == Component.REAL:
            greens_function = greens_function.real
        elif self.component == Component.IMAG:
            greens_function = greens_function.imag

        progress.stop()

        return Dynamic(
            self.grid,
            greens_function,
            reduction=self.reduction,
            component=self.component,
            hermitian=np.allclose(self.moments, self.moments.transpose(0, 2, 1).conj()),
        )

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
