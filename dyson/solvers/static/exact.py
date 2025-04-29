"""Exact diagonalisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np, util
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver

if TYPE_CHECKING:
    from typing import Any

    from dyson.typing import Array


class Exact(StaticSolver):
    """Exact diagonalisation of the supermatrix form of the self-energy.

    Args:
        matrix: The self-energy supermatrix.
        nphys: Number of physical degrees of freedom.
    """

    def __init__(
        self,
        matrix: Array,
        nphys: int,
        hermitian: bool = True,
    ):
        """Initialise the solver.

        Args:
            matrix: The self-energy supermatrix.
            nphys: Number of physical degrees of freedom.
            hermitian: Whether the matrix is hermitian.
        """
        self._matrix = matrix
        self._nphys = nphys
        self.hermitian = hermitian

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> Exact:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        return cls(
            self_energy.matrix(static), self_energy.nphys, hermitian=self_energy.hermitian, **kwargs
        )

    def kernel(self) -> None:
        """Run the solver."""
        if self.hermitian:
            self.eigvals, self.eigvecs = util.eig(self.matrix, hermitian=self.hermitian)
        else:
            self.eigvals, self.eigvecs = util.eig_biorth(self.matrix, hermitian=self.hermitian)

    @property
    def matrix(self) -> Array:
        """Get the self-energy supermatrix."""
        return self._matrix

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys
