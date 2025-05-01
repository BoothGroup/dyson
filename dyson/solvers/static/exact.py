"""Exact diagonalisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

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
        bra: The bra state vector mapping the supermatrix to the physical space.
        ket: The ket state vector mapping the supermatrix to the physical space.
    """

    def __init__(
        self,
        matrix: Array,
        bra: Array,
        ket: Array | None = None,
        hermitian: bool = True,
    ):
        """Initialise the solver.

        Args:
            matrix: The self-energy supermatrix.
            bra: The bra state vector mapping the supermatrix to the physical space.
            ket: The ket state vector mapping the supermatrix to the physical space. If `None`, use
                the same vectors as `bra`.
            hermitian: Whether the matrix is hermitian.
        """
        self._matrix = matrix
        self._bra = bra
        self._ket = ket
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
        size = self_energy.nphys + self_energy.naux
        bra = np.array([util.unit_vector(size, i) for i in range(self_energy.nphys)])
        return cls(
            self_energy.matrix(static),
            bra,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    def kernel(self) -> None:
        """Run the solver."""
        # Get the raw eigenvalues and eigenvectors
        if self.hermitian:
            eigvals, eigvecs = util.eig(self.matrix, hermitian=self.hermitian)
        else:
            eigvals, (left, right) = util.eig_lr(self.matrix, hermitian=self.hermitian)
            eigvecs = np.array([left, right])

        # Get the full map onto physical + auxiliary and rotate the eigenvectors
        vectors = util.null_space_basis(self.bra, ket=self.ket)
        if self.ket is None or self.hermitian:
            rotation = np.concatenate([self.bra, vectors[0]], axis=0)
            eigvecs = rotation @ eigvecs
        else:
            rotation = (
                np.concatenate([self.ket, vectors[0]], axis=0),
                np.concatenate([self.bra, vectors[1]], axis=0),
            )
            eigvecs = np.array([rotation[0] @ eigvecs[0], rotation[1] @ eigvecs[1]])

        # Store the eigenvalues and eigenvectors
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    @property
    def matrix(self) -> Array:
        """Get the self-energy supermatrix."""
        return self._matrix

    @property
    def bra(self) -> Array:
        """Get the bra state vector mapping the supermatrix to the physical space."""
        return self._bra

    @property
    def ket(self) -> Array:
        """Get the ket state vector mapping the supermatrix to the physical space."""
        if self._ket is None:
            return self._bra
        return self._ket

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.bra.shape[0]
