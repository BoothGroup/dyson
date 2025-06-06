"""Exact diagonalisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from typing import Any

    from dyson.expressions.expression import BaseExpression
    from dyson.typing import Array


class Exact(StaticSolver):
    """Exact diagonalisation of the supermatrix form of the self-energy.

    Args:
        matrix: The self-energy supermatrix.
        bra: The bra state vector mapping the supermatrix to the physical space.
        ket: The ket state vector mapping the supermatrix to the physical space.
    """

    hermitian: bool = True
    _options: set[str] = {"hermitian"}

    def __init__(  # noqa: D417
        self,
        matrix: Array,
        bra: Array,
        ket: Array | None = None,
        **kwargs: Any,
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
        for key, val in kwargs.items():
            if key not in self._options:
                raise ValueError(f"Unknown option for {self.__class__.__name__}: {key}")
            setattr(self, key, val)

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> Exact:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        size = self_energy.nphys + self_energy.naux
        bra = ket = np.array([util.unit_vector(size, i) for i in range(self_energy.nphys)])
        if overlap is not None:
            hermitian = self_energy.hermitian
            orth = util.matrix_power(overlap, 0.5, hermitian=hermitian)[0]
            unorth = util.matrix_power(overlap, -0.5, hermitian=hermitian)[0]
            bra = util.rotate_subspace(bra, orth.T.conj())
            ket = util.rotate_subspace(ket, orth) if not hermitian else bra
            static = unorth @ static @ unorth
            self_energy = self_energy.rotate_couplings(
                unorth if hermitian else (unorth, unorth.T.conj())
            )
        return cls(
            self_energy.matrix(static),
            bra,
            ket,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> Exact:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        matrix = expression.build_matrix()
        bra = np.array([expression.get_state_bra(i) for i in range(expression.nphys)])
        ket = (
            np.array([expression.get_state_ket(i) for i in range(expression.nphys)])
            if not expression.hermitian
            else None
        )
        return cls(matrix, bra, ket, hermitian=expression.hermitian, **kwargs)

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Get the raw eigenvalues and eigenvectors
        if self.hermitian:
            eigvals, eigvecs = util.eig(self.matrix, hermitian=self.hermitian)
        else:
            eigvals, (left, right) = util.eig_lr(self.matrix, hermitian=self.hermitian)
            eigvecs = np.array([left, right])

        # Get the full map onto physical + auxiliary and rotate the eigenvectors
        vectors = util.null_space_basis(self.bra, ket=self.ket if not self.hermitian else None)
        if self.ket is None or self.hermitian:
            rotation = np.concatenate([self.bra, vectors[0]], axis=0)
            eigvecs = rotation @ eigvecs
        else:
            # Ensure biorthonormality of auxiliary vectors
            overlap = vectors[1].T.conj() @ vectors[0]
            overlap -= self.ket.T.conj() @ self.bra
            vectors = (
                vectors[0],
                vectors[1] @ util.matrix_power(overlap, -1, hermitian=False)[0],
            )
            rotation = (
                np.concatenate([self.bra, vectors[1]], axis=0),
                np.concatenate([self.ket, vectors[0]], axis=0),
            )
            eigvecs = np.array([rotation[0] @ eigvecs[0], rotation[1] @ eigvecs[1]])

        # Store the result
        self.result = Spectral(eigvals, eigvecs, self.nphys)

        return self.result

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
