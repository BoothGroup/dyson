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
            eigvals, (left, right) = util.eig_biorth(self.matrix, hermitian=self.hermitian)
            eigvecs = np.array([left, right])

        # Find the null space of ⟨bra|ket⟩ to get the map onto the auxiliary space
        vectors = util.null_space_basis(self.bra, ket=self.ket)

        # Get the full map onto physical + auxiliary and rotated the eigenvectors
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


class BlockExact(StaticSolver):
    """Exact diagonalisation of blocks of the supermatrix form of the self-energy.

    Args:
        matrices: The self-energy supermatrices.
        bras: The bra state vector mapping the supermatrices to the physical space.
        kets: The ket state vector mapping the supermatrices to the physical space.

    Note:
        The resulting Green's function is orthonormalised such that the zeroth moment is identity.
        This may not be the desired behaviour in cases where your blocks do not span the full space.
    """

    Solver = Exact

    def __init__(
        self,
        matrices: list[Array],
        bras: list[Array],
        kets: list[Array] | None = None,
        hermitian: bool = True,
    ):
        """Initialise the solver.

        Args:
            matrices: The self-energy supermatrices.
            bras: The bra state vector mapping the supermatrices to the physical space.
            kets: The ket state vector mapping the supermatrices to the physical space. If `None`,
                use the same vectors as `bra`.
            hermitian: Whether the matrix is hermitian.
        """
        self.solvers = [
            self.Solver(matrix, bra, ket, hermitian=hermitian)
            for matrix, bra, ket in zip(matrices, bras, kets or bras)
        ]
        self.hermitian = hermitian

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> BlockExact:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            For the block-wise solver, this function separates the self-energy into occupied and
            virtual parts.
        """
        self_energy_parts = (self_energy.occupied(), self_energy.virtual())
        bra = np.array([util.unit_vector(self_energy.nphys, i) for i in range(self_energy.nphys)])
        return cls(
            [part.matrix(static) for part in self_energy_parts],
            [bra for _ in self_energy_parts],
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    def kernel(self) -> None:
        """Run the solver."""
        # Run the solvers
        for solver in self.solvers:
            solver.kernel()

        # Get the eigenvalues and eigenvectors
        eigvals_list = []
        left_list = []
        right_list = []
        for solver in self.solvers:
            eigvals, eigvecs = solver.get_eigenfunctions()
            eigvals_list.append(eigvals)
            left, right = util.unpack_vectors(eigvecs)
            left_list.append(left)
            right_list.append(right)

        # Combine the eigenvalues and eigenvectors
        eigvals = np.concatenate(eigvals_list)
        left = util.concatenate_paired_vectors(left_list, self.nphys)
        if not self.hermitian:
            right = util.concatenate_paired_vectors(right_list, self.nphys)

        # Biorthogonalise the eigenvectors
        if self.hermitian:
            eigvecs = util.orthonormalise(left, transpose=True)
        else:
            eigvecs = np.array(util.biorthonormalise(left, right, transpose=True))

        # Store the eigenvalues and eigenvectors
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        if not len(set(solver.nphys for solver in self.solvers)) == 1:
            raise ValueError(
                "All solvers must have the same number of physical degrees of freedom."
            )
        return self.solvers[0].nphys
