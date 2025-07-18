"""Container for an spectral representation (eigenvalues and eigenvectors) of a matrix."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.representations.lehmann import Lehmann
from dyson.representations.representation import BaseRepresentation

if TYPE_CHECKING:
    from dyson.typing import Array


class Spectral(BaseRepresentation):
    r"""Spectral representation matrix with a known number of physical degrees of freedom.

    The eigendecomposition (spectral decomposition) of a matrix consists of the eigenvalues
    :math:`\lambda_k` and eigenvectors :math:`v_{pk}` that represent the matrix as

    .. math::
        \sum_{k} \lambda_k v_{pk} u_{qk}^*,

    where the eigenvectors have right-handed components :math:`v` and left-handed components
    :math:`u`.

    Note that the order of eigenvectors is `(left, right)`, whilst they act in the order
    `(right, left)` in the above equation. The naming convention is chosen to be consistent with the
    eigenvalue decomposition, where :math:`v` may be an eigenvector acting on the right of a matrix,
    and :math:`u` is an eigenvector acting on the left of a matrix.
    """

    def __init__(
        self,
        eigvals: Array,
        eigvecs: Array,
        nphys: int,
        sort: bool = False,
        chempot: float | None = None,
    ):
        """Initialise the object.

        Args:
            eigvals: Eigenvalues of the matrix.
            eigvecs: Eigenvectors of the matrix.
            nphys: Number of physical degrees of freedom.
            sort: Sort the eigenfunctions by eigenvalue.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.
        """
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._nphys = nphys
        self.chempot = chempot
        if sort:
            self.sort_()
        if not self.hermitian:
            if eigvecs.ndim != 3:
                raise ValueError(
                    f"Couplings must be 3D for a non-Hermitian system, but got {eigvecs.ndim}D."
                )
            if eigvecs.shape[0] != 2:
                raise ValueError(
                    f"Couplings must have shape (2, nphys, naux) for a non-Hermitian system, "
                    f"but got {eigvecs.shape}."
                )

    @classmethod
    def from_matrix(
        cls, matrix: Array, nphys: int, hermitian: bool = True, chempot: float | None = None
    ) -> Spectral:
        """Create a spectrum from a matrix.

        Args:
            matrix: Matrix to diagonalise.
            nphys: Number of physical degrees of freedom.
            hermitian: Whether the matrix is Hermitian.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.

        Returns:
            Spectrum object.
        """
        if hermitian:
            eigvals, eigvecs = util.eig(matrix, hermitian=True)
        else:
            eigvals, (left, right) = util.eig_lr(matrix, hermitian=False)
            eigvecs = np.array([left, right])
        return cls(eigvals, eigvecs, nphys, chempot=chempot)

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
    ) -> Spectral:
        """Create a spectrum from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.

        Returns:
            Spectrum object.
        """
        return cls(
            *self_energy.diagonalise_matrix(static, overlap=overlap),
            self_energy.nphys,
            chempot=self_energy.chempot,
        )

    def sort_(self) -> None:
        """Sort the eigenfunctions by eigenvalue.

        Note:
            The object is sorted in place.
        """
        idx = np.argsort(self.eigvals)
        self._eigvals = self.eigvals[idx]
        self._eigvecs = self.eigvecs[..., idx]

    def _get_matrix_block(self, slices: tuple[slice, slice]) -> Array:
        """Get a block of the matrix.

        Args:
            slices: Slices to select.

        Returns:
            Block of the matrix.
        """
        left, right = util.unpack_vectors(self.eigvecs)
        return util.einsum("pk,qk,k->pq", right[slices[0]], left[slices[1]].conj(), self.eigvals)

    def get_static_self_energy(self) -> Array:
        """Get the static part of the self-energy.

        Returns:
            Static self-energy.
        """
        return self._get_matrix_block((slice(self.nphys), slice(self.nphys)))

    def get_auxiliaries(self) -> tuple[Array, Array]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Returns:
            Auxiliary energies and couplings.
        """
        phys = slice(None, self.nphys)
        aux = slice(self.nphys, None)

        # Project back to the auxiliary subspace
        subspace = self._get_matrix_block((aux, aux))

        # If there are no auxiliaries, return here
        if subspace.size == 0:
            energies = np.empty((0))
            couplings = np.empty((self.nphys, 0) if self.hermitian else (2, self.nphys, 0))
            return energies, couplings

        # Diagonalise the subspace to get the energies and basis for the couplings
        # TODO: check if already diagonal
        energies, rotation = util.eig_lr(subspace, hermitian=self.hermitian)

        # Project back to the couplings
        couplings_right = self._get_matrix_block((phys, aux))
        if not self.hermitian:
            couplings_left = self._get_matrix_block((aux, phys)).T.conj()

        # Rotate the couplings to the auxiliary basis
        if self.hermitian:
            couplings = couplings_right @ rotation[0]
        else:
            couplings = np.array([couplings_left @ rotation[0], couplings_right @ rotation[1]])

        return energies, couplings

    def get_dyson_orbitals(self) -> tuple[Array, Array]:
        """Get the Dyson orbitals.

        Returns:
            Dyson orbitals.
        """
        return self.eigvals, self.eigvecs[..., : self.nphys, :]

    def get_overlap(self) -> Array:
        """Get the overlap matrix in the physical space.

        Returns:
            Overlap matrix.
        """
        _, orbitals = self.get_dyson_orbitals()
        left, right = util.unpack_vectors(orbitals)
        return util.einsum("pk,qk->pq", right, left.conj())

    def get_self_energy(self, chempot: float | None = None) -> Lehmann:
        """Get the Lehmann representation of the self-energy.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the self-energy.
        """
        if chempot is None:
            chempot = self.chempot
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_auxiliaries(), chempot=chempot)

    def get_greens_function(self, chempot: float | None = None) -> Lehmann:
        """Get the Lehmann representation of the Green's function.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the Green's function.
        """
        if chempot is None:
            chempot = self.chempot
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_dyson_orbitals(), chempot=chempot)

    def combine(self, *args: Spectral, chempot: float | None = None) -> Spectral:
        """Combine multiple spectral representations.

        Args:
            args: Spectral representations to combine.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.

        Returns:
            Combined spectral representation.
        """
        # TODO: just concatenate the eigenvectors...?
        args = (self, *args)
        if len(set(arg.nphys for arg in args)) != 1:
            raise ValueError(
                "All Spectral objects must have the same number of physical degrees of freedom."
            )
        nphys = args[0].nphys

        # Sum the overlap and static self-energy matrices -- double counting is not an issue
        # with shared static parts because the overlap matrix accounts for the separation
        static = sum([arg.get_static_self_energy() for arg in args], np.zeros((nphys, nphys)))
        overlap = sum([arg.get_overlap() for arg in args], np.zeros((nphys, nphys)))

        # Check the chemical potentials
        if chempot is None:
            if any(arg.chempot is not None for arg in args):
                chempots = [arg.chempot for arg in args if arg.chempot is not None]
                if not all(np.isclose(chempots[0], part) for part in chempots[1:]):
                    raise ValueError(
                        "If not chempot is passed to combine, all chemical potentials must be "
                        "equal in the inputs."
                    )
                chempot = chempots[0]

        # Get the auxiliaries
        energies = np.zeros((0))
        left = np.zeros((nphys, 0))
        right = np.zeros((nphys, 0))
        for arg in args:
            energies_i, couplings_i = arg.get_auxiliaries()
            energies = np.concatenate([energies, energies_i])
            if arg.hermitian:
                left = np.concatenate([left, couplings_i], axis=1)
            else:
                left_i, right_i = util.unpack_vectors(couplings_i)
                left = np.concatenate([left, left_i], axis=1)
                right = np.concatenate([right, right_i], axis=1)
        couplings = np.array([left, right]) if not args[0].hermitian else left

        # Solve the eigenvalue problem
        self_energy = Lehmann(energies, couplings)
        result = Spectral(
            *self_energy.diagonalise_matrix(static, overlap=overlap), nphys, chempot=chempot
        )

        return result

    @cached_property
    def overlap(self) -> Array:
        """Get the overlap matrix (the zeroth moment of the Green's function)."""
        orbitals = self.get_dyson_orbitals()[1]
        left, right = util.unpack_vectors(orbitals)
        return util.einsum("pk,qk->pq", right, left.conj())

    @property
    def eigvals(self) -> Array:
        """Get the eigenvalues."""
        return self._eigvals

    @property
    def eigvecs(self) -> Array:
        """Get the eigenvectors."""
        return self._eigvecs

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys

    @property
    def neig(self) -> int:
        """Get the number of eigenvalues."""
        return self.eigvals.shape[0]

    @property
    def hermitian(self) -> bool:
        """Check if the spectrum is Hermitian."""
        return self.eigvecs.ndim == 2

    def __eq__(self, other: object) -> bool:
        """Check if two Lehmann representations are equal."""
        if not isinstance(other, Spectral):
            return NotImplemented
        if other.nphys != self.nphys:
            return False
        if other.neig != self.neig:
            return False
        if other.hermitian != self.hermitian:
            return False
        if other.chempot != self.chempot:
            return False
        return np.allclose(other.eigvals, self.eigvals) and np.allclose(other.eigvecs, self.eigvecs)

    def __hash__(self) -> int:
        """Hash the object."""
        return hash((tuple(self.eigvals), tuple(self.eigvecs.flatten()), self.nphys, self.chempot))
