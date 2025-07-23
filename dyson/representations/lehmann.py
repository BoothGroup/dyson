"""Container for a Lehmann representation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import scipy.linalg

from dyson import numpy as np
from dyson import util
from dyson.representations.enums import Reduction
from dyson.representations.representation import BaseRepresentation
from dyson.typing import Array

if TYPE_CHECKING:
    from typing import Iterable, Iterator

    import pyscf.agf2.aux


@contextmanager
def shift_energies(lehmann: Lehmann, shift: float) -> Iterator[None]:
    """Shift the energies of a Lehmann representation using a context manager.

    Args:
        lehmann: The Lehmann representation to shift.
        shift: The amount to shift the energies by.

    Yields:
        None
    """
    original_energies = lehmann.energies
    lehmann._energies = original_energies + shift  # pylint: disable=protected-access
    try:
        yield
    finally:
        lehmann._energies = original_energies  # pylint: disable=protected-access


class Lehmann(BaseRepresentation):
    r"""Lehman representation.

    The Lehmann representation is a set of poles :math:`\epsilon_k` and couplings :math:`v_{pk}`
    that can be downfolded into a frequency-dependent function as

    .. math::
        \sum_{k} \frac{v_{pk} u_{qk}^*}{\omega - \epsilon_k},

    where the couplings are between the poles :math:`k` and the physical space :math:`p` and
    :math:`q`, and may be non-Hermitian. The couplings :math:`v` are right-handed vectors, and
    :math:`u` are left-handed vectors.

    Note that the order of the couplings is `(left, right)`, whilst they act in the order
    `(right, left)` in the numerator. The naming convention is chosen to be consistent with the
    eigenvalue decomposition, where :math:`v` may be an eigenvector acting on the right of a matrix,
    and :math:`u` is an eigenvector acting on the left of a matrix.
    """

    def __init__(
        self,
        energies: Array,
        couplings: Array,
        chempot: float = 0.0,
        sort: bool = True,
    ):
        """Initialise the object.

        Args:
            energies: Energies of the poles.
            couplings: Couplings of the poles to a physical space. For a non-Hermitian system, they
                should be have three dimensions, with the first dimension indexing `(left, right)`.
            chempot: Chemical potential.
            sort: Sort the poles by energy.
        """
        self._energies = energies
        self._couplings = couplings
        self._chempot = chempot
        if sort:
            self.sort_()
        if not self.hermitian:
            if couplings.ndim != 3:
                raise ValueError(
                    f"Couplings must be 3D for a non-Hermitian system, but got {couplings.ndim}D."
                )
            if couplings.shape[0] != 2:
                raise ValueError(
                    f"Couplings must have shape (2, nphys, naux) for a non-Hermitian system, "
                    f"but got {couplings.shape}."
                )

    @classmethod
    def from_pyscf(cls, auxspace: pyscf.agf2.aux.AuxSpace | Lehmann) -> Lehmann:
        """Construct a Lehmann representation from a PySCF auxiliary space.

        Args:
            auxspace: The auxiliary space.

        Returns:
            A Lehmann representation.
        """
        if isinstance(auxspace, Lehmann):
            return auxspace
        return cls(
            energies=auxspace.energy,
            couplings=auxspace.coupling,
            chempot=auxspace.chempot,
        )

    @classmethod
    def from_empty(cls, nphys: int) -> Lehmann:
        """Construct an empty Lehmann representation.

        Args:
            nphys: The number of physical degrees of freedom.

        Returns:
            An empty Lehmann representation.
        """
        return cls(
            energies=np.zeros((0,)),
            couplings=np.zeros((nphys, 0)),
            chempot=0.0,
            sort=False,
        )

    def sort_(self) -> None:
        """Sort the poles by energy.

        Note:
            The object is sorted in place.
        """
        idx = np.argsort(self.energies)
        self._energies = self.energies[idx]
        self._couplings = self.couplings[..., idx]

    @property
    def energies(self) -> Array:
        """Get the energies."""
        return self._energies

    @property
    def couplings(self) -> Array:
        """Get the couplings."""
        return self._couplings

    @property
    def chempot(self) -> float:
        """Get the chemical potential."""
        return self._chempot

    @property
    def hermitian(self) -> bool:
        """Get a boolean indicating if the system is Hermitian."""
        return self.couplings.ndim == 2

    def unpack_couplings(self) -> tuple[Array, Array]:
        """Unpack the couplings.

        Returns:
            A tuple of left and right couplings.
        """
        if self.hermitian:
            return cast(tuple[Array, Array], (self.couplings, self.couplings))
        return cast(tuple[Array, Array], (self.couplings[0], self.couplings[1]))

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.unpack_couplings()[0].shape[0]

    @property
    def naux(self) -> int:
        """Get the number of auxiliary degrees of freedom."""
        return self.unpack_couplings()[0].shape[1]

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the couplings."""
        return np.result_type(self.energies, self.couplings)

    def __repr__(self) -> str:
        """Return a string representation of the Lehmann representation."""
        return f"Lehmann(nphys={self.nphys}, naux={self.naux}, chempot={self.chempot})"

    def mask(self, mask: Array | slice, deep: bool = True) -> Lehmann:
        """Return a part of the Lehmann representation according to a mask.

        Args:
            mask: The mask to apply.
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new Lehmann representation including only the masked states.
        """
        # Mask the energies and couplings
        energies = self.energies[mask]
        couplings = self.couplings[..., mask]

        # Copy the couplings if requested
        if deep:
            energies = energies.copy()
            couplings = couplings.copy()

        return self.__class__(energies, couplings, chempot=self.chempot, sort=False)

    def physical(self, weight: float = 0.1, deep: bool = True) -> Lehmann:
        """Return the physical (large weight) part of the Lehmann representation.

        Args:
            weight: The weight to use for the physical part.
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new Lehmann representation including only the physical part.
        """
        return self.mask(self.weights() > weight, deep=deep)

    def occupied(self, deep: bool = True) -> Lehmann:
        """Return the occupied part of the Lehmann representation.

        Args:
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new Lehmann representation including only the occupied part.
        """
        return self.mask(self.energies < self.chempot, deep=deep)

    def virtual(self, deep: bool = True) -> Lehmann:
        """Return the virtual part of the Lehmann representation.

        Args:
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new Lehmann representation including only the virtual part.
        """
        return self.mask(self.energies >= self.chempot, deep=deep)

    def copy(self, chempot: float | None = None, deep: bool = True) -> Lehmann:
        """Return a copy of the Lehmann representation.

        Args:
            chempot: The chemical potential to use for the copy. If `None`, the original
                chemical potential is used.
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new Lehmann representation.
        """
        energies = self.energies
        couplings = self.couplings
        if chempot is None:
            chempot = self.chempot

        # Copy the couplings if requested
        if deep:
            energies = energies.copy()
            couplings = couplings.copy()

        return self.__class__(energies, couplings, chempot=self.chempot, sort=False)

    def rotate_couplings(self, rotation: Array | tuple[Array, Array]) -> Lehmann:
        r"""Rotate the couplings and return a new Lehmann representation.

        For rotation matrix :math:`R`, the couplings are rotated as

        .. math::
            \tilde{\mathbf{v}} = R^\dagger \mathbf{v}, \quad
            \tilde{\mathbf{u}} = R^\dagger \mathbf{u},

        where :math:`v` are the right couplings and :math:`u` are the left couplings.

        Args:
            rotation: The rotation matrix to apply to the couplings. If the matrix has three
                dimensions, the first dimension is used to rotate the left couplings, and the
                second dimension is used to rotate the right couplings.

        Returns:
            A new Lehmann representation with the couplings rotated into the new basis.
        """
        if not isinstance(rotation, tuple) and rotation.ndim == 2:
            couplings = util.einsum("...pk,pq->...qk", self.couplings, rotation.conj())
        else:
            left, right = self.unpack_couplings()
            if isinstance(rotation, tuple) or rotation.ndim == 3:
                rot_left, rot_right = rotation
            else:
                rot_left = rot_right = rotation
            couplings = np.array(
                [
                    rot_left.T.conj() @ left,
                    rot_right.T.conj() @ right,
                ],
            )
        return self.__class__(
            self.energies,
            couplings,
            chempot=self.chempot,
            sort=False,
        )

    # Methods to calculate moments:

    def moments(self, order: int | Iterable[int], reduction: Reduction = Reduction.NONE) -> Array:
        r"""Calculate the moment(s) of the Lehmann representation.

        The moments are defined as

        .. math::
            T_{pq}^{n} = \sum_{k} v_{pk} u_{qk}^* \epsilon_k^n,

        where :math:`T_{pq}^{n}` is the moment of order :math:`n` in the physical space. In terms of
        the frequency-dependency, the moments can be written as the integral

        .. math::
            T_{pq}^{n} = \int_{-\infty}^{\infty} d\omega \, \left[ \sum_{k}
            \frac{v_{pk} u_{qk}^*}{\omega - \epsilon_k} \right] \, \omega^n,

        where the integral is over the entire real line for central moments.

        Args:
            order: The order(s) of the moment(s).
            reduction: The reduction to apply to the moments.

        Returns:
            The moment(s) of the Lehmann representation.
        """
        squeeze = False
        if isinstance(order, int):
            order = [order]
            squeeze = True
        orders = np.asarray(order)

        # Get the subscript depending on the reduction
        if Reduction(reduction) == Reduction.NONE:
            subscript = "pk,qk,nk->npq"
        elif Reduction(reduction) == Reduction.DIAG:
            subscript = "pk,pk,nk->np"
        elif Reduction(reduction) == Reduction.TRACE:
            subscript = "pk,pk,nk->n"
        else:
            Reduction(reduction).raise_invalid_representation()

        # Contract the moments
        left, right = self.unpack_couplings()
        moments = util.einsum(
            subscript,
            right,
            left.conj(),
            self.energies[None] ** orders[:, None],
        )
        if squeeze:
            moments = moments[0]

        return moments

    moment = moments

    def chebyshev_moments(
        self,
        order: int | Iterable[int],
        scaling: tuple[float, float] | None = None,
        scale_couplings: bool = False,
    ) -> Array:
        r"""Calculate the Chebyshev polynomial moment(s) of the Lehmann representation.

        The Chebyshev moments are defined as

        .. math::
            T_{pq}^{n} = \sum_{k} v_{pk} u_{qk}^* P_n(\epsilon_k),

        where :math:`P_n(x)` is the Chebyshev polynomial of order :math:`n`.

        Args:
            order: The order(s) of the moment(s).
            scaling: Scaling factors to ensure the energy scale of the Lehmann representation is
                in `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`. If
                `None`, the default scaling is computed as
                `(max(energies) - min(energies)) / (2.0 - 1e-3)` and
                `(max(energies) + min(energies)) / 2.0`, respectively.
            scale_couplings: Scale the couplings as well as the energy spectrum. This is generally
                necessary for Chebyshev moments of a self-energy, but not for a Green's function.

        Returns:
            The Chebyshev polynomial moment(s) of the Lehmann representation.
        """
        if scaling is None:
            scaling = util.get_chebyshev_scaling_parameters(
                self.energies.min(), self.energies.max()
            )
        squeeze = False
        if isinstance(order, int):
            order = [order]
            squeeze = True
        max_order = max(order)
        orders = set(order)

        # Scale the spectrum
        left, right = self.unpack_couplings()
        energies = (self.energies - scaling[1]) / scaling[0]
        if scale_couplings:
            left = left / scaling[0]
            right = right / scaling[0]

        # Calculate the Chebyshev moments
        moments = np.zeros((len(orders), self.nphys, self.nphys), dtype=self.dtype)
        vecs = (right, right * energies[None])
        idx = 0
        if 0 in orders:
            moments[idx] = vecs[0] @ left.T.conj()
            idx += 1
        if 1 in orders:
            moments[idx] = vecs[1] @ left.T.conj()
            idx += 1
        for i in range(2, max_order + 1):
            vecs = (vecs[1], 2 * energies * vecs[1] - vecs[0])
            if i in orders:
                moments[idx] = vecs[1] @ left.T.conj()
                idx += 1
        if squeeze:
            moments = moments[0]

        return moments

    chebyshev_moment = chebyshev_moments

    # Methods associated with the supermatrix:

    def matrix(self, physical: Array, chempot: bool | float = False) -> Array:
        r"""Build the dense supermatrix form of the Lehmann representation.

        The supermatrix is defined as

        .. math::
            \begin{bmatrix}
                \mathbf{f} & \mathbf{v} \\
                \mathbf{u}^\dagger & \boldsymbol{\epsilon} \mathbf{I}
            \end{bmatrix},

        where :math:`\mathbf{f}` is the physical space part of the supermatrix, provided as an
        argument.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.

        Returns:
            The supermatrix form of the Lehmann representation.
        """
        energies = self.energies
        left, right = self.unpack_couplings()
        if chempot:
            if chempot is True:
                chempot = self.chempot
            energies -= chempot

        # If there are no auxiliary states, return the physical matrix
        if self.naux == 0:
            return physical

        # Build the supermatrix
        matrix = np.block([[physical, right], [left.T.conj(), np.diag(energies)]])

        return matrix

    def diagonal(self, physical: Array, chempot: bool | float = False) -> Array:
        r"""Build the diagonal supermatrix form of the Lehmann representation.

        The diagonal supermatrix is defined as

        .. math::
            \begin{bmatrix} \mathrm{diag}(\mathbf{f}) & \boldsymbol{\epsilon} \end{bmatrix},

        where :math:`\mathbf{f}` is the physical space part of the supermatrix, provided as an
        argument.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.

        Returns:
            The diagonal supermatrix form of the Lehmann representation.
        """
        energies = self.energies
        if chempot:
            if chempot is True:
                chempot = self.chempot
            energies -= chempot

        # Build the supermatrix diagonal
        diagonal = np.concatenate((np.diag(physical), energies))

        return diagonal

    def matvec(self, physical: Array, vector: Array, chempot: bool | float = False) -> Array:
        r"""Apply the supermatrix to a vector.

        The matrix-vector product is defined as

        .. math::
            \begin{bmatrix}
                \mathbf{x}_\mathrm{phys} \\
                \mathbf{x}_\mathrm{aux}
            \end{bmatrix}
            =
            \begin{bmatrix}
                \mathbf{f} & \mathbf{v} \\
                \mathbf{u}^\dagger & \mathbf{\epsilon} \mathbf{I}
            \end{bmatrix}
            \begin{bmatrix}
                \mathbf{r}_\mathrm{phys} \\
                \mathbf{r}_\mathrm{aux}
            \end{bmatrix},

        where :math:`\mathbf{f}` is the physical space part of the supermatrix, and the input
        vector :math:`\mathbf{r}` is spans both the physical and auxiliary spaces.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            vector: The vector to apply the supermatrix to.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.

        Returns:
            The result of applying the supermatrix to the vector.
        """
        left, right = self.unpack_couplings()
        energies = self.energies
        if chempot:
            if chempot is True:
                chempot = self.chempot
            energies -= chempot
        if vector.shape[0] != (self.nphys + self.naux):
            raise ValueError(
                f"Vector shape {vector.shape} does not match supermatrix shape "
                f"{(self.nphys + self.naux, self.nphys + self.naux)}"
            )

        # Contract the supermatrix
        vector_phys, vector_aux = np.split(vector, [self.nphys])
        result_phys = util.einsum("pq,q...->p...", physical, vector_phys)
        result_phys += util.einsum("pk,k...->p...", right, vector_aux)
        result_aux = util.einsum("pk,p...->k...", left.conj(), vector_phys)
        result_aux += util.einsum("k,k...->k...", energies, vector_aux)
        result = np.concatenate((result_phys, result_aux), axis=0)

        return result

    def diagonalise_matrix(
        self, physical: Array, chempot: bool | float = False, overlap: Array | None = None
    ) -> tuple[Array, Array]:
        r"""Diagonalise the supermatrix.

        The eigenvalue problem is defined as

        .. math::
            \begin{bmatrix}
                \mathbf{f} & \mathbf{v} \\
                \mathbf{u}^\dagger & \mathbf{\epsilon} \mathbf{1}
            \end{bmatrix}
            \begin{bmatrix}
                \mathbf{x}_\mathrm{phys} \\
                \mathbf{x}_\mathrm{aux}
            \end{bmatrix}
            =
            E
            \begin{bmatrix}
                \mathbf{x}_\mathrm{phys} \\
                \mathbf{x}_\mathrm{aux}
            \end{bmatrix},

        where :math:`\mathbf{f}` is the physical space part of the supermatrix, and the eigenvectors
        :math:`\mathbf{x}` span both the physical and auxiliary spaces.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.
            overlap: The overlap matrix to use for the physical space part of the supermatrix. If
                `None`, the identity matrix is used.

        Returns:
            The eigenvalues and eigenvectors of the supermatrix.

        Note:
            If a non-identity overlap matrix is provided, this is equivalent to performing a
            generalised eigenvalue decomposition of the supermatrix, with the overlap in the
            auxiliary space assumed to be the identity.
        """
        # Orthogonalise the physical space if overlap is provided
        lehmann = self
        if overlap is not None:
            orth = util.matrix_power(overlap, -0.5, hermitian=False)[0]
            unorth = util.matrix_power(overlap, 0.5, hermitian=False)[0]
            physical = orth @ physical @ orth
            lehmann = lehmann.rotate_couplings(orth if self.hermitian else (orth, orth.T.conj()))

        # Get the chemical potential
        if chempot is True:
            chempot = self.chempot
        else:
            chempot = float(chempot)

        # Diagonalise the supermatrix
        matrix = lehmann.matrix(physical, chempot=chempot)
        if self.hermitian:
            eigvals, eigvecs = util.eig(matrix, hermitian=True)
            if overlap is not None:
                eigvecs = util.rotate_subspace(eigvecs, unorth.T.conj())
        else:
            eigvals, eigvecs_tuple = util.eig_lr(matrix, hermitian=False)
            if overlap is not None:
                left, right = eigvecs_tuple
                left = util.rotate_subspace(left, unorth.T.conj())
                right = util.rotate_subspace(right, unorth)
                eigvecs_tuple = (left, right)
            eigvecs = np.array(eigvecs_tuple)

        return eigvals, eigvecs

    def diagonalise_matrix_with_projection(
        self, physical: Array, chempot: bool | float = False, overlap: Array | None = None
    ) -> tuple[Array, Array]:
        r"""Diagonalise the supermatrix and project the eigenvectors into the physical space.

        The projection of the eigenvectors is

        .. math::
            \mathbf{x}_\mathrm{phys} = \mathbf{P}_\mathrm{phys} \mathbf{x},

        where :math:`\mathbf{P}_\mathrm{phys}` is the projection operator onto the physical space,
        which can be written as

        .. math::
            \mathbf{P}_\mathrm{phys} = \begin{bmatrix} \mathbf{I} & 0 \\ 0 & 0 \end{bmatrix},

        within the supermatrix block structure of :meth:`matrix`.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.
            overlap: The overlap matrix to use for the physical space part of the supermatrix. If
                `None`, the identity matrix is used.

        Returns:
            The eigenvalues and eigenvectors of the supermatrix, with the eigenvectors projected
            into the physical space.

        See Also:
            :meth:`diagonalise_matrix` for the full eigenvalue decomposition of the supermatrix.
        """
        eigvals, eigvecs = self.diagonalise_matrix(physical, chempot=chempot, overlap=overlap)
        eigvecs_projected = eigvecs[..., : self.nphys, :]
        return eigvals, eigvecs_projected

    # Methods associated with a quasiparticle representation:

    def weights(self, occupancy: float = 1.0) -> Array:
        r"""Get the weights of the residues in the Lehmann representation.

        The weights are defined as

        .. math::
            w_k = \sum_{p} v_{pk} u_{pk}^*,

        where :math:`w_k` is the weight of residue :math:`k`.

        Args:
            occupancy: The occupancy of the states.

        Returns:
            The weights of each state.
        """
        left, right = self.unpack_couplings()
        weights = util.einsum("pk,pk->k", right, left.conj()) * occupancy
        return weights

    def as_orbitals(
        self, occupancy: float = 1.0, mo_coeff: Array | None = None
    ) -> tuple[
        Array,
        Array,
        Array,
    ]:
        """Convert the Lehmann representation to an orbital representation.

        Args:
            occupancy: The occupancy of the states.
            mo_coeff: The molecular orbital coefficients. If given, the couplings will have their
                physical dimension rotated into the AO basis according to these coefficients.

        Returns:
            The energies, coefficients, and occupancies of the states.

        Note:
            This representation is intended to be compatible with PySCF's mean-field representation
            of molecular orbitals.
        """
        if not self.hermitian:
            raise NotImplementedError("Cannot convert non-Hermitian system orbitals.")
        energies = self.energies
        couplings, _ = self.unpack_couplings()
        coeffs = couplings if mo_coeff is None else mo_coeff @ couplings
        occupancies = np.concatenate(
            [
                np.abs(self.occupied().weights(occupancy=occupancy)),
                np.zeros(self.virtual().naux),
            ]
        )
        return energies, coeffs, occupancies

    def as_perturbed_mo_energy(self) -> Array:
        r"""Return an array of :math:`N_\mathrm{phys}` pole energies according to best overlap.

        The pole energies are selected as

        .. math::
            \epsilon_p = \epsilon_k \quad \text{where} \quad k = \arg\max_{k} |v_{pk} u_{pk}^*|,

        where :math:`\epsilon_p` is the energy of the physical state :math:`p`, and :math:`k` is the
        index of a pole in the Lehmann representation.

        Returns:
            The selected energies.

        Note:
            The return value of this function is intended to be compatible with
            :attr:`pyscf.scf.hf.SCF.mo_energy`, i.e. it represents a reduced quasiparticle picture
            consisting of :math:`N_\mathrm{phys}` energies that are picked from the poles of the
            Lehmann representation, according to the best overlap with the MO of the same index.
        """
        left, right = self.unpack_couplings()
        weights = right * left.conj()
        energies = [self.energies[np.argmax(np.abs(weights[i]))] for i in range(self.nphys)]
        return np.asarray(energies)

    # Methods associated with a static approximation to a self-energy:

    def as_static_potential(self, mo_energy: Array, eta: float = 1e-2) -> Array:
        r"""Convert the Lehmann representation to a static potential.

        The static potential is defined as

        .. math::
            V_{pq} = \mathrm{Re}\left[ \sum_{k} \frac{v_{pk} u_{qk}^*}{\epsilon_p - \epsilon_k
            \pm i \eta} \right].

        Args:
            mo_energy: The molecular orbital energies.
            eta: The broadening parameter.

        Returns:
            The static potential.

        Note:
            The static potential in this format is common in methods such as quasiparticle
            self-consistent :math:`GW` calculations.
        """
        left, right = self.unpack_couplings()
        energies = self.energies + np.sign(self.energies - self.chempot) * 1.0j * eta
        denom = mo_energy[:, None] - energies[None]

        # Calculate the static potential
        static = util.einsum("pk,qk,pk->pq", right, left.conj(), 1.0 / denom).real
        static = 0.5 * (static + static.T)

        return static

    # Methods for combining Lehmann representations:

    def split_physical(self, nocc: int) -> tuple[Lehmann, Lehmann]:
        """Split the physical domain of Lehmann representation into occupied and virtual parts.

        Args:
            nocc: The number of occupied states.

        Returns:
            The Lehmann representation coupled with the occupied and virtual parts, as separate
            Lehmann representations.

        Note:
            The Fermi level (value at which the parts are separated) is defined by the chemical
            potential :attr:`chempot`.
        """
        occ = self.__class__(
            self.energies,
            self.couplings[..., :nocc, :],
            chempot=self.chempot,
            sort=False,
        )
        vir = self.__class__(
            self.energies,
            self.couplings[..., nocc:, :],
            chempot=self.chempot,
            sort=False,
        )
        return occ, vir

    def combine_physical(self, other: Lehmann) -> Lehmann:
        """Combine the physical domain of two Lehmann representations.

        Args:
            other: The other Lehmann representation to combine with.

        Returns:
            A new Lehmann representation that is the combination of the two.

        Raises:
            ValueError: If the two representations have different chemical potentials.
        """
        if not np.isclose(self.chempot, other.chempot):
            raise ValueError(
                f"Cannot combine Lehmann representations with different chemical potentials: "
                f"{self.chempot} and {other.chempot}"
            )

        # Combine the energies and couplings
        energies = np.concatenate((self.energies, other.energies), axis=0)
        if self.hermitian and other.hermitian:
            couplings = scipy.linalg.block_diag(self.couplings, other.couplings)
        else:
            left_self, right_self = self.unpack_couplings()
            left_other, right_other = other.unpack_couplings()
            couplings = np.array(
                [
                    np.concatenate((left_self, left_other), axis=-1),
                    np.concatenate((right_self, right_other), axis=-1),
                ]
            )

        return self.__class__(energies, couplings, chempot=self.chempot, sort=True)

    def concatenate(self, other: Lehmann) -> Lehmann:
        """Concatenate two Lehmann representations.

        Args:
            other: The other Lehmann representation to concatenate.

        Returns:
            A new Lehmann representation that is the concatenation of the two.

        Raises:
            ValueError: If the two representations have different physical dimensions or chemical
                potentials.
        """
        if self.nphys != other.nphys:
            raise ValueError(
                f"Cannot combine Lehmann representations with different physical dimensions: "
                f"{self.nphys} and {other.nphys}"
            )
        if not np.isclose(self.chempot, other.chempot):
            raise ValueError(
                f"Cannot combine Lehmann representations with different chemical potentials: "
                f"{self.chempot} and {other.chempot}"
            )

        # Combine the energies and couplings
        energies = np.concatenate((self.energies, other.energies))
        if self.hermitian and other.hermitian:
            couplings = np.concatenate((self.couplings, other.couplings), axis=-1)
        else:
            left_self, right_self = self.unpack_couplings()
            left_other, right_other = other.unpack_couplings()
            couplings = np.array(
                [
                    np.concatenate((left_self, left_other), axis=-1),
                    np.concatenate((right_self, right_other), axis=-1),
                ]
            )

        return self.__class__(energies, couplings, chempot=self.chempot, sort=False)

    def __eq__(self, other: object) -> bool:
        """Check if two spectral representations are equal."""
        if not isinstance(other, Lehmann):
            return NotImplemented
        if other.nphys != self.nphys:
            return False
        if other.naux != self.naux:
            return False
        if other.hermitian != self.hermitian:
            return False
        if other.chempot != self.chempot:
            return False
        return np.allclose(other.energies, self.energies) and (
            np.allclose(other.couplings, self.couplings)
        )

    def __hash__(self) -> int:
        """Return a hash of the Lehmann representation."""
        return hash((tuple(self.energies), tuple(self.couplings.flatten()), self.chempot))
