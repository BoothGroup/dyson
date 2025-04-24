"""Container for a Lehmann representation."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

from dyson import numpy as np
from dyson.typing import Array

if TYPE_CHECKING:
    from typing import Iterable, Literal, TypeAlias

    import pyscf.agf2.aux

    Couplings: TypeAlias = Array | tuple[Array, Array]

einsum = functools.partial(np.einsum, optimize=True)  # TODO: Move


class Lehmann:
    r"""Lehman representation.

    The Lehmann representation is a set of poles :math:`\epsilon_k` and couplings :math:`v_{pk}`
    that can be downfolded into a frequency-dependent function as

    .. math::
        \sum_{k} \frac{v_{pk} v_{qk}^*}{\omega - \epsilon_k},

    where the couplings are between the poles :math:`k` and the physical space :math:`p` and
    :math:`q`, and may be non-Hermitian.
    """

    def __init__(
        self,
        energies: Array,
        couplings: Couplings,
        chempot: float = 0.0,
        sort: bool = True,
    ):
        """Initialise the object.

        Args:
            energies: Energies of the poles.
            couplings: Couplings of the poles to a physical space. For a non-Hermitian system, a
                tuple of left and right couplings is required.
            chempot: Chemical potential.
            sort: Sort the poles by energy.
        """
        self._energies = energies
        self._couplings = couplings
        self._chempot = chempot
        if sort:
            self.sort_()

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

    def sort_(self) -> None:
        """Sort the poles by energy.

        Note:
            The object is sorted in place.
        """
        idx = np.argsort(self.energies)
        self._energies = self.energies[idx]
        if self.hermitian:
            self._couplings = self.couplings[idx]
        else:
            left, right = self.couplings
            self._couplings = (left[idx], right[idx])

    @property
    def energies(self) -> Array:
        """Get the energies."""
        return self._energies

    @property
    def couplings(self) -> Couplings:
        """Get the couplings."""
        return self._couplings

    @property
    def chempot(self) -> float:
        """Get the chemical potential."""
        return self._chempot

    @property
    def hermitian(self) -> bool:
        """Get a boolean indicating if the system is Hermitian."""
        return not isinstance(self.couplings, tuple)

    def unpack_couplings(self) -> tuple[Array, Array]:
        """Unpack the couplings.

        Returns:
            A tuple of left and right couplings.
        """
        if self.hermitian:
            return cast(tuple[Array, Array], (self.couplings, self.couplings))
        return cast(tuple[Array, Array], self.couplings)

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
        return np.result_type(self.energies, *self.unpack_couplings())

    def __repr__(self) -> str:
        """Return a string representation of the Lehmann representation."""
        return f"Lehmann(nphys={self.nphys}, naux={self.naux}, chempot={self.chempot})"

    def mask(self, mask: Array | slice, deep: bool = True):
        """Return a part of the Lehmann representation according to a mask.

        Args:
            mask: The mask to apply.
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new Lehmann representation including only the masked states.
        """
        # Mask the energies and couplings
        energies = self.energies[mask]
        couplings = self.couplings
        if self.hermitian:
            couplings = couplings[:, mask]  # type: ignore[call-overload]
        else:
            couplings = (couplings[0][:, mask], couplings[1][:, mask])

        # Copy the couplings if requested
        if deep:
            if self.hermitian:
                couplings = couplings.copy()  # type: ignore[union-attr]
            else:
                couplings = (couplings[0].copy(), couplings[1].copy())
            energies = energies.copy()

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
            if self.hermitian:
                couplings = couplings.copy()  # type: ignore[union-attr]
            else:
                couplings = (couplings[0].copy(), couplings[1].copy())
            energies = energies.copy()

        return self.__class__(energies, couplings, chempot=self.chempot, sort=False)

    # Methods to calculate moments:

    def moments(self, order: int | Iterable[int]) -> Array:
        r"""Calculate the moment(s) of the Lehmann representation.

        The moments are defined as

        .. math::
            T_{pq}^{n} = \sum_{k} v_{pk} v_{qk}^* \epsilon_k^n,

        where :math:`T_{pq}^{n}` is the moment of order :math:`n` in the physical space.

        Args:
            order: The order(s) of the moment(s).

        Returns:
            The moment(s) of the Lehmann representation.
        """
        squeeze = False
        if isinstance(order, int):
            order = [order]
            squeeze = True
        orders = np.asarray(order)

        # Contract the moments
        left, right = self.unpack_couplings()
        moments = einsum(
            "pk,qk,nk->npq",
            left,
            right.conj(),
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
        """Calculate the Chebyshev polynomial moment(s) of the Lehmann representation.

        The Chebyshev moments are defined as

        .. math::
            T_{pq}^{n} = \sum_{k} v_{pk} v_{qk}^* P_n(\epsilon_k),

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
            emin = self.energies.min()
            emax = self.energies.max()
            scaling = (
                (emax - emin) / (2.0 - 1e-3),
                (emax + emin) / 2.0,
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
        vecs = (left, left * energies[None])
        idx = 0
        if 0 in orders:
            moments[idx] = vecs[0] @ right.T.conj()
            idx += 1
        if 1 in orders:
            moments[idx] = vecs[1] @ right.T.conj()
            idx += 1
        for i in range(2, max_order + 1):
            vecs = (vecs[1], 2 * energies * vecs[1] - vecs[0])
            if i in orders:
                moments[idx] = vecs[1] @ right.T.conj()
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
            \begin{pmatrix}
                \mathbf{f} & \mathbf{v} \\
                \mathbf{v}^\dagger & \mathbf{\epsilon} \mathbf{1}
            \end{pmatrix},

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

        # Build the supermatrix
        matrix = np.block([[physical, left], [right.T.conj(), np.diag(energies)]])

        return matrix

    def diagonal(self, physical: Array, chempot: bool | float = False) -> Array:
        r"""Build the diagonal supermatrix form of the Lehmann representation.

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
            \begin{pmatrix}
                \mathbf{x}_\mathrm{phys} \\
                \mathbf{x}_\mathrm{aux}
            \end{pmatrix}
            =
            \begin{pmatrix}
                \mathbf{f} & \mathbf{v} \\
                \mathbf{v}^\dagger & \mathbf{\epsilon} \mathbf{1}
            \end{pmatrix}
            \begin{pmatrix}
                \mathbf{r}_\mathrm{phys} \\
                \mathbf{r}_\mathrm{aux}
            \end{pmatrix},

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
        result_phys = einsum("pq,q...->p...", physical, vector_phys)
        result_phys += einsum("pk,k...->p...", left, vector_aux)
        result_aux = einsum("pk,p...->k...", right.conj(), vector_phys)
        result_aux += einsum("k,k...->k...", energies, vector_aux)
        result = np.concatenate((result_phys, result_aux), axis=0)

        return result

    def diagonalise_matrix(
        self, physical: Array, chempot: bool | float = False
    ) -> tuple[Array, Array]:
        r"""Diagonalise the supermatrix.

        The eigenvalue problem is defined as

        .. math::
            \begin{pmatrix}
                \mathbf{f} & \mathbf{v} \\
                \mathbf{v}^\dagger & \mathbf{\epsilon} \mathbf{1}
            \end{pmatrix}
            \begin{pmatrix}
                \mathbf{x}_\mathrm{phys} \\
                \mathbf{x}_\mathrm{aux}
            \end{pmatrix}
            =
            E
            \begin{pmatrix}
                \mathbf{x}_\mathrm{phys} \\
                \mathbf{x}_\mathrm{aux}
            \end{pmatrix},

        where :math:`\mathbf{f}` is the physical space part of the supermatrix, and the eigenvectors
        :math:`\mathbf{x}` span both the physical and auxiliary spaces.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.

        Returns:
            The eigenvalues and eigenvectors of the supermatrix.
        """
        matrix = self.matrix(physical, chempot=chempot)
        if self.hermitian:
            eigvals, eigvecs = np.linalg.eigh(matrix)
        else:
            eigvals, eigvecs = np.linalg.eig(matrix)
        return eigvals, eigvecs

    def diagonalise_matrix_with_projection(
        self, physical: Array, chempot: bool | float = False
    ) -> tuple[Array, Couplings]:
        """Diagonalise the supermatrix and project the eigenvectors into the physical space.

        Args:
            physical: The matrix to use for the physical space part of the supermatrix.
            chempot: Whether to include the chemical potential in the supermatrix. If `True`, the
                chemical potential from :attr:`chempot` is used. If a float is given, that value is
                used.

        Returns:
            The eigenvalues and eigenvectors of the supermatrix, with the eigenvectors projected
            into the physical space.
        """
        eigvals, eigvecs = self.diagonalise_matrix(physical, chempot=chempot)
        eigvecs_projected: Couplings
        if self.hermitian:
            eigvecs_projected = eigvecs[: self.nphys]
        else:
            left = eigvecs[: self.nphys]
            right = np.linalg.inv(eigvecs).T.conj()[: self.nphys]
            eigvecs_projected = (left, right)
        return eigvals, eigvecs_projected

    # Methods associated with a quasiparticle representation:

    def weights(self, occupancy: float = 1.0) -> Array:
        r"""Get the weights of the residues in the Lehmann representation.

        The weights are defined as

        .. math::
            w_k = \sum_{p} v_{pk} v_{pk}^*,

        where :math:`w_k` is the weight of residue :math:`k`.

        Args:
            occupancy: The occupancy of the states.

        Returns:
            The weights of each state.
        """
        left, right = self.unpack_couplings()
        weights = einsum("pk,pk->k", left, right.conj()) * occupancy
        return weights

    def as_orbitals(self, occupancy: float = 1.0, mo_coeff: Array | None = None) -> tuple[
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

        Returns:
            The selected energies.

        Note:
            The return value of this function is intended to be compatible with
            :attr:`pyscf.scf.hf.SCF.mo_energy`, i.e. it represents a reduced quasiparticle picture
            consisting of :math:`N_\mathrm{phys}` energies that are picked from the poles of the
            Lehmann representation, according to the best overlap with the MO of the same index.
        """
        left, right = self.unpack_couplings()
        weights = left * right.conj()
        energies = [self.energies[np.argmax(np.abs(weights[i]))] for i in range(self.nphys)]
        return np.asarray(energies)

    # Methods associated with a static approximation to a self-energy:

    def as_static_potential(self, mo_energy: Array, eta: float = 1e-2) -> Array:
        r"""Convert the Lehmann representation to a static potential.

        The static potential is defined as

        .. math::
            V_{pq} = \mathrm{Re}\left[ \sum_{k} \frac{v_{pk} v_{qk}^*}{\epsilon_p - \epsilon_k
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
        static = einsum("pk,qk,pk->pq", left, right.conj(), 1.0 / denom).real
        static = 0.5 * (static + static.T)

        return static

    # Methods associated with a dynamic realisation of the Lehmann representation:

    def on_grid(
        self,
        grid: Array,
        eta: float = 1e-1,
        ordering: Literal["time-ordered", "advanced", "retarded"] = "time-ordered",
        axis: Literal["real", "imag"] = "real",
        trace: bool = False,
    ) -> Array:
        r"""Calculate the Lehmann representation on a grid.

        The imaginary frequency representation is defined as

        .. math::
            \sum_{k} \frac{v_{pk} v_{qk}^*}{i \omega - \epsilon_k},

        and the real frequency representation is defined as

        .. math::
            \sum_{k} \frac{v_{pk} v_{qk}^*}{\omega - \epsilon_k \pm i \eta},

        where the sign of the broadening factor is determined by the time ordering.

        where :math:`\omega` is the frequency grid, :math:`\epsilon_k` are the poles, and

        Args:
            grid: The grid to realise the Lehmann representation on.
            eta: The broadening parameter.
            ordering: The time ordering representation.
            axis: The frequency axis to calculate use.
            trace: Whether to return only the trace.

        Returns:
            The Lehmann representation on the grid.
        """
        left, right = self.unpack_couplings()

        # Get the signs for the time ordering
        if ordering == "time-ordered":
            signs = np.sign(self.energies - self.chempot)
        elif ordering == "advanced":
            signs = -np.ones_like(self.energies)
        elif ordering == "retarded":
            signs = np.ones_like(self.energies)
        else:
            raise ValueError(f"Unknown ordering: {ordering}")

        # Get the axis
        if axis == "real":
            denom = grid[:, None] + (signs * 1.0j * eta - self.energies)[None]
        elif axis == "imag":
            denom = 1.0j * grid[:, None] - self.energies[None]
        else:
            raise ValueError(f"Unknown axis: {axis}")

        # Realise the Lehmann representation
        func = einsum(f"pk,pk,wk->{'w' if trace else 'wpq'}", left, right.conj(), 1.0 / denom)

        return func

    # Methods for combining Lehmann representations:

    def concatenate(self, other: Lehmann) -> Lehmann:
        """Concatenate two Lehmann representations.

        Args:
            other: The other Lehmann representation to concatenate.

        Returns:
            A new Lehmann representation that is the concatenation of the two.
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
        couplings: Couplings
        if self.hermitian:
            couplings = np.concatenate((self.couplings, other.couplings), axis=1)
        else:
            left_self, right_self = self.unpack_couplings()
            left_other, right_other = other.unpack_couplings()
            couplings = (
                np.concatenate((left_self, left_other), axis=1),
                np.concatenate((right_self, right_other), axis=1),
            )

        return self.__class__(energies, couplings, chempot=self.chempot, sort=False)

    def __add__(self, other: Lehmann) -> Lehmann:
        """Add two Lehmann representations.

        Args:
            other: The other Lehmann representation to add.

        Returns:
            A new Lehmann representation that is the sum of the two.
        """
        return self.concatenate(other)

    def __sub__(self, other: Lehmann) -> Lehmann:
        """Subtract two Lehmann representations.

        Args:
            other: The other Lehmann representation to subtract.

        Returns:
            A new Lehmann representation that is the difference of the two.

        Note:
            Subtracting Lehmann representations requires either non-Hermiticity or complex-valued
            couplings. The latter should maintain Hermiticity.
        """
        other_couplings = other.couplings
        if self.hermitian:
            other_couplings = 1.0j * other_couplings  # type: ignore[operator]
        else:
            other_couplings = (-other_couplings[0], other_couplings[1])
        other_factored = self.__class__(
            other.energies,
            other_couplings,
            chempot=other.chempot,
            sort=False,
        )
        return self.concatenate(other_factored)
