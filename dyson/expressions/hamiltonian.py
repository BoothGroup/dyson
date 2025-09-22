"""Hamiltonian-driven expressions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from dyson import numpy as np
from dyson import util
from dyson.expressions.expression import BaseExpression
from dyson.representations.enums import Reduction

if TYPE_CHECKING:
    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF
    from scipy.sparse import spmatrix as SparseArray

    from dyson.typing import Array


class Hamiltonian(BaseExpression):
    """Hamiltonian-driven expressions for the Green's function."""

    def __init__(
        self,
        hamiltonian: Array | SparseArray,
        bra: Array | None = None,
        ket: Array | None = None,
    ):
        """Initialise the expression.

        Args:
            hamiltonian: Hamiltonian matrix.
            bra: Bra excitation vector. If not passed, a unit vectors are used. See
                :meth:`~dyson.expressions.expression.BaseExpression.get_excitation_bra`.
            ket: Ket excitation vector. If not passed, ``bra`` is used. See
                :meth:`~dyson.expressions.expression.BaseExpression.get_excitation_ket`.
        """
        self._hamiltonian = hamiltonian
        self._bra = bra
        self._ket = ket

        if isinstance(hamiltonian, np.ndarray):
            self.hermitian_upfolded = np.allclose(hamiltonian, hamiltonian.T.conj())
        else:
            self.hermitian_upfolded = (hamiltonian - hamiltonian.H).nnz == 0
        self.hermitian_downfolded = self.hermitian_upfolded and ket is None

    @classmethod
    def from_mf(cls, mf: RHF) -> Hamiltonian:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object.

        Returns:
            Expression object.
        """
        raise NotImplementedError("Cannot create Hamiltonian expression from mean-field object.")

    def apply_hamiltonian(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return self._hamiltonian @ vector

    def apply_hamiltonian_left(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the left.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return vector @ self._hamiltonian

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self._hamiltonian.diagonal()

    def build_matrix(self) -> Array:
        """Build the Hamiltonian matrix.

        Returns:
            Hamiltonian matrix.
        """
        if isinstance(self._hamiltonian, np.ndarray):
            return self._hamiltonian
        else:
            size = self.diagonal().size
            if size > 2048:
                warnings.warn(
                    "The Hamiltonian matrix is large. This may take a while to compute.",
                    UserWarning,
                    2,
                )
            return self._hamiltonian.todense()

    def get_excitation_bra(self, orbital: int) -> Array:
        r"""Obtain the bra vector corresponding to a fermionic operator acting on the ground state.

        The bra vector is the excitation vector corresponding to the bra state, which may or may not
        be the same as the ket state vector.

        Args:
            orbital: Orbital index.

        Returns:
            Bra excitation vector.

        See Also:
            :func:`get_excitation_vector`: Function to get the excitation vector when the bra and
            ket are the same.
        """
        if self._bra is None:
            return util.unit_vector(self.shape[0], orbital)
        return self._bra[orbital]

    def get_excitation_ket(self, orbital: int) -> Array:
        r"""Obtain the ket vector corresponding to a fermionic operator acting on the ground state.

        The ket vector is the excitation vector corresponding to the ket state, which may or may not
        be the same as the bra state vector.

        Args:
            orbital: Orbital index.

        Returns:
            Ket excitation vector.

        See Also:
            :func:`get_excitation_vector`: Function to get the excitation vector when the bra and
            ket are the same.
        """
        if self._ket is None:
            return self.get_excitation_bra(orbital)
        return self._ket[orbital]

    get_excitation_vector = get_excitation_ket
    get_excitation_vector.__doc__ = BaseExpression.get_excitation_vector.__doc__

    def build_se_moments(self, nmom: int, reduction: Reduction = Reduction.NONE) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.
            reduction: Reduction method to apply to the moments.

        Returns:
            Moments of the self-energy.
        """
        raise NotImplementedError("Self-energy moments not implemented for Hamiltonian.")

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        raise NotImplementedError("Molecule object not available for Hamiltonian expression.")

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False

    @property
    def nphys(self) -> int:
        """Number of physical orbitals."""
        return self._bra.shape[0] if self._bra is not None else 1

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        raise NotImplementedError("Excitation sectors not implemented for Hamiltonian.")

    @property
    def nconfig(self) -> int:
        """Number of configurations in the non-singles sectors."""
        raise NotImplementedError("Excitation sectors not implemented for Hamiltonian.")

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the Hamiltonian matrix."""
        assert self._hamiltonian.ndim == 2
        return cast(tuple[int, int], self._hamiltonian.shape)

    @property
    def nocc(self) -> int:
        """Number of occupied orbitals."""
        raise NotImplementedError("Orbital occupancy not defined for Hamiltonian.")

    @property
    def nvir(self) -> int:
        """Number of virtual orbitals."""
        raise NotImplementedError("Orbital occupancy not defined for Hamiltonian.")
