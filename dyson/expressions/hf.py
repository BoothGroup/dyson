"""Hartree--Fock (HF) expressions."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.expressions.expression import BaseExpression, ExpressionCollection

if TYPE_CHECKING:
    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array


class BaseHF(BaseExpression):
    """Base class for HF expressions."""

    hermitian = True

    def __init__(
        self,
        mol: Mole,
        mo_energy: Array,
    ):
        """Initialise the expression.

        Args:
            mol: Molecule object.
            mo_energy: Molecular orbital energies.
        """
        self._mol = mol
        self._mo_energy = mo_energy

    @classmethod
    def from_mf(cls, mf: RHF) -> BaseHF:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object.

        Returns:
            Expression object.
        """
        return cls(mf.mol, mf.mo_energy)

    def apply_hamiltonian(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return self.diagonal() * vector

    @abstractmethod
    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        pass

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments.

        Returns:
            Self-energy moments.
        """
        return np.zeros((nmom, self.nphys, self.nphys))

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        return self._mol

    @property
    def mo_energy(self) -> Array:
        """Molecular orbital energies."""
        return self._mo_energy

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return True

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return 0


class HF_1h(BaseHF):  # pylint: disable=invalid-name
    """HF expressions for the hole Green's function."""

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self.mo_energy[: self.nocc]

    def get_excitation_vector(self, orbital: int) -> Array:
        r"""Obtain the vector corresponding to a fermionic operator acting on the ground state.

        This vector is a generalisation of

        .. math::
            f_i^{\pm} \left| \Psi_0 \right>

        where :math:`f_i^{\pm}` is the fermionic creation or annihilation operator, or a product
        thereof, depending on the particular expression and what Green's function it corresponds to.

        The vector defines the excitaiton manifold probed by the Green's function corresponding to
        the expression.

        Args:
            orbital: Orbital index.

        Returns:
            Excitation vector.
        """
        if orbital < self.nocc:
            return util.unit_vector(self.shape[0], orbital)
        return np.zeros(self.shape[0])

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nocc


class HF_1p(BaseHF):  # pylint: disable=invalid-name
    """HF expressions for the particle Green's function."""

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self.mo_energy[self.nocc :]

    def get_excitation_vector(self, orbital: int) -> Array:
        r"""Obtain the vector corresponding to a fermionic operator acting on the ground state.

        This vector is a generalisation of

        .. math::
            f_i^{\pm} \left| \Psi_0 \right>

        where :math:`f_i^{\pm}` is the fermionic creation or annihilation operator, or a product
        thereof, depending on the particular expression and what Green's function it corresponds to.

        The vector defines the excitaiton manifold probed by the Green's function corresponding to
        the expression.

        Args:
            orbital: Orbital index.

        Returns:
            Excitation vector.
        """
        if orbital >= self.nocc:
            return util.unit_vector(self.shape[0], orbital - self.nocc)
        return np.zeros(self.shape[0])

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nvir


class HF_Dyson(BaseHF):  # pylint: disable=invalid-name
    """HF expressions for the Dyson Green's function."""

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self.mo_energy

    def get_excitation_vector(self, orbital: int) -> Array:
        r"""Obtain the vector corresponding to a fermionic operator acting on the ground state.

        This vector is a generalisation of

        .. math::
            f_i^{\pm} \left| \Psi_0 \right>

        where :math:`f_i^{\pm}` is the fermionic creation or annihilation operator, or a product
        thereof, depending on the particular expression and what Green's function it corresponds to.

        The vector defines the excitaiton manifold probed by the Green's function corresponding to
        the expression.

        Args:
            orbital: Orbital index.

        Returns:
            Excitation vector.
        """
        return util.unit_vector(self.shape[0], orbital)

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nocc + self.nvir

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False


HF = ExpressionCollection(HF_1h, HF_1p, HF_Dyson, None, name="HF")
