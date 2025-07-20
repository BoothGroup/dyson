"""Full configuration interaction (FCI) expressions."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from pyscf import ao2mo, fci

from dyson.expressions.expression import BaseExpression, ExpressionCollection

if TYPE_CHECKING:
    from typing import Callable

    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array


class BaseFCI(BaseExpression):
    """Base class for FCI expressions."""

    hermitian_downfolded = True
    hermitian_upfolded = True

    SIGN: int
    DELTA_ALPHA: int
    DELTA_BETA: int
    STATE_FUNC: Callable[[Array, int, tuple[int, int], int], Array]

    def __init__(
        self,
        mol: Mole,
        e_fci: Array,
        c_fci: Array,
        hamiltonian: Array,
        diagonal: Array,
        chempot: Array | float = 0.0,
    ):
        """Initialise the expression.

        Args:
            mol: Molecule object.
            e_fci: FCI eigenvalues.
            c_fci: FCI eigenvectors.
            hamiltonian: Hamiltonian matrix.
            diagonal: Diagonal of the FCI Hamiltonian.
            chempot: Chemical potential.
        """
        self._mol = mol
        self._e_fci = e_fci
        self._c_fci = c_fci
        self._hamiltonian = hamiltonian
        self._diagonal = diagonal
        self._chempot = chempot

    @classmethod
    def from_fci(cls, ci: fci.FCI, h1e: Array, h2e: Array) -> BaseFCI:
        """Create an expression from an FCI object.

        Args:
            ci: FCI object.
            h1e: One-electron Hamiltonian matrix.
            h2e: Two-electron Hamiltonian matrix.

        Returns:
            Expression object.
        """
        if ci.mol is None:
            raise ValueError("FCI object must be initialised with a molecule.")
        nelec = (ci.mol.nelec[0] + cls.DELTA_ALPHA, ci.mol.nelec[1] + cls.DELTA_BETA)
        hamiltonian = ci.absorb_h1e(h1e, h2e, ci.mol.nao, nelec, 0.5)
        diagonal = ci.make_hdiag(h1e, h2e, ci.mol.nao, nelec)
        return cls(
            ci.mol,
            ci.eci,
            ci.ci,
            hamiltonian,
            diagonal,
        )

    @classmethod
    def from_mf(cls, mf: RHF) -> BaseFCI:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object.

        Returns:
            Expression object.
        """
        h1e = mf.mo_coeff.T.conj() @ mf.get_hcore() @ mf.mo_coeff
        h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)  # pylint: disable=protected-access
        ci = fci.direct_spin1.FCI(mf.mol)
        ci.verbose = 0
        ci.kernel(h1e, h2e, mf.mol.nao, mf.mol.nelec)
        return cls.from_fci(ci, h1e, h2e)

    def apply_hamiltonian(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        nelec = (self.nocc + self.DELTA_ALPHA, self.nocc + self.DELTA_BETA)
        result = fci.direct_spin1.contract_2e(
            self.hamiltonian,
            vector,
            self.nphys,
            nelec,
            self.link_index,
        )
        result -= (self.e_fci + self.chempot) * vector
        return self.SIGN * result

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self.SIGN * (self._diagonal - (self.e_fci + self.chempot))

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
        return self.STATE_FUNC(
            self.c_fci,
            self.nphys,
            (self.nocc, self.nocc),
            orbital,
        ).ravel()

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        raise NotImplementedError("Self-energy moments not implemented for FCI.")

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        return self._mol

    @property
    def e_fci(self) -> Array:
        """FCI eigenvalues."""
        return self._e_fci

    @property
    def c_fci(self) -> Array:
        """FCI eigenvectors."""
        return self._c_fci

    @property
    def hamiltonian(self) -> Array:
        """Hamiltonian matrix."""
        return self._hamiltonian

    @property
    def chempot(self) -> Array | float:
        """Chemical potential."""
        return self._chempot

    @functools.cached_property
    def link_index(self) -> tuple[Array, Array]:
        """Index helpers."""
        nelec = (self.nocc + self.DELTA_ALPHA, self.nocc + self.DELTA_BETA)
        return (
            fci.cistring.gen_linkstr_index_trilidx(range(self.nphys), nelec[0]),
            fci.cistring.gen_linkstr_index_trilidx(range(self.nphys), nelec[1]),
        )

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        link_index = self.link_index
        return len(link_index[0]) * len(link_index[1]) - self.nsingle


class FCI_1h(BaseFCI):  # pylint: disable=invalid-name
    """FCI expressions for the hole Green's function."""

    SIGN = -1
    DELTA_ALPHA = -1
    DELTA_BETA = 0
    STATE_FUNC = staticmethod(fci.addons.des_a)

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nocc


class FCI_1p(BaseFCI):  # pylint: disable=invalid-name
    """FCI expressions for the particle Green's function."""

    SIGN = 1
    DELTA_ALPHA = 1
    DELTA_BETA = 0
    STATE_FUNC = staticmethod(fci.addons.cre_a)

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nvir


class FCI(ExpressionCollection):
    """Collection of FCI expressions for different parts of the Green's function."""

    _hole = FCI_1h
    _particle = FCI_1p
    _name = "FCI"
