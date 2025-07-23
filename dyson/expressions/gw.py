"""GW approximation expressions [hedin1965]_ [aryasetiawan1998]_ [zhu2021]_.

.. [hedin1965] Hedin, L. (1965). New Method for Calculating the One-Particle Green’s Function with
   Application to the Electron-Gas Problem. Physical Review, 139(3A), A796–A823.
   https://doi.org/10.1103/physrev.139.a796

.. [aryasetiawan1998] Aryasetiawan, F., & Gunnarsson, O. (1998). The GW method. Reports on Progress
   in Physics, 61(3), 237–312. https://doi.org/10.1088/0034-4885/61/3/002

.. [zhu2021] Zhu, T., & Chan, G. K. (2021). All-Electron Gaussian-Based G0W0 for valence and core
   excitation energies of periodic systems. Journal of Chemical Theory and Computation, 17(2),
   727–741. https://doi.org/10.1021/acs.jctc.0c00704

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import gw, lib

from dyson import numpy as np
from dyson import util
from dyson.expressions.expression import BaseExpression, ExpressionCollection

if TYPE_CHECKING:
    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array


class BaseGW_Dyson(BaseExpression):
    """Base class for GW expressions for the Dyson Green's function."""

    hermitian_downfolded = True
    hermitian_upfolded = False

    def __init__(
        self,
        mol: Mole,
        gw_obj: gw.GW,
        eris: Array | None = None,
    ) -> None:
        """Initialise the expression.

        Args:
            mol: Molecule object.
            gw_obj: GW object from PySCF.
            eris: Density fitted electron repulsion integrals from PySCF.
        """
        self._mol = mol
        self._gw = gw_obj
        self._eris = eris if eris is not None else gw_obj.ao2mo()

        if getattr(self._gw._scf, "xc", "hf") != "hf":
            raise NotImplementedError(
                "GW expressions currently only support Hartree--Fock mean-field objects."
            )

    @classmethod
    def from_mf(cls, mf: RHF) -> BaseGW_Dyson:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object.

        Returns:
            Expression object.
        """
        return cls.from_gw(gw.GW(mf))

    @classmethod
    def from_gw(cls, gw: gw.GW) -> BaseGW_Dyson:
        """Create an expression from a GW object.

        Args:
            gw: GW object.

        Returns:
            Expression object.
        """
        return cls(gw._scf.mol, gw)

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        raise NotImplementedError("Self-energy moments not implemented for GW.")

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
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nocc * self.nocc * self.nvir + self.nvir * self.nvir * self.nocc

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        return self._mol

    @property
    def gw(self) -> gw.GW:
        """GW object."""
        return self._gw

    @property
    def eris(self) -> Array:
        """Density fitted electron repulsion integrals."""
        return self._eris

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False


class TDAGW_Dyson(BaseGW_Dyson):
    """GW expressions with Tamm--Dancoff (TDA) approximation for the Dyson Green's function."""

    def apply_hamiltonian(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        # Get the slices for each sector
        o1 = slice(None, self.nocc)
        v1 = slice(self.nocc, self.nocc + self.nvir)
        o2 = slice(self.nocc + self.nvir, self.nocc + self.nvir + self.nocc * self.nocc * self.nvir)
        v2 = slice(self.nocc + self.nvir + self.nocc * self.nocc * self.nvir, None)

        # Get the blocks of the ERIs
        Lia = self.eris[:, o1, v1]
        Lai = self.eris[:, v1, o1]
        Lij = self.eris[:, o1, o1]
        Lab = self.eris[:, v1, v1]

        # Get the blocks of the vector
        vector_o1 = vector[o1]
        vector_v1 = vector[v1]
        vector_o2 = vector[o2].reshape(self.nocc, self.nocc, self.nvir)
        vector_v2 = vector[v2].reshape(self.nocc, self.nvir, self.nvir)

        # Get the energy denominators
        mo_energy = self.gw._scf.mo_energy if self.gw.mo_energy is None else self.gw.mo_energy
        e_ija = lib.direct_sum("i+j-a->ija", mo_energy[o1], mo_energy[o1], mo_energy[v1])
        e_iab = lib.direct_sum("a+b-i->iab", mo_energy[v1], mo_energy[v1], mo_energy[o1])

        # Perform the contractions
        r_o1 = mo_energy[o1] * vector_o1
        r_o1 += util.einsum("Qik,Qcl,klc->i", Lij, Lai, vector_o2) * 2
        r_o1 += util.einsum("Qid,Qkc,kcd->i", Lia.conj(), Lia.conj(), vector_v2) * 2

        r_v1 = mo_energy[v1] * vector_v1
        r_v1 += util.einsum("Qak,Qcl,klc->a", Lai, Lai, vector_o2) * 2
        r_v1 += util.einsum("Qad,Qkc,kcd->a", Lab.conj(), Lia.conj(), vector_v2) * 2

        r_o2 = util.einsum("Qki,Qaj,k->ija", Lij.conj(), Lai.conj(), vector_o1)
        r_o2 += util.einsum("Qbi,Qaj,b->ija", Lai.conj(), Lai.conj(), vector_v1)
        r_o2 += util.einsum("ija,ija->ija", e_ija, vector_o2)
        r_o2 -= util.einsum("Qja,Qlc,ilc->ija", Lia, Lia, vector_o2) * 2

        r_v2 = util.einsum("Qjb,Qia,j->iab", Lia, Lia, vector_o1)
        r_v2 += util.einsum("Qcb,Qia,c->iab", Lab, Lia, vector_v1)
        r_v2 += util.einsum("iab,iab->iab", e_iab, vector_v2)
        r_v2 += util.einsum("Qia,Qkc,kcb->iab", Lia, Lia, vector_v2) * 2

        return np.concatenate([r_o1, r_v1, r_o2.ravel(), r_v2.ravel()])

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        # Get the slices for each sector
        o1 = slice(None, self.nocc)
        v1 = slice(self.nocc, None)

        # Get the blocks of the ERIs
        Lia = self.eris[:, o1, v1]
        Lai = self.eris[:, v1, o1]

        # Get the energy denominators
        mo_energy = self.gw._scf.mo_energy if self.gw.mo_energy is None else self.gw.mo_energy
        e_ija = lib.direct_sum("i+j-a->ija", mo_energy[o1], mo_energy[o1], mo_energy[v1])
        e_iab = lib.direct_sum("a+b-i->iab", mo_energy[v1], mo_energy[v1], mo_energy[o1])

        # Build the diagonal
        diag_o1 = mo_energy[o1].copy()
        diag_v1 = mo_energy[v1].copy()
        diag_o2 = e_ija.ravel()
        diag_o2 -= util.einsum("Qja,Qaj,ii->ija", Lia, Lai, np.eye(self.nocc)).ravel()
        diag_v2 = e_iab.ravel()
        diag_v2 += util.einsum("Qai,Qia,bb->iab", Lai, Lia, np.eye(self.nvir)).ravel()

        return np.concatenate([diag_o1, diag_v1, diag_o2, diag_v2])


class TDAGW(ExpressionCollection):
    """Collection of TDAGW expressions for different parts of the Green's function."""

    _dyson = TDAGW_Dyson
    _name = "TDA-GW"
