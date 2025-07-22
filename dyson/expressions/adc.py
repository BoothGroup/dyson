"""Algebraic diagrammatic construction theory (ADC) expressions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pyscf import adc, ao2mo

from dyson import numpy as np
from dyson import util
from dyson.expressions.expression import BaseExpression, ExpressionCollection

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any

    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array


class BaseADC(BaseExpression):
    """Base class for ADC expressions."""

    hermitian_downfolded = True
    hermitian_upfolded = False

    PYSCF_ADC: ModuleType
    SIGN: int
    METHOD: str = "adc(2)"
    METHOD_TYPE: str = "ip"

    def __init__(self, mol: Mole, adc_obj: adc.radc.RADC, imds: Any, eris: Any) -> None:
        """Initialise the expression.

        Args:
            mol: The molecule object.
            adc_obj: PySCF ADC object.
            imds: Intermediates from PySCF.
            eris: Electron repulsion integrals from PySCF.
        """
        self._mol = mol
        self._adc_obj = adc_obj
        self._imds = imds
        self._eris = eris

    @classmethod
    def from_adc(cls, adc_obj: adc.radc.RADC) -> BaseADC:
        """Construct an MP2 expression from an ADC object.

        Args:
            adc_obj: ADC object.

        Returns:
            Expression object.
        """
        if adc_obj.t1 is None or adc_obj.t2 is None:
            warnings.warn("ADC object is not converged.", UserWarning, stacklevel=2)
        eris = adc_obj.transform_integrals()
        imds = cls.PYSCF_ADC.get_imds(adc_obj, eris)
        return cls(adc_obj.mol, adc_obj, imds, eris)

    @classmethod
    def from_mf(cls, mf: RHF) -> BaseADC:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object.

        Returns:
            Expression object.
        """
        adc_obj = adc.radc.RADC(mf)
        adc_obj.method = cls.METHOD
        adc_obj.method_type = cls.METHOD_TYPE
        adc_obj.kernel_gs()
        return cls.from_adc(adc_obj)

    def apply_hamiltonian(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        if np.iscomplexobj(vector):
            if np.max(np.abs(vector.imag)) > 1e-11:
                raise ValueError("ADC does not support complex vectors.")
            vector = vector.real
        return self.PYSCF_ADC.matvec(self._adc_obj, self._imds, self._eris)(vector) * self.SIGN

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self.PYSCF_ADC.get_diag(self._adc_obj, self._imds, self._eris) * self.SIGN

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        return self._mol

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False


class BaseADC_1h(BaseADC):
    """Base class for ADC expressions with one-hole Green's function."""

    PYSCF_ADC = adc.radc_ip
    SIGN = -1
    METHOD_TYPE = "ip"

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


class BaseADC_1p(BaseADC):
    """Base class for ADC expressions with one-particle Green's function."""

    PYSCF_ADC = adc.radc_ea
    SIGN = 1
    METHOD_TYPE = "ea"

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


class ADC2_1h(BaseADC_1h):
    """ADC(2) expressions for the one-hole Green's function."""

    METHOD = "adc(2)"

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        # Get the orbital energies and coefficients
        eo = self._adc_obj.mo_energy[: self.nocc]
        ev = self._adc_obj.mo_energy[self.nocc :]
        co = self._adc_obj.mo_coeff[:, : self.nocc]
        cv = self._adc_obj.mo_coeff[:, self.nocc :]

        # Rotate the two-electron integrals
        ooov = ao2mo.kernel(self._adc_obj.mol, (co, co, co, cv), compact=False)
        ooov = ooov.reshape(eo.size, eo.size, eo.size, ev.size)
        left = ooov * 2 - ooov.swapaxes(1, 2)

        # Recursively build the moments
        moments_occ: list[Array] = []
        for i in range(nmom):
            moments_occ.append(util.einsum("ikla,jkla->ij", left, ooov.conj()))
            if i < nmom - 1:
                left = (
                    +util.einsum("ikla,k->ikla", left, eo)
                    + util.einsum("ikla,l->ikla", left, eo)
                    - util.einsum("ikla,a->ikla", left, ev)
                )

        # Include the virtual contributions
        moments = np.array(
            [util.block_diag(moment, np.zeros((self.nvir, self.nvir))) for moment in moments_occ]
        )

        return moments

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nocc * self.nocc * self.nvir


class ADC2_1p(BaseADC_1p):
    """ADC(2) expressions for the one-particle Green's function."""

    METHOD = "adc(2)"

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        # Get the orbital energies and coefficients
        eo = self._adc_obj.mo_energy[: self.nocc]
        ev = self._adc_obj.mo_energy[self.nocc :]
        co = self._adc_obj.mo_coeff[:, : self.nocc]
        cv = self._adc_obj.mo_coeff[:, self.nocc :]

        # Rotate the two-electron integrals
        vvvo = ao2mo.kernel(self._adc_obj.mol, (cv, cv, cv, co), compact=False)
        vvvo = vvvo.reshape(ev.size, ev.size, ev.size, eo.size)
        left = vvvo * 2 - vvvo.swapaxes(1, 2)

        # Recursively build the moments
        moments_vir: list[Array] = []
        for i in range(nmom):
            moments_vir.append(util.einsum("acdi,bcdi->ab", left, vvvo.conj()))
            if i < nmom - 1:
                left = (
                    +util.einsum("acdi,c->acdi", left, ev)
                    + util.einsum("acdi,d->acdi", left, ev)
                    - util.einsum("acdi,i->acdi", left, eo)
                )

        # Include the occupied contributions
        moments = np.array(
            [util.block_diag(np.zeros((self.nocc, self.nocc)), moment) for moment in moments_vir]
        )

        return moments

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nvir * self.nvir * self.nocc


class ADC2x_1h(BaseADC_1h):
    """ADC(2)-x expressions for the one-hole Green's function."""

    METHOD = "adc(2)-x"

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        raise NotImplementedError("Self-energy moments not implemented for ADC(2)-x.")

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nocc * self.nocc * self.nvir


class ADC2x_1p(BaseADC_1p):
    """ADC(2)-x expressions for the one-particle Green's function."""

    METHOD = "adc(2)-x"

    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        raise NotImplementedError("Self-energy moments not implemented for ADC(2)-x.")

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nvir * self.nvir * self.nocc


class ADC2(ExpressionCollection):
    """Collection of ADC(2) expressions for different parts of the Green's function."""

    _hole = ADC2_1h
    _particle = ADC2_1p
    _name = "ADC(2)"


class ADC2x(ExpressionCollection):
    """Collection of ADC(2)-x expressions for different parts of the Green's function."""

    _hole = ADC2x_1h
    _particle = ADC2x_1p
    _name = "ADC(2)-x"
