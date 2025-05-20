"""Tests for :class:`~dyson.expressions`."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pyscf
import pytest

from dyson import util
from dyson.expressions import CCSD, FCI, HF

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter


def test_init(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the instantiation of the expression from a mean-field object."""
    expression = expression_cls.from_mf(mf)
    assert expression.mol is mf.mol
    assert expression.nphys == mf.mol.nao
    assert expression.nocc == mf.mol.nelectron // 2
    assert expression.nvir == mf.mol.nao - mf.mol.nelectron // 2


def test_hamiltonian(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the Hamiltonian of the expression."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = expression.diagonal()
    hamiltonian = expression.build_matrix()

    assert np.allclose(np.diag(hamiltonian), diagonal)
    assert hamiltonian.shape == expression.shape
    assert (expression.nconfig + expression.nsingle) == diagonal.size


def test_gf_moments(mf: scf.hf.RHF, expression_cls: dict[str, type[BaseExpression]]) -> None:
    """Test the Green's function moments of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    hamiltonian = expression.build_matrix()

    # Construct the moments
    moments = np.zeros((2, expression.nphys, expression.nphys))
    for i, j in itertools.product(range(expression.nphys), repeat=2):
        bra = expression.get_state_bra(j)
        ket = expression.get_state_ket(i)
        moments[0, i, j] += bra.conj() @ ket
        moments[1, i, j] += np.einsum("j,i,ij->", bra.conj(), ket, hamiltonian)

    # Compare the moments to the reference
    ref = expression.build_gf_moments(2)

    assert np.allclose(ref[0], moments[0])
    assert np.allclose(ref[1], moments[1])


def test_static(
    mf: scf.hf.RHF,
    expression_cls: dict[str, type[BaseExpression]],
    exact_cache: ExactGetter,
) -> None:
    """Test the static self-energy of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    gf_moments = expression.build_gf_moments(2)

    # Get the static self-energy
    exact = exact_cache(mf, expression)
    static = exact.result.get_static_self_energy()

    assert np.allclose(static, gf_moments[1])


def test_hf(mf: scf.hf.RHF) -> None:
    """Test the HF expression."""
    hf_h = HF["1h"].from_mf(mf)
    hf_p = HF["1p"].from_mf(mf)
    gf_h_moments = hf_h.build_gf_moments(2)
    gf_p_moments = hf_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_h_moments, h1e, factor=1.0)

    assert np.abs(energy - mf.energy_elec()[0]) < 1e-8

    # Get the Fock matrix Fock matrix from the moments
    fock_ref = np.einsum("pq,pi,qj->ij", mf.get_fock(), mf.mo_coeff, mf.mo_coeff)
    fock = gf_h_moments[1] + gf_p_moments[1]

    assert np.allclose(fock, fock_ref)


def test_ccsd(mf: scf.hf.RHF) -> None:
    """Test the CCSD expression."""
    ccsd = CCSD["1h"].from_mf(mf)
    gf_moments = ccsd.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments, h1e, factor=1.0)
    energy_ref = pyscf.cc.CCSD(mf).run(conv_tol=1e-10).e_tot - mf.mol.energy_nuc()

    with pytest.raises(AssertionError):
        # Galitskii--Migdal should not capture the energy for CCSD
        assert np.abs(energy - energy_ref) < 1e-8


def test_fci(mf: scf.hf.RHF) -> None:
    """Test the FCI expression."""
    fci = FCI["1h"].from_mf(mf)
    gf_moments = fci.build_gf_moments(2)
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments, h1e, factor=1.0)
    energy_ref = pyscf.fci.FCI(mf).kernel()[0] - mf.mol.energy_nuc()

    assert np.abs(energy - energy_ref) < 1e-8
