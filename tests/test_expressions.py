"""Tests for :class:`~dyson.expressions`."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pyscf
import pytest

from dyson import util
from dyson.expressions import ADC2, CCSD, FCI, HF, TDAGW, ADC2x
from dyson.solvers import Davidson, Exact

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


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

    if expression_cls in ADC2._classes:
        # ADC(2)-x diagonal is set to ADC(2) diagonal in PySCF for better Davidson convergence
        assert np.allclose(np.diag(hamiltonian), diagonal)
    assert hamiltonian.shape == expression.shape
    assert (expression.nconfig + expression.nsingle) == diagonal.size

    vector = np.random.random(expression.nconfig + expression.nsingle)
    hv = expression.apply_hamiltonian_right(vector)
    try:
        vh = expression.apply_hamiltonian_left(vector)
    except NotImplementedError:
        vh = None

    assert np.allclose(hv, hamiltonian @ vector)
    if vh is not None:
        assert np.allclose(vh, vector @ hamiltonian)


def test_gf_moments(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the Green's function moments of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    hamiltonian = expression.build_matrix()

    # Construct the moments
    moments = np.zeros((2, expression.nphys, expression.nphys))
    for i, j in itertools.product(range(expression.nphys), repeat=2):
        bra = expression.get_excitation_bra(j)
        ket = expression.get_excitation_ket(i)
        moments[0, j, i] += bra.conj() @ ket
        moments[1, j, i] += bra.conj() @ hamiltonian @ ket

    # Compare the moments to the reference
    ref = expression.build_gf_moments(2)

    assert np.allclose(ref[0], moments[0])
    assert np.allclose(ref[1], moments[1])


def test_static(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test the static self-energy of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    gf_moments = expression.build_gf_moments(2)

    # Get the static self-energy
    exact = exact_cache(mf, expression_cls)
    assert exact.result is not None
    greens_function = exact.result.get_greens_function()
    static = exact.result.get_static_self_energy()

    assert helper.have_equal_moments(gf_moments, greens_function, 2)
    assert np.allclose(static, gf_moments[1])


def test_hf(mf: scf.hf.RHF) -> None:
    """Test the HF expression."""
    hf_h = HF.h.from_mf(mf)
    hf_p = HF.p.from_mf(mf)
    hf_dyson = HF["dyson"].from_mf(mf)
    gf_h_moments = hf_h.build_gf_moments(2)
    gf_p_moments = hf_p.build_gf_moments(2)
    gf_dyson_moments = hf_dyson.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_h_moments, h1e, factor=1.0)

    assert np.abs(energy - mf.energy_elec()[0]) < 1e-8

    # Get the Fock matrix Fock matrix from the moments
    fock_ref = np.einsum("pq,pi,qj->ij", mf.get_fock(), mf.mo_coeff, mf.mo_coeff)
    fock = gf_h_moments[1] + gf_p_moments[1]

    assert np.allclose(fock, fock_ref)
    assert np.allclose(gf_dyson_moments[1], fock)

    # Get the Green's function from the Exact solver
    exact_h = Exact.from_expression(hf_h)
    exact_h.kernel()
    exact_p = Exact.from_expression(hf_p)
    exact_p.kernel()
    assert exact_h.result is not None
    assert exact_p.result is not None
    result = exact_h.result.combine(exact_p.result)

    assert np.allclose(result.get_greens_function().as_perturbed_mo_energy(), mf.mo_energy)


def test_ccsd(mf: scf.hf.RHF) -> None:
    """Test the CCSD expression."""
    ccsd_h = CCSD.h.from_mf(mf)
    ccsd_p = CCSD.p.from_mf(mf)
    pyscf_ccsd = pyscf.cc.CCSD(mf)
    pyscf_ccsd.run(conv_tol=1e-10, conv_tol_normt=1e-8)
    gf_moments_h = ccsd_h.build_gf_moments(2)
    gf_moments_p = ccsd_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = pyscf_ccsd.e_tot - mf.mol.energy_nuc()

    correct_energy = np.abs(energy - energy_ref) < 1e-8
    if mf.mol.nelectron > 2:
        # Galitskii--Migdal should not capture the energy for CCSD for >2 electrons
        with pytest.raises(AssertionError):
            assert correct_energy
    else:
        assert correct_energy

    # Get the Green's function from the Davidson solver
    davidson = Davidson.from_expression(ccsd_h, nroots=3)
    davidson.kernel()
    ip_ref, _ = pyscf_ccsd.ipccsd(nroots=3)

    assert davidson.result is not None
    assert np.allclose(davidson.result.eigvals[0], -ip_ref[-1])

    # Check the RDM
    rdm1 = gf_moments_h[0].copy()
    rdm1 += rdm1.T.conj()
    rdm1_ref = pyscf_ccsd.make_rdm1(with_mf=True)

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))


def test_fci(mf: scf.hf.RHF) -> None:
    """Test the FCI expression."""
    fci_h = FCI.h.from_mf(mf)
    fci_p = FCI.p.from_mf(mf)
    pyscf_fci = pyscf.fci.FCI(mf)
    gf_moments_h = fci_h.build_gf_moments(2)
    gf_moments_p = fci_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = pyscf_fci.kernel()[0] - mf.mol.energy_nuc()

    assert np.abs(energy - energy_ref) < 1e-8

    # Check the RDM
    rdm1 = fci_h.build_gf_moments(1)[0] * 2
    rdm1_ref = pyscf_fci.make_rdm1(pyscf_fci.ci, mf.mol.nao, mf.mol.nelectron)

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))

    if mf.mol.nelectron <= 2:
        # CCSD should match FCI for <=2 electrons for the hole moments
        ccsd = CCSD.h.from_mf(mf)
        gf_moments_ccsd = ccsd.build_gf_moments(2)

        assert np.allclose(gf_moments_ccsd[0], gf_moments_h[0])
        assert np.allclose(gf_moments_ccsd[1], gf_moments_h[1])


def test_adc2(mf: scf.hf.RHF) -> None:
    """Test the ADC(2) expression."""
    adc_h = ADC2.h.from_mf(mf)
    adc_p = ADC2.p.from_mf(mf)
    pyscf_adc = pyscf.adc.ADC(mf)
    gf_moments_h = adc_h.build_gf_moments(2)
    gf_moments_p = adc_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = mf.energy_elec()[0] + pyscf_adc.kernel_gs()[0]

    assert np.abs(energy - energy_ref) < 1e-8

    # Get the Green's function from the Davidson solver
    davidson = Davidson.from_expression(adc_h, nroots=3)
    davidson.kernel()
    ip_ref, _, _, _ = pyscf_adc.kernel(nroots=3)

    assert davidson.result is not None
    assert np.allclose(davidson.result.eigvals[0], -ip_ref[-1])

    # Check the RDM
    rdm1 = adc_h.build_gf_moments(1)[0] * 2
    rdm1_ref = np.diag(mf.mo_occ)  # No correlated ground state!

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))


def test_adc2x(mf: scf.hf.RHF) -> None:
    """Test the ADC(2)-x expression."""
    adc_h = ADC2x.h.from_mf(mf)
    adc_p = ADC2x.p.from_mf(mf)
    pyscf_adc = pyscf.adc.ADC(mf)
    pyscf_adc.method = "adc(2)-x"
    gf_moments_h = adc_h.build_gf_moments(2)
    gf_moments_p = adc_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = mf.energy_elec()[0] + pyscf_adc.kernel_gs()[0]

    assert np.abs(energy - energy_ref) < 1e-8

    # Get the Green's function from the Davidson solver
    davidson = Davidson.from_expression(adc_h, nroots=3)
    davidson.kernel()
    ip_ref, _, _, _ = pyscf_adc.kernel(nroots=3)

    assert davidson.result is not None
    assert np.allclose(davidson.result.eigvals[0], -ip_ref[-1])

    # Check the RDM
    rdm1 = adc_h.build_gf_moments(1)[0] * 2
    rdm1_ref = np.diag(mf.mo_occ)  # No correlated ground state!

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))


def test_tdagw(mf: scf.hf.RHF, exact_cache: ExactGetter) -> None:
    """Test the TDAGW expression."""
    tdagw = TDAGW["dyson"].from_mf(mf)
    dft = mf.to_rks()
    dft.xc = "hf"

    td = pyscf.tdscf.dTDA(dft)
    td.nstates = np.sum(mf.mo_occ > 0) * np.sum(mf.mo_occ == 0)
    td.kernel()
    td.xy = np.array([(x, np.zeros_like(x)) for x, y in td.xy])
    gw_obj = pyscf.gw.GW(dft, tdmf=td, freq_int="exact")
    gw_obj.kernel()

    # Get the IPs and EAs from the Exact solver
    solver = exact_cache(mf, TDAGW["dyson"])
    assert solver.result is not None
    gf = solver.result.get_greens_function()
    mo_energy = gf.as_perturbed_mo_energy()

    # No diagonal approximation in TDAGW so large error
    assert np.abs(mo_energy[tdagw.nocc - 1] - gw_obj.mo_energy[tdagw.nocc - 1]) < 1e-3
    assert np.abs(mo_energy[tdagw.nocc] - gw_obj.mo_energy[tdagw.nocc]) < 1e-3
