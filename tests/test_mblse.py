"""Tests for :mod:`~dyson.solvers.static.mblse`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson import util
from dyson.representations.spectral import Spectral
from dyson.solvers import MBLSE, MLSE
from dyson.expressions.hf import HF

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import ExpressionCollection

    from .conftest import ExactGetter, Helper


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_central_moments(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    max_cycle: int,
) -> None:
    """Test the recovery of the exact central moments from the MBLSE solver."""
    # Get the quantities required from the expression
    if "h" not in expression_method or "p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    nmom_gf = max_cycle * 2 + 4
    nmom_se = nmom_gf - 2
    gf_moments = expression_h.build_gf_moments(nmom_gf) + expression_p.build_gf_moments(nmom_gf)
    static, se_moments = util.gf_moments_to_se_moments(gf_moments)

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian_downfolded and expression_p.hermitian_downfolded

    # Run the MBLSE solver
    solver = MBLSE(static, se_moments, hermitian=hermitian)
    solver.kernel()
    assert solver.result is not None

    # Recover the moments
    static_recovered = solver.result.get_static_self_energy()
    self_energy = solver.result.get_self_energy()

    assert helper.are_equal_arrays(static, static_recovered)
    assert helper.have_equal_moments(se_moments, self_energy, nmom_se)


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_vs_exact_solver_central(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    max_cycle: int,
) -> None:
    # Get the quantities required from the expressions
    if "h" not in expression_method or "p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    nmom_se = max_cycle * 2 + 2

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian_downfolded and expression_p.hermitian_downfolded

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None
    result_exact_ph = Spectral.combine(exact_h.result, exact_p.result)

    # Get the self-energy and Green's function from the exact solver
    static_exact = result_exact_ph.get_static_self_energy()
    self_energy_exact = result_exact_ph.get_self_energy()
    greens_function_exact = result_exact_ph.get_greens_function()
    static_h_exact = exact_h.result.get_static_self_energy()
    static_p_exact = exact_p.result.get_static_self_energy()
    se_h_moments_exact = exact_h.result.get_self_energy().moments(range(nmom_se))
    se_p_moments_exact = exact_p.result.get_self_energy().moments(range(nmom_se))
    overlap_h = exact_h.result.get_overlap()
    overlap_p = exact_p.result.get_overlap()

    # Solve the Hamiltonian with MBLSE
    mblse_h = MBLSE(
        static_h_exact,
        se_h_moments_exact,
        overlap=overlap_h,
        hermitian=hermitian,
    )
    result_h = mblse_h.kernel()
    mblse_p = MBLSE(
        static_p_exact,
        se_p_moments_exact,
        overlap=overlap_p,
        hermitian=hermitian,
    )
    result_p = mblse_p.kernel()
    result_ph = Spectral.combine(result_h, result_p)

    # Recover the self-energy and Green's function from the MBLSE solver
    static = result_ph.get_static_self_energy()
    self_energy = result_ph.get_self_energy()
    greens_function = result_ph.get_greens_function()

    assert helper.are_equal_arrays(static, static_exact)
    assert helper.have_equal_moments(self_energy, se_h_moments_exact + se_p_moments_exact, nmom_se)
    assert helper.have_equal_moments(self_energy, self_energy_exact, nmom_se)
    assert helper.recovers_greens_function(static, self_energy, greens_function, 4)
    assert helper.have_equal_moments(greens_function, greens_function_exact, nmom_se)


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_mlse(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    max_cycle: int,
) -> None:
    """Test MLSE solver against MBLSE solver."""
    # Get the quantities required from the expression
    if "h" not in expression_method or "p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    if expression_method == HF:
        pytest.skip("Skipping test for HF expression, numerical issues for MLSE")
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    nmom_gf = max_cycle * 2 + 4
    nmom_se = nmom_gf - 2
    gf_moments = expression_h.build_gf_moments(nmom_gf) + expression_p.build_gf_moments(nmom_gf)
    static, se_moments = util.gf_moments_to_se_moments(gf_moments)
    gf_moments = util.einsum("...ij,ij->...ij", gf_moments, np.eye(gf_moments.shape[-1]))
    se_moments = util.einsum("...ij,ij->...ij", se_moments, np.eye(se_moments.shape[-1]))
    static = gf_moments[1]

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian_downfolded and expression_p.hermitian_downfolded

    # Run the MBLSE solver
    solver = MBLSE(static, se_moments, hermitian=hermitian)
    solver.kernel()
    assert solver.result is not None

    # Recover the moments
    static_mblse = solver.result.get_static_self_energy()
    moments_mblse = solver.result.get_self_energy().moments(range(nmom_se))

    # Run the MLSE solver
    for i in range(gf_moments.shape[-1]):
        solver_mlse = MLSE(static[i, i], se_moments[:, i, i], hermitian=True)
        solver_mlse.kernel()

        # Recover the moments
        static_mlse = solver_mlse.result.get_static_self_energy()
        moments_mlse = solver_mlse.result.get_self_energy().moments(range(nmom_se))

        assert helper.are_equal_arrays(static_mblse[i, i], static_mlse)
        assert helper.have_equal_moments(moments_mblse[:, i, i], moments_mlse, nmom_se)
