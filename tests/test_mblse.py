"""Tests for :mod:`~dyson.solvers.static.mblse`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dyson import util
from dyson.expressions.fci import BaseFCI
from dyson.solvers import MBLSE
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_central_moments(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: dict[str, type[BaseExpression]],
    max_cycle: int,
) -> None:
    """Test the recovery of the exact central moments from the MBLSE solver."""
    # Get the quantities required from the expression
    if "1h" not in expression_method or "1p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    nmom_gf = max_cycle * 2 + 4
    nmom_se = nmom_gf - 2
    gf_moments = expression_h.build_gf_moments(nmom_gf) + expression_p.build_gf_moments(nmom_gf)
    static, se_moments = util.gf_moments_to_se_moments(gf_moments)

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian and not (isinstance(expression_p, BaseFCI) and max_cycle > 1)

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
@pytest.mark.parametrize("shared_static", [True, False])
def test_vs_exact_solver_central(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: dict[str, type[BaseExpression]],
    exact_cache: ExactGetter,
    max_cycle: int,
    shared_static: bool,
) -> None:
    # Get the quantities required from the expressions
    if "1h" not in expression_method or "1p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    nmom_se = max_cycle * 2 + 2

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian and not (isinstance(expression_p, BaseFCI) and max_cycle > 1)

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method["1h"])
    exact_p = exact_cache(mf, expression_method["1p"])
    assert exact_h.result is not None
    assert exact_p.result is not None
    result_exact_ph = Spectral.combine(exact_h.result, exact_p.result, shared_static=False)

    # Get the self-energy and Green's function from the exact solver
    static_exact = result_exact_ph.get_static_self_energy()
    self_energy_exact = result_exact_ph.get_self_energy()
    greens_function_exact = result_exact_ph.get_greens_function()
    static_h_exact = exact_h.result.get_static_self_energy()
    static_p_exact = exact_p.result.get_static_self_energy()
    se_h_moments_exact = exact_h.result.get_self_energy().moments(range(nmom_se))
    se_p_moments_exact = exact_p.result.get_self_energy().moments(range(nmom_se))

    # Solve the Hamiltonian with MBLSE
    mblse_h = MBLSE(
        static_h_exact if not shared_static else static_exact,
        se_h_moments_exact,
        hermitian=hermitian,
    )
    result_h = mblse_h.kernel()
    mblse_p = MBLSE(
        static_p_exact if not shared_static else static_exact,
        se_p_moments_exact,
        hermitian=hermitian,
    )
    result_p = mblse_p.kernel()
    result_ph = Spectral.combine(result_h, result_p, shared_static=shared_static)

    # Recover the self-energy and Green's function from the MBLSE solver
    static = result_ph.get_static_self_energy()
    self_energy = result_ph.get_self_energy()
    greens_function = result_ph.get_greens_function()

    assert helper.are_equal_arrays(static, static_exact)
    assert helper.have_equal_moments(self_energy, se_h_moments_exact + se_p_moments_exact, nmom_se)
    assert helper.have_equal_moments(self_energy, self_energy_exact, nmom_se)
    assert helper.recovers_greens_function(static, self_energy, greens_function, 4)
    assert helper.have_equal_moments(greens_function, greens_function_exact, nmom_se)
