"""Tests for :mod:`~dyson.solvers.static.mblgf`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dyson import util
from dyson.solvers import MBLGF
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_central_moments(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: type[BaseExpression],
    max_cycle: int,
) -> None:
    """Test the recovery of the exact central moments from the MBLGF solver."""
    # Get the quantities required from the expression
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    nmom_gf = max_cycle * 2 + 2
    nmom_se = nmom_gf - 2
    gf_moments = expression_h.build_gf_moments(nmom_gf) + expression_p.build_gf_moments(nmom_gf)
    se_static, se_moments = util.gf_moments_to_se_moments(gf_moments)

    # Run the MBLGF solver
    solver = MBLGF(gf_moments, hermitian=expression_h.hermitian)
    solver.kernel()

    # Recover the Green's function and self-energy
    static = solver.result.get_static_self_energy()
    self_energy = solver.result.get_self_energy()
    greens_function = solver.result.get_greens_function()

    assert helper.have_equal_moments(greens_function, gf_moments, nmom_gf)
    assert helper.have_equal_moments(static, se_static, nmom_se)
    assert helper.have_equal_moments(self_energy, se_moments, nmom_se)


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_vs_exact_solver_central(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: dict[str, type[BaseExpression]],
    exact_cache: ExactGetter,
    max_cycle: int,
) -> None:
    """Test the MBLGF solver for central moments."""
    # Get the quantities required from the expressions
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    nmom_gf = max_cycle * 2 + 2

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method["1h"])
    exact_p = exact_cache(mf, expression_method["1p"])
    result_exact_ph = Spectral.combine(exact_h.result, exact_p.result, shared_static=False)

    # Get the self-energy and Green's function from the exact solver
    static_exact = result_exact_ph.get_static_self_energy()
    self_energy_exact = result_exact_ph.get_self_energy()
    greens_function_exact = result_exact_ph.get_greens_function()
    gf_h_moments_exact = exact_h.result.get_greens_function().moments(range(nmom_gf))
    gf_p_moments_exact = exact_p.result.get_greens_function().moments(range(nmom_gf))

    # Solve the Hamiltonian with MBLGF
    mblgf_h = MBLGF(gf_h_moments_exact, hermitian=expression_h.hermitian)
    mblgf_h.kernel()
    mblgf_p = MBLGF(gf_p_moments_exact, hermitian=expression_p.hermitian)
    mblgf_p.kernel()
    result_ph = Spectral.combine(mblgf_h.result, mblgf_p.result, shared_static=False)

    assert helper.have_equal_moments(
        mblgf_h.result.get_self_energy(), exact_h.result.get_self_energy(), nmom_gf - 2
    )
    assert helper.have_equal_moments(
        mblgf_p.result.get_self_energy(), exact_p.result.get_self_energy(), nmom_gf - 2
    )

    # Recover the hole Green's function from the MBLGF solver
    greens_function = mblgf_h.result.get_greens_function()

    assert helper.have_equal_moments(greens_function, gf_h_moments_exact, nmom_gf)

    # Recover the particle Green's function from the MBLGF solver
    greens_function = mblgf_p.result.get_greens_function()

    assert helper.have_equal_moments(greens_function, gf_p_moments_exact, nmom_gf)

    # Recover the self-energy and Green's function from the recovered MBLGF solver
    static = result_ph.get_static_self_energy()
    self_energy = result_ph.get_self_energy()
    greens_function = result_ph.get_greens_function()

    assert helper.are_equal_arrays(static, static_exact)
    assert helper.have_equal_moments(self_energy, self_energy_exact, nmom_gf - 2)
    assert helper.have_equal_moments(greens_function, greens_function_exact, nmom_gf)
    assert helper.recovers_greens_function(static, self_energy, greens_function, 4)
