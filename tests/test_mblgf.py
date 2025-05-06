"""Tests for :mod:`~dyson.solvers.static.mblgf`."""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers import MBLGF, Exact, Componentwise
from dyson.expressions.ccsd import BaseCCSD

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression
    from .conftest import Helper, ExactGetter


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
    static = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    greens_function = solver.get_greens_function()

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
    # Get the quantities required from the expressions
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    nmom_gf = max_cycle * 2 + 2

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method["1h"])
    exact_p = exact_cache(mf, expression_method["1p"])
    exact = Componentwise(exact_h, exact_p, shared_static=False)
    exact.kernel()

    # Get the self-energy and Green's function from the exact solver
    greens_function_exact = exact.get_greens_function()
    gf_h_moments_exact = greens_function_exact.occupied().moments(range(nmom_gf))
    gf_p_moments_exact = greens_function_exact.virtual().moments(range(nmom_gf))

    # Solve the Hamiltonian with MBLGF
    mblgf_h = MBLGF(gf_h_moments_exact, hermitian=expression_h.hermitian)
    mblgf_p = MBLGF(gf_p_moments_exact, hermitian=expression_p.hermitian)
    mblgf = Componentwise(mblgf_h, mblgf_p, shared_static=False)
    mblgf.kernel()

    # Recover the hole Green's function from the MBLGF solver
    greens_function = mblgf_h.get_greens_function()

    #np.set_printoptions(precision=4, suppress=True, linewidth=110)
    #print(greens_function.moments(0).real)
    #print(gf_h_moments_exact[0].real)
    assert helper.have_equal_moments(greens_function, gf_h_moments_exact, nmom_gf)

    # Recover the particle Green's function from the MBLGF solver
    greens_function = mblgf_p.get_greens_function()

    assert helper.have_equal_moments(greens_function, gf_p_moments_exact, nmom_gf)

    # Recover the self-energy and Green's function from the recovered MBLGF solver
    static = mblgf.get_static_self_energy()
    self_energy = mblgf.get_self_energy()
    greens_function = mblgf.get_greens_function()

    assert helper.are_equal_arrays(static, exact.get_static_self_energy())
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    print(mblgf_h.get_self_energy().moments(0))
    print(mblgf_p.get_self_energy().moments(0))
    print(self_energy.moments(0))
    print(exact.get_self_energy().occupied().moments(0))
    print(exact.get_self_energy().virtual().moments(0))
    print(exact.get_self_energy().moments(0))
    print([util.scaled_error(a, b) for a, b in zip(self_energy.moments(range(nmom_gf-2)), exact.get_self_energy().moments(range(nmom_gf-2)))])
    assert helper.have_equal_moments(self_energy, exact.get_self_energy(), nmom_gf - 2)
    print(greens_function.moments(1))
    print(gf_h_moments_exact[1] + gf_p_moments_exact[1])
    print(greens_function.moments(2))
    print(gf_h_moments_exact[2] + gf_p_moments_exact[2])
    print([util.scaled_error(a, b) for a, b in zip(greens_function.moments(range(nmom_gf)), gf_h_moments_exact + gf_p_moments_exact)])
    assert helper.have_equal_moments(greens_function, gf_h_moments_exact + gf_p_moments_exact, nmom_gf)
    assert helper.have_equal_moments(greens_function, greens_function_exact, nmom_gf)
    assert helper.recovers_greens_function(static, self_energy, greens_function)
