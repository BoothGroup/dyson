"""Tests for :mod:`~dyson.solvers.static.mblse`."""

from __future__ import annotations

from contextlib import nullcontext
import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers import MBLSE, Exact, Componentwise
from dyson.expressions.ccsd import BaseCCSD
from dyson.expressions.fci import BaseFCI

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
    """Test the recovery of the exact central moments from the MBLSE solver."""
    # Get the quantities required from the expression
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    nmom_gf = max_cycle * 2 + 4
    nmom_se = nmom_gf - 2
    gf_moments = expression_h.build_gf_moments(nmom_gf) + expression_p.build_gf_moments(nmom_gf)
    static, se_moments = util.gf_moments_to_se_moments(gf_moments, allow_non_identity=True)

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian and not (isinstance(expression_p, BaseFCI) and max_cycle > 1)

    # Run the MBLSE solver
    solver = MBLSE(static, se_moments, hermitian=hermitian)
    solver.kernel()

    # Recover the moments
    static_recovered = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()

    assert helper.are_equal_arrays(static, static_recovered)
    assert helper.have_equal_moments(se_moments, self_energy, nmom_se)


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
    nmom_se = max_cycle * 2 + 2

    # Check if we need a non-Hermitian solver
    hermitian = expression_h.hermitian and not (isinstance(expression_p, BaseFCI) and max_cycle > 1)

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method["1h"])
    exact_p = exact_cache(mf, expression_method["1p"])
    exact = Componentwise(exact_h, exact_p, shared_static=False)
    exact.kernel()

    # Get the self-energy and Green's function from the exact solver
    static_exact = exact.get_static_self_energy()
    self_energy_exact = exact.get_self_energy()
    greens_function_exact = exact.get_greens_function()
    se_h_moments_exact = self_energy_exact.occupied().moments(range(nmom_se))
    se_p_moments_exact = self_energy_exact.virtual().moments(range(nmom_se))

    # Solve the Hamiltonian with MBLSE
    mblse_h = MBLSE(static_exact, se_h_moments_exact, hermitian=hermitian)
    mblse_p = MBLSE(static_exact, se_p_moments_exact, hermitian=hermitian)
    mblse = Componentwise(mblse_h, mblse_p, shared_static=True)
    mblse.kernel()

    # Recover the hole self-energy and Green's function from the MBLSE solver
    static = mblse_h.get_static_self_energy()
    self_energy = mblse_h.get_self_energy()
    greens_function = mblse_h.get_greens_function()

    assert helper.are_equal_arrays(mblse_h.get_static_self_energy(), static_exact)
    assert helper.are_equal_arrays(static, static_exact)
    assert helper.have_equal_moments(self_energy, se_h_moments_exact, nmom_se)

    # Recover the particle self-energy and Green's function from the MBLSE solver
    static = mblse_p.get_static_self_energy()
    self_energy = mblse_p.get_self_energy()
    greens_function = mblse_p.get_greens_function()

    assert helper.are_equal_arrays(mblse_p.get_static_self_energy(), static_exact)
    assert helper.are_equal_arrays(static, static_exact)
    assert helper.have_equal_moments(self_energy, se_p_moments_exact, nmom_se)

    # Recover the self-energy and Green's function from the MBLSE solver
    static = mblse.get_static_self_energy()
    self_energy = mblse.get_self_energy()
    greens_function = mblse.get_greens_function()

    assert helper.are_equal_arrays(static, static_exact)
    assert helper.have_equal_moments(self_energy, se_h_moments_exact + se_p_moments_exact, nmom_se)
    assert helper.have_equal_moments(self_energy, self_energy_exact, nmom_se)
    assert helper.recovers_greens_function(static, self_energy, greens_function)
