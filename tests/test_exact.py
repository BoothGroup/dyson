"""Tests for :mod:`~dyson.solvers.static.exact`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dyson.solvers import Exact
from dyson.representations.spectral import Spectral

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression, ExpressionCollection

    from .conftest import ExactGetter, Helper


def test_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test the exact solver."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian
    solver = exact_cache(mf, expression_cls)

    assert solver.result is not None
    assert solver.nphys == expression.nphys
    assert solver.hermitian == expression.hermitian

    # Get the self-energy and Green's function from the solver
    static = solver.result.get_static_self_energy()
    self_energy = solver.result.get_self_energy()
    greens_function = solver.result.get_greens_function()

    assert self_energy.nphys == expression.nphys
    assert greens_function.nphys == expression.nphys

    # Recover the Green's function from the recovered self-energy
    overlap = greens_function.moment(0)
    solver = Exact.from_self_energy(static, self_energy, overlap=overlap)
    solver.kernel()
    assert solver.result is not None
    static_other = solver.result.get_static_self_energy()
    self_energy_other = solver.result.get_self_energy()
    greens_function_other = solver.result.get_greens_function()

    assert helper.are_equal_arrays(static, static_other)
    assert helper.have_equal_moments(self_energy, self_energy_other, 4)
    assert helper.have_equal_moments(greens_function, greens_function_other, 4)


def test_vs_exact_solver_central(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
) -> None:
    """Test the exact solver for central moments."""
    # Get the quantities required from the expressions
    if "h" not in expression_method or "p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None
    result_ph = Spectral.combine(exact_h.result, exact_p.result)

    # Recover the hole self-energy and Green's function
    static = exact_h.result.get_static_self_energy()
    self_energy = exact_h.result.get_self_energy()
    greens_function = exact_h.result.get_greens_function()

    assert helper.recovers_greens_function(static, self_energy, greens_function, 4)

    # Recover the particle self-energy and Green's function
    static = exact_p.result.get_static_self_energy()
    self_energy = exact_p.result.get_self_energy()
    greens_function = exact_p.result.get_greens_function()

    assert helper.recovers_greens_function(static, self_energy, greens_function, 4)

    # Recover the self-energy and Green's function
    static = result_ph.get_static_self_energy()
    self_energy = result_ph.get_self_energy()
    greens_function = result_ph.get_greens_function()

    assert helper.has_orthonormal_couplings(greens_function)
    assert helper.recovers_greens_function(static, self_energy, greens_function, 4)
