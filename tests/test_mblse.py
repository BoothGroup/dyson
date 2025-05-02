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
from .test_exact import (
    _compare_moments,
    _compare_static,
    _check_self_energy_to_greens_function,
    _check_central_greens_function_orthogonality,
)

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


def test_central_moments(mf: scf.hf.RHF, expression_method: type[BaseExpression]) -> None:
    """Test the recovery of the exact central moments from the MBLSE solver."""
    # Get the quantities required from the expression
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    gf_moments = expression_h.build_gf_moments(6) + expression_p.build_gf_moments(6)
    static, se_moments = util.gf_moments_to_se_moments(gf_moments, allow_non_identity=True)

    # Run the MBLSE solver
    solver = MBLSE(static, se_moments, hermitian=expression_h.hermitian)
    solver.kernel()

    # Recover the moments
    static_recovered = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    se_moments_recovered = self_energy.moments(range(4))

    assert _compare_static(static, static_recovered)
    assert _compare_moments(se_moments, se_moments_recovered)


def test_vs_exact_solver_central(
    mf: scf.hf.RHF, expression_method: dict[str, type[BaseExpression]]
) -> None:
    # Get the quantities required from the expressions
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = [expression_h.diagonal(), expression_p.diagonal()]
    hamiltonian = [expression_h.build_matrix(), expression_p.build_matrix()]
    bra = [
        np.array([expression_h.get_state_bra(i) for i in range(expression_h.nphys)]),
        np.array([expression_p.get_state_bra(i) for i in range(expression_p.nphys)]),
    ]
    ket = [
        np.array([expression_h.get_state_ket(i) for i in range(expression_h.nphys)]),
        np.array([expression_p.get_state_ket(i) for i in range(expression_p.nphys)]),
    ]

    # Context for non-Hermitian CCSD which currently doesn't recover orthogonality
    ctx = pytest.raises(AssertionError) if isinstance(expression_h, BaseCCSD) else nullcontext()

    # Solve the Hamiltonian exactly
    exact_h = Exact(hamiltonian[0], bra[0], ket[0], hermitian=expression_h.hermitian)
    exact_p = Exact(hamiltonian[1], bra[1], ket[1], hermitian=expression_p.hermitian)
    exact = Componentwise(exact_h, exact_p)
    exact.kernel()

    # Get the self-energy and Green's function from the exact solver
    static_exact = exact.get_static_self_energy()
    self_energy_exact = exact.get_self_energy()
    greens_function_exact = exact.get_greens_function()
    se_h_moments_exact = self_energy_exact.occupied().moments(range(4))
    se_p_moments_exact = self_energy_exact.virtual().moments(range(4))

    # Solve the Hamiltonian with MBLSE
    mblse_h = MBLSE(static_exact, se_h_moments_exact, hermitian=expression_h.hermitian)
    mblse_p = MBLSE(static_exact, se_p_moments_exact, hermitian=expression_p.hermitian)
    mblse = Componentwise(mblse_h, mblse_p)
    mblse.kernel()

    # Recover the hole self-energy and Green's function from the MBLSE solver
    static = mblse_h.get_static_self_energy()
    self_energy = mblse_h.get_self_energy()
    greens_function = mblse_h.get_greens_function()
    se_h_moments = self_energy.occupied().moments(range(4))

    assert _compare_static(static, static_exact)
    assert _compare_moments(se_h_moments, se_h_moments_exact)

    # Recover the particle self-energy and Green's function from the MBLSE solver
    static = mblse_p.get_static_self_energy()
    self_energy = mblse_p.get_self_energy()
    greens_function = mblse_p.get_greens_function()
    se_p_moments = self_energy.virtual().moments(range(4))

    print([util.scaled_error(a, b) for a, b in zip(se_h_moments, se_h_moments_exact)])
    print([util.scaled_error(a, b) for a, b in zip(se_p_moments, se_p_moments_exact)])
    assert _compare_static(static, static_exact)
    assert _compare_moments(se_p_moments, se_p_moments_exact)

    # Recover the self-energy and Green's function from the MBLSE solver
    static = mblse.get_static_self_energy()
    self_energy = mblse.get_self_energy()
    greens_function = mblse.get_greens_function()
    se_h_moments = self_energy.occupied().moments(range(4))
    se_p_moments = self_energy.virtual().moments(range(4))

    assert _compare_static(static, static_exact)
    assert _compare_moments(se_h_moments, se_h_moments_exact)
    assert _compare_moments(se_p_moments, se_p_moments_exact)
    assert _compare_moments(self_energy.moments(0), self_energy_exact.moments(0))
    with ctx:
        assert _compare_moments(greens_function.moments(0), greens_function_exact.moments(0))
    assert _check_self_energy_to_greens_function(static, self_energy, greens_function)
