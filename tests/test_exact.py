"""Tests for :mod:`~dyson.solvers.static.exact`."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers import Exact, Componentwise
from dyson.expressions.ccsd import BaseCCSD

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.typing import Array
    from dyson.expressions.expression import BaseExpression


def _compare_moments(moments1: Array, moments2: Array, tol: float = 1e-8) -> bool:
    """Compare two sets of moments."""
    return all(util.scaled_error(m1, m2) < tol for m1, m2 in zip(moments1, moments2))


def _compare_static(static1: Array, static2: Array, tol: float = 1e-8) -> bool:
    """Compare two static self-energies."""
    return util.scaled_error(static1, static2) < tol


def _check_self_energy_to_greens_function(
    static: Array, self_energy: Lehmann, greens_function: Lehmann, tol: float = 1e-8
) -> None:
    """Check a self-energy recovers the Green's function."""
    greens_function_other = Lehmann(*self_energy.diagonalise_matrix_with_projection(static))
    moments = greens_function.moments(range(2))
    moments_other = greens_function_other.moments(range(2))
    return _compare_moments(moments, moments_other, tol=tol)


def _check_central_greens_function_orthogonality(greens_function: Lehmann, tol: float = 1e-8) -> bool:
    """Check the orthogonality of the central Green's function."""
    return _compare_moments(greens_function.moment(0), np.eye(greens_function.nphys), tol=tol)


def test_exact_solver(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the exact solver."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = expression.diagonal()
    hamiltonian = expression.build_matrix()
    bra = np.array([expression.get_state_bra(i) for i in range(expression.nphys)])
    ket = np.array([expression.get_state_ket(i) for i in range(expression.nphys)])

    # Solve the Hamiltonian
    solver = Exact(hamiltonian, bra, ket, hermitian=expression.hermitian)
    solver.kernel()

    assert solver.matrix is hamiltonian
    assert solver.bra is bra
    assert solver.ket is ket
    assert solver.nphys == expression.nphys
    assert solver.hermitian == expression.hermitian

    # Get the self-energy and Green's function from the solver
    static = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    greens_function = solver.get_greens_function()

    assert self_energy.nphys == expression.nphys
    assert greens_function.nphys == expression.nphys

    # Recover the Green's function from the recovered self-energy
    solver = Exact.from_self_energy(static, self_energy)
    solver.kernel()
    static_other = solver.get_static_self_energy()
    self_energy_other = solver.get_self_energy()
    greens_function_other = solver.get_greens_function()

    assert _compare_static(static, static_other)
    assert _compare_moments(self_energy.moments(range(2)), self_energy_other.moments(range(2)))


def test_exact_solver_central(
    mf: scf.hf.RHF, expression_method: dict[str, type[BaseExpression]]
) -> None:
    """Test the exact solver for central moments."""
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

    # Solve the Hamiltonians
    solver_h = Exact(hamiltonian[0], bra[0], ket[0], hermitian=expression_h.hermitian)
    solver_h.kernel()
    solver_p = Exact(hamiltonian[1], bra[1], ket[1], hermitian=expression_p.hermitian)
    solver_p.kernel()

    # Get the self-energy and Green's function from the solvers
    static = solver_h.get_static_self_energy() + solver_p.get_static_self_energy()
    self_energy = Lehmann.concatenate(solver_h.get_self_energy(), solver_p.get_self_energy())
    greens_function = Lehmann.concatenate(
        solver_h.get_greens_function(), solver_p.get_greens_function()
    )

    with ctx:
        assert _check_central_greens_function_orthogonality(greens_function)

    # Recover the Green's function from the recovered self-energy
    solver = Exact.from_self_energy(static, self_energy)
    solver.kernel()
    static_other = solver.get_static_self_energy()
    self_energy_other = solver.get_self_energy()
    greens_function_other = solver.get_greens_function()

    with ctx:
        assert _compare_moments(greens_function.moments(range(2)), greens_function_other.moments(range(2)))

    # Use the component-wise solver
    solver = Componentwise(solver_h, solver_p)
    solver.kernel()

    # Get the self-energy and Green's function from the solvers
    static = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    greens_function = solver.get_greens_function()

    assert _compare_static(static, static_other)
    assert _compare_static(static, greens_function.moment(1))
    assert _compare_moments(self_energy.moments(range(2)), self_energy_other.moments(range(2)))
    with ctx:
        assert _check_central_greens_function_orthogonality(greens_function)
    with ctx:
        assert _compare_moments(greens_function.moments(range(2)), greens_function_other.moments(range(2)))
    with ctx:
        assert _check_self_energy_to_greens_function(static, self_energy, greens_function)
