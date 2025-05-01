"""Tests for :mod:`~dyson.solvers.static.exact`."""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers import Exact, Componentwise
from dyson.expressions.ccsd import BaseCCSD

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


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

    assert np.allclose(static, static_other)
    assert np.allclose(self_energy.moment(0), self_energy_other.moment(0))
    assert np.allclose(self_energy.moment(1), self_energy_other.moment(1))


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

    if isinstance(expression_h, BaseCCSD):
        # Needs additional biorthogonalisation
        with pytest.raises(AssertionError):
            assert np.allclose(greens_function.moment(0), np.eye(greens_function.nphys))
    else:
        assert np.allclose(greens_function.moment(0), np.eye(greens_function.nphys))

    # Recover the Green's function from the recovered self-energy
    solver = Exact.from_self_energy(static, self_energy)
    solver.kernel()
    greens_function_other = solver.get_greens_function()

    if isinstance(expression_h, BaseCCSD):
        # Needs additional biorthogonalisation
        with pytest.raises(AssertionError):
            assert np.allclose(greens_function.moment(0), greens_function_other.moment(0))
            assert np.allclose(greens_function.moment(1), greens_function_other.moment(1))
    else:
        assert np.allclose(greens_function.moment(0), greens_function_other.moment(0))
        assert np.allclose(greens_function.moment(1), greens_function_other.moment(1))

    # Use the component-wise solver to do the same plus orthogonalise in the full space
    solver = Componentwise(solver_h, solver_p)
    solver.kernel()

    # Get the self-energy and Green's function from the solvers
    static = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    greens_function = solver.get_greens_function()

    assert np.allclose(greens_function.moment(0), np.eye(greens_function.nphys))

    # Recover the Green's function from the self-energy
    greens_function_other = Lehmann(*self_energy.diagonalise_matrix_with_projection(static))

    assert np.allclose(greens_function.moment(0), greens_function_other.moment(0))
    assert np.allclose(greens_function.moment(1), greens_function_other.moment(1))
