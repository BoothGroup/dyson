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
    from .conftest import Helper, ExactGetter


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
    solver.kernel()

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

    assert helper.are_equal_arrays(static, static_other)
    assert helper.have_equal_moments(self_energy, self_energy_other, 2)


def test_exact_solver_central(
    helper: Helper, mf: scf.hf.RHF, expression_method: dict[str, type[BaseExpression]]
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

    assert helper.has_orthonormal_couplings(greens_function)

    # Recover the Green's function from the recovered self-energy
    solver = Exact.from_self_energy(static, self_energy)
    solver.kernel()
    static_other = solver.get_static_self_energy()
    self_energy_other = solver.get_self_energy()
    greens_function_other = solver.get_greens_function()

    assert helper.have_equal_moments(greens_function, greens_function_other, 2)

    # Use the component-wise solver
    solver = Componentwise(solver_h, solver_p, shared_static=False)
    solver.kernel()

    # Get the self-energy and Green's function from the solvers
    static = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    greens_function = solver.get_greens_function()

    assert helper.are_equal_arrays(static, static_other)
    assert helper.have_equal_moments(self_energy, self_energy_other, 2)
    assert helper.have_equal_moments(greens_function, greens_function_other, 2)
    assert helper.are_equal_arrays(static, greens_function.moment(1))
    assert helper.has_orthonormal_couplings(greens_function)
    assert helper.recovers_greens_function(static, self_energy, greens_function)
