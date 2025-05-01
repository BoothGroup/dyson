"""Tests for :module:`~dyson.solvers.static.davidson`."""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers import Davidson, Exact, Componentwise

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


def test_vs_exact_solver(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test Davidson compared to the exact solver."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 512:  # TODO: Make larger for CI runs
        pytest.skip("Skipping test for large Hamiltonian")
    if expression.nsingle == (expression.nocc + expression.nvir):
        pytest.skip("Skipping test for central Hamiltonian")
    diagonal = expression.diagonal()
    hamiltonian = expression.build_matrix()
    bra = np.array([expression.get_state_bra(i) for i in range(expression.nphys)])
    ket = np.array([expression.get_state_ket(i) for i in range(expression.nphys)])

    # Solve the Hamiltonian exactly
    exact = Exact(hamiltonian, bra, ket, hermitian=expression.hermitian)
    exact.kernel()

    # Solve the Hamiltonian with Davidson
    davidson = Davidson(
        expression.apply_hamiltonian,
        expression.diagonal(),
        bra,
        ket,
        nroots=expression.nsingle + expression.nconfig,  # Get all the roots
        hermitian=expression.hermitian,
    )
    davidson.kernel()

    assert davidson.matvec == expression.apply_hamiltonian
    assert np.all(davidson.diagonal == expression.diagonal())
    assert davidson.nphys == expression.nphys

    # Get the self-energy and Green's function from the Davidson solver
    static = davidson.get_static_self_energy()
    self_energy = davidson.get_self_energy()
    greens_function = davidson.get_greens_function()

    # Get the self-energy and Green's function from the exact solver
    static_exact = exact.get_static_self_energy()
    self_energy_exact = exact.get_self_energy()
    greens_function_exact = exact.get_greens_function()

    if expression.hermitian:
        # Left-handed eigenvectors not converged for non-Hermitian Davidson  # TODO
        assert np.allclose(static, static_exact)
        assert np.allclose(self_energy.moment(0), self_energy_exact.moment(0))
        assert np.allclose(self_energy.moment(1), self_energy_exact.moment(1))


def test_vs_exact_solver_central(
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

    # Solve the Hamiltonian exactly
    exact_h = Exact(hamiltonian[0], bra[0], ket[0], hermitian=expression_h.hermitian)
    exact_h.kernel()
    exact_p = Exact(hamiltonian[1], bra[1], ket[1], hermitian=expression_p.hermitian)
    exact_p.kernel()

    # Solve the Hamiltonian with Davidson
    davidson_h = Davidson(
        expression_h.apply_hamiltonian,
        diagonal[0],
        bra[0],
        ket[0],
        nroots=expression_h.nsingle + expression_h.nconfig,  # Get all the roots
        hermitian=expression_h.hermitian,
        conv_tol=1e-11,
        conv_tol_residual=1e-8,
    )
    davidson_h.kernel()
    davidson_p = Davidson(
        expression_p.apply_hamiltonian,
        diagonal[1],
        bra[1],
        ket[1],
        nroots=expression_p.nsingle + expression_p.nconfig,  # Get all the roots
        hermitian=expression_p.hermitian,
        conv_tol=1e-11,
        conv_tol_residual=1e-8,
    )
    davidson_p.kernel()

    # Get the self-energy and Green's function from the Davidson solver
    static = davidson_h.get_static_self_energy() + davidson_p.get_static_self_energy()
    self_energy = Lehmann.concatenate(
        davidson_h.get_self_energy(), davidson_p.get_self_energy()
    )
    greens_function = (
        Lehmann.concatenate(davidson_h.get_greens_function(), davidson_p.get_greens_function())
    )

    # Get the self-energy and Green's function from the exact solvers
    static_exact = exact_h.get_static_self_energy() + exact_p.get_static_self_energy()
    self_energy_exact = Lehmann.concatenate(
        exact_h.get_self_energy(), exact_p.get_self_energy()
    )
    greens_function_exact = (
        Lehmann.concatenate(exact_h.get_greens_function(), exact_p.get_greens_function())
    )

    if expression_h.hermitian and expression_p.hermitian:
        # Left-handed eigenvectors not converged for non-Hermitian Davidson  # TODO
        assert np.allclose(static, static_exact)
        assert np.allclose(self_energy.moment(0), self_energy_exact.moment(0))
        assert np.allclose(self_energy.moment(1), self_energy_exact.moment(1))

    # Use the component-wise solvers
    exact = Componentwise(exact_h, exact_p)
    exact.kernel()
    davidson = Componentwise(davidson_h, davidson_p)
    davidson.kernel()

    # Get the self-energy and Green's function from the Davidson solver
    static = davidson.get_static_self_energy()
    self_energy = davidson.get_self_energy()
    greens_function = davidson.get_greens_function()

    # Get the self-energy and Green's function from the exact solver
    static_exact = exact.get_static_self_energy()
    self_energy_exact = exact.get_self_energy()
    greens_function_exact = exact.get_greens_function()

    if expression_h.hermitian and expression_p.hermitian:
        # Left-handed eigenvectors not converged for non-Hermitian Davidson  # TODO
        assert np.allclose(greens_function.moment(0), np.eye(greens_function.nphys))
        assert np.allclose(greens_function_exact.moment(0), np.eye(greens_function.nphys))
        assert np.allclose(static, static_exact)
        assert np.allclose(self_energy.moment(0), self_energy_exact.moment(0))
        assert np.allclose(self_energy.moment(1), self_energy_exact.moment(1), atol=1e-5)
        assert np.allclose(greens_function.moment(1), greens_function_exact.moment(1), atol=1e-5)
