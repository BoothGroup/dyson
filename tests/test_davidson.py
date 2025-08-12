"""Tests for :module:`~dyson.results.static.davidson`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dyson import numpy as np
from dyson.representations.lehmann import Lehmann
from dyson.representations.spectral import Spectral
from dyson.solvers import Davidson

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression, ExpressionCollection
    from dyson.typing import Array

    from .conftest import ExactGetter, Helper


def test_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test Davidson compared to the exact solver."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:  # TODO: Make larger for CI runs?
        pytest.skip("Skipping test for large Hamiltonian")
    if expression.nsingle == (expression.nocc + expression.nvir):
        pytest.skip("Skipping test for central Hamiltonian")
    bra: Array = np.array(expression.get_excitation_bras())
    ket: Array = np.array(expression.get_excitation_kets())

    # Solve the Hamiltonian exactly
    exact = exact_cache(mf, expression_cls)
    assert exact.result is not None

    # Solve the Hamiltonian with Davidson
    davidson = Davidson(
        expression.apply_hamiltonian,
        expression.diagonal(),
        bra,
        ket,
        nroots=expression.nsingle + expression.nconfig,  # Get all the roots
        hermitian=expression.hermitian_upfolded,
    )
    davidson.kernel()
    assert davidson.result is not None

    assert davidson.matvec == expression.apply_hamiltonian
    assert np.all(davidson.diagonal == expression.diagonal())
    assert davidson.nphys == expression.nphys
    assert exact.matrix.shape == (davidson.nroots, davidson.nroots)

    # Get the self-energy and Green's function from the Davidson solver
    static = davidson.result.get_static_self_energy()
    self_energy = davidson.result.get_self_energy()
    greens_function = davidson.result.get_greens_function()

    # Get the self-energy and Green's function from the exact solver
    static_exact = exact.result.get_static_self_energy()
    self_energy_exact = exact.result.get_self_energy()
    greens_function_exact = exact.result.get_greens_function()

    if expression.hermitian_upfolded:
        # Left-handed eigenvectors not converged for non-Hermitian Davidson  # TODO
        assert helper.are_equal_arrays(static, static_exact)
        assert helper.have_equal_moments(self_energy, self_energy_exact, 4)
        assert helper.have_equal_moments(greens_function, greens_function_exact, 4)


def test_vs_exact_solver_central(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
) -> None:
    """Test the exact solver for central moments."""
    # Get the quantities required from the expressions
    if "o" not in expression_method or "p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = [expression_h.diagonal(), expression_p.diagonal()]
    bra = (
        np.array(expression_h.get_excitation_bras()),
        np.array(expression_p.get_excitation_bras()),
    )
    ket = (
        np.array(expression_h.get_excitation_kets()),
        np.array(expression_p.get_excitation_kets()),
    )

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None

    # Solve the Hamiltonian with Davidson
    davidson_h = Davidson(
        expression_h.apply_hamiltonian,
        diagonal[0],
        bra[0],
        ket[0],
        nroots=expression_h.nsingle + expression_h.nconfig,  # Get all the roots
        hermitian=expression_h.hermitian_upfolded,
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
        hermitian=expression_p.hermitian_upfolded,
        conv_tol=1e-11,
        conv_tol_residual=1e-8,
    )
    davidson_p.kernel()
    assert davidson_h.result is not None
    assert davidson_p.result is not None

    # Get the self-energy and Green's function from the Davidson solver
    static = davidson_h.result.get_static_self_energy() + davidson_p.result.get_static_self_energy()
    self_energy = Lehmann.concatenate(
        davidson_h.result.get_self_energy(), davidson_p.result.get_self_energy()
    )
    greens_function = Lehmann.concatenate(
        davidson_h.result.get_greens_function(), davidson_p.result.get_greens_function()
    )

    # Get the self-energy and Green's function from the exact solvers
    static_exact = exact_h.result.get_static_self_energy() + exact_p.result.get_static_self_energy()
    self_energy_exact = Lehmann.concatenate(
        exact_h.result.get_self_energy(), exact_p.result.get_self_energy()
    )
    greens_function_exact = Lehmann.concatenate(
        exact_h.result.get_greens_function(), exact_p.result.get_greens_function()
    )

    if expression_h.hermitian_upfolded and expression_p.hermitian_upfolded:
        # Left-handed eigenvectors not converged for non-Hermitian Davidson  # TODO
        assert helper.are_equal_arrays(static, static_exact)
        assert helper.have_equal_moments(self_energy, self_energy_exact, 2)

    # Use the component-wise solvers
    result_exact = Spectral.combine(exact_h.result, exact_p.result)
    result_davidson = Spectral.combine(davidson_h.result, davidson_p.result)

    # Get the self-energy and Green's function from the Davidson solver
    static = result_davidson.get_static_self_energy()
    self_energy = result_davidson.get_self_energy()
    greens_function = result_davidson.get_greens_function()

    # Get the self-energy and Green's function from the exact solver
    static_exact = result_exact.get_static_self_energy()
    self_energy_exact = result_exact.get_self_energy()
    greens_function_exact = result_exact.get_greens_function()

    if expression_h.hermitian_upfolded and expression_p.hermitian_upfolded:
        # Left-handed eigenvectors not converged for non-Hermitian Davidson  # TODO
        assert helper.are_equal_arrays(static, static_exact)
        assert helper.have_equal_moments(self_energy, self_energy_exact, 2)
        assert helper.are_equal_arrays(greens_function.moment(1), static)
        assert helper.are_equal_arrays(greens_function_exact.moment(1), static_exact)
        assert helper.recovers_greens_function(static, self_energy, greens_function)
        assert helper.recovers_greens_function(
            static_exact, self_energy_exact, greens_function_exact
        )
        assert helper.has_orthonormal_couplings(greens_function)
        assert helper.has_orthonormal_couplings(greens_function_exact)
