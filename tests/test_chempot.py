"""Tests for :module:`~dyson.results.static.chempot`."""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.lehmann import Lehmann
from dyson.spectral import Spectral
from dyson.solvers import AufbauPrinciple, AuxiliaryShift, Exact

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression
    from .conftest import Helper, ExactGetter


@pytest.mark.parametrize("method", ["direct", "bisect", "global"])
def test_aufbau_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: dict[str, type[BaseExpression]],
    exact_cache: ExactGetter,
    method: str,
) -> None:
    """Test AufbauPrinciple compared to the exact solver."""
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    if not expression_h.hermitian and method != "global":
        pytest.skip("Skipping test for non-Hermitian Hamiltonian with negative weights")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_h)
    exact_p = exact_cache(mf, expression_p)
    result_exact = Spectral.combine(exact_h.result, exact_p.result)

    # Solve the Hamiltonian with AufbauPrinciple
    with pytest.raises(ValueError):
        # Needs nelec
        aufbau = AufbauPrinciple.from_self_energy(
            result_exact.get_static_self_energy(),
            result_exact.get_self_energy(),
            method=method,
        )
    aufbau = AufbauPrinciple.from_self_energy(
        result_exact.get_static_self_energy(),
        result_exact.get_self_energy(),
        nelec=mf.mol.nelectron,
        method=method,
    )
    aufbau.kernel()

    # Get the Green's function and number of electrons
    greens_function = aufbau.result.get_greens_function()
    nelec = np.sum(greens_function.occupied().weights(2.0))

    # Find the best number of electrons
    best = np.min(
        [
            np.abs(mf.mol.nelectron - np.sum(greens_function.mask(slice(i)).weights(2.0)))
            for i in range(greens_function.naux)
        ]
    )

    assert np.isclose(np.abs(mf.mol.nelectron - nelec), best)


def test_shift_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: dict[str, type[BaseExpression]],
    exact_cache: ExactGetter,
) -> None:
    """Test AuxiliaryShift compared to the exact solver."""
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_h)
    exact_p = exact_cache(mf, expression_p)
    result_exact = Spectral.combine(exact_h.result, exact_p.result)

    # Solve the Hamiltonian with AuxiliaryShift
    with pytest.raises(ValueError):
        # Needs nelec
        solver = AuxiliaryShift.from_self_energy(
            result_exact.get_static_self_energy(),
            result_exact.get_self_energy(),
        )
    solver = AuxiliaryShift.from_self_energy(
        result_exact.get_static_self_energy(),
        result_exact.get_self_energy(),
        nelec=mf.mol.nelectron,
    )
    solver.kernel()

    # Get the Green's function and number of electrons
    greens_function = solver.result.get_greens_function()
    nelec = np.sum(greens_function.occupied().weights(2.0))

    assert np.isclose(np.abs(mf.mol.nelectron - nelec), 0.0)
