"""Tests for :module:`~dyson.results.static.chempot`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dyson.solvers import AufbauPrinciple, AuxiliaryShift
from dyson import numpy as np

from .conftest import _get_central_result

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import ExpressionCollection

    from .conftest import ExactGetter, Helper


@pytest.mark.parametrize("method", ["direct", "bisect", "global"])
def test_aufbau_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    method: str,
) -> None:
    """Test AufbauPrinciple compared to the exact solver."""
    result_exact = _get_central_result(
        helper, mf, expression_method, exact_cache, allow_hermitian=method == "global"
    )

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
    assert aufbau.result is not None

    # Get the Green's function and number of electrons
    greens_function = aufbau.result.get_greens_function()
    nelec: int = np.sum(greens_function.occupied().weights(2.0))

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
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
) -> None:
    """Test AuxiliaryShift compared to the exact solver."""
    result_exact = _get_central_result(
        helper, mf, expression_method, exact_cache, allow_hermitian=True
    )

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
        conv_tol=1e-9,
    )
    solver.kernel()
    assert solver.result is not None

    # Get the Green's function and number of electrons
    greens_function = solver.result.get_greens_function()
    nelec: int = np.sum(greens_function.occupied().weights(2.0))

    assert np.abs(mf.mol.nelectron - nelec) < 1e-7
