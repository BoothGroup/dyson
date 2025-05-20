"""Tests for :module:`~dyson.results.static.downfolded`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.solvers import Downfolded
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


def test_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: dict[str, type[BaseExpression]],
    exact_cache: ExactGetter,
) -> None:
    """Test Downfolded compared to the exact solver."""
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_h)
    exact_p = exact_cache(mf, expression_p)
    result_exact = Spectral.combine(exact_h.result, exact_p.result)

    # Solve the Hamiltonian with Downfolded
    downfolded = Downfolded.from_self_energy(
        result_exact.get_static_self_energy(),
        result_exact.get_self_energy(),
        eta=1e-9,
    )
    downfolded.kernel()

    # Get the targetted energies
    guess = downfolded.guess
    energy_downfolded = downfolded.result.eigvals[
        np.argmin(np.abs(downfolded.result.eigvals - guess))
    ]
    energy_exact = result_exact.eigvals[np.argmin(np.abs(result_exact.eigvals - energy_downfolded))]

    assert np.abs(energy_exact - energy_downfolded) < 1e-8
