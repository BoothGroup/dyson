"""Tests for :module:`~dyson.solvers.dynamic.cpgf`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.lehmann import Lehmann
from dyson.solvers import CPGF
from dyson.spectral import Spectral
from dyson.grids import RealFrequencyGrid
from dyson.expressions.hf import BaseHF

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression, ExpressionCollection

    from .conftest import ExactGetter, Helper


def test_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test CPGF compared to the exact solver."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:  # TODO: Make larger for CI runs?
        pytest.skip("Skipping test for large Hamiltonian")
    if expression.nsingle == (expression.nocc + expression.nvir):
        pytest.skip("Skipping test for central Hamiltonian")
    if isinstance(expression, BaseHF):
        pytest.skip("Skipping test for HF Hamiltonian")
    grid = RealFrequencyGrid.from_uniform(-2, 2, 16, 0.1)

    # Solve the Hamiltonian exactly
    exact = exact_cache(mf, expression_cls)
    assert exact.result is not None
    gf_exact = grid.evaluate_lehmann(exact.result.get_greens_function(), ordering="advanced")

    # Solve the Hamiltonian with CorrectionVector
    cpgf = CPGF.from_self_energy(
        exact.result.get_static_self_energy(),
        exact.result.get_self_energy(),
        overlap=exact.result.get_greens_function().moment(0),
        grid=grid,
        max_cycle=512,
        ordering="advanced",
    )
    gf = cpgf.kernel()

    assert np.allclose(gf, gf_exact)
