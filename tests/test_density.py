"""Tests for :module:`~dyson.results.static.density`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dyson import numpy as np
from dyson.representations.spectral import Spectral
from dyson.solvers import DensityRelaxation
from dyson.solvers.static.density import get_fock_matrix_function

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import ExpressionCollection

    from .conftest import ExactGetter, Helper


def test_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
) -> None:
    """Test DensityRelaxation compared to the exact solver."""
    if "h" not in expression_method or "p" not in expression_method:
        pytest.skip("Skipping test for Dyson only expression")
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None
    result_exact = Spectral.combine(exact_h.result, exact_p.result)

    # Solve the Hamiltonian with DensityRelaxation
    get_fock = get_fock_matrix_function(mf)
    solver = DensityRelaxation.from_self_energy(
        result_exact.get_static_self_energy(),
        result_exact.get_self_energy(),
        nelec=mf.mol.nelectron,
        get_static=get_fock,
    )
    solver.kernel()
    assert solver.result is not None

    # Get the Green's function
    greens_function = solver.result.get_greens_function()
    rdm1 = greens_function.occupied().moment(0) * 2.0

    assert solver.converged
    assert np.isclose(np.trace(rdm1), mf.mol.nelectron, atol=1e-2)
