"""Tests for :module:`~dyson.grids`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.grids import GridRF, GridIF, GridIT, transform
from dyson.representations.enums import Ordering, Reduction, Component

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


@pytest.mark.parametrize("ordering", [Ordering.RETARDED, Ordering.ADVANCED, Ordering.ORDERED])
@pytest.mark.parametrize("reduction", [Reduction.NONE, Reduction.DIAG, Reduction.TRACE])
@pytest.mark.parametrize("component", [Component.FULL, Component.REAL, Component.IMAG])
@pytest.mark.parametrize("beta", [16, 32, 64])
def test_fourier_transform_imag(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
    ordering: Ordering,
    reduction: Reduction,
    component: Component,
    beta: float,
) -> None:
    """Test Fourier transform between imaginary time and frequency."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:  # TODO: Make larger for CI runs?
        pytest.skip("Skipping test for large Hamiltonian")

    # Build the grids
    grid_if = GridIF.from_uniform(256, beta=beta)
    grid_it = GridIT.from_uniform(256, beta=beta)

    # Solve the Hamiltonian exactly
    exact = exact_cache(mf, expression_cls)
    assert exact.result is not None
    gf_if = grid_if.evaluate_lehmann(
        exact.result.get_greens_function(),
        ordering=ordering,
        reduction=reduction,
        component=component,
    )
    gf_it = grid_it.evaluate_lehmann(
        exact.result.get_greens_function(),
        ordering=ordering,
        reduction=reduction,
        component=component,
    )

    # Transform the Green's functions
    gf_it_recov = transform(gf_if, grid_it)
    gf_if_recov = transform(gf_it, grid_if)
