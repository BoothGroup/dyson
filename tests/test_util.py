"""Tests for :module:`~dyson.util`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyscf
import pytest

from dyson import util

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


def test_moments_conversion(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test the conversion of moments between self-energy and Green's function."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian
    solver = exact_cache(mf, expression_cls)

    assert solver.result is not None
    assert solver.nphys == expression.nphys
    assert solver.hermitian == expression.hermitian

    # Get the self-energy and Green's function from the solver
    static = solver.result.get_static_self_energy()
    self_energy = solver.result.get_self_energy()
    greens_function = solver.result.get_greens_function()

    assert self_energy.nphys == expression.nphys
    assert greens_function.nphys == expression.nphys

    # Get the moments from the self-energy and Green's function
    se_moments = self_energy.moments(range(4))
    gf_moments = greens_function.moments(range(6))

    # Recover the self-energy from the Green's function moments
    static_other, se_moments_other = util.gf_moments_to_se_moments(gf_moments)
    gf_moments_other = util.se_moments_to_gf_moments(static, se_moments, overlap=gf_moments[0])

    assert helper.are_equal_arrays(static, static_other)
    if expression.hermitian:
        assert helper.have_equal_moments(se_moments, se_moments_other, 4)
        assert helper.have_equal_moments(gf_moments, gf_moments_other, 6)
    else:
        assert helper.have_equal_moments(se_moments, se_moments_other, 4, tol=5e-7)
        assert helper.have_equal_moments(gf_moments, gf_moments_other, 6, tol=5e-7)
