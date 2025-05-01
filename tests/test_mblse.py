"""Tests for :mod:`~dyson.solvers.static.mblse`."""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.lehmann import Lehmann
from dyson.solvers import MBLSE, Exact

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


def test_central_moments(mf: scf.hf.RHF, expression_method: type[BaseExpression]) -> None:
    """Test the recovery of the exact central moments from the MBLSE solver."""
    # Get the quantities required from the expression
    expression_h = expression_method["1h"].from_mf(mf)
    expression_p = expression_method["1p"].from_mf(mf)
    gf_moments = expression_h.build_gf_moments(6) + expression_p.build_gf_moments(6)
    static, se_moments = util.gf_moments_to_se_moments(gf_moments, allow_non_identity=True)

    # Run the MBLSE solver
    solver = MBLSE(static, se_moments, hermitian=expression_h.hermitian)
    solver.kernel()

    # Recover the moments
    static_recovered = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    se_moments_recovered = self_energy.moments(range(4))

    assert np.allclose(static, static_recovered)
    assert np.allclose(se_moments[0], se_moments_recovered[0])
    assert np.allclose(se_moments[1], se_moments_recovered[1])
    assert np.allclose(se_moments[2], se_moments_recovered[2], atol=1e-4)
    assert np.allclose(se_moments[3], se_moments_recovered[3], atol=1e-4)
