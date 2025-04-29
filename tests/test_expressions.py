"""Tests for :class:`~dyson.expressions`."""

from __future__ import annotations

import itertools
import pytest
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


def test_init(mf: scf.hf.RHF, expressions: dict[str, type[BaseExpression]], sector: str) -> None:
    """Test the instantiation of the expression from a mean-field object."""
    expression = expressions[sector].from_mf(mf)
    assert expression.mol is mf.mol
    assert expression.nphys == mf.mol.nao
    assert expression.nocc == mf.mol.nelectron // 2
    assert expression.nvir == mf.mol.nao - mf.mol.nelectron // 2


def test_hamiltonian(
    mf: scf.hf.RHF, expressions: dict[str, type[BaseExpression]], sector: str
) -> None:
    """Test the Hamiltonian of the expression."""
    expression = expressions[sector].from_mf(mf)
    diagonal = expression.diagonal()
    if diagonal.size > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    hamiltonian = expression.build_matrix()

    assert np.allclose(np.diag(hamiltonian), diagonal)
    assert hamiltonian.shape == expression.shape


def test_gf_moments(mf: scf.hf.RHF, expressions: dict[str, type[BaseExpression]]) -> None:
    """Test the Green's function moments of the expression."""
    expression = (expressions["1h"].from_mf(mf), expressions["1p"].from_mf(mf))
    diagonal = (expression[0].diagonal(), expression[1].diagonal())
    if any(d.size > 1024 for d in diagonal):
        pytest.skip("Skipping test for large Hamiltonian")
    hamiltonian = (expression[0].build_matrix(), expression[1].build_matrix())

    moments = np.zeros((2, expression[0].nphys, expression[0].nphys))
    for i, j in itertools.product(range(expression[0].nphys), repeat=2):
        bra = expression[0].get_state_bra(j)
        ket = expression[0].get_state_ket(i)
        moments[0, i, j] += bra.conj() @ ket
        moments[1, i, j] += np.einsum("j,i,ij->", bra.conj(), ket, hamiltonian[0])
        bra = expression[1].get_state_bra(j)
        ket = expression[1].get_state_ket(i)
        moments[0, i, j] += bra.conj() @ ket
        moments[1, i, j] += np.einsum("j,i,ij->", bra.conj(), ket, hamiltonian[1])

    ref = expression[0].build_gf_moments(2) + expression[1].build_gf_moments(2)

    assert np.allclose(ref[0], moments[0])
    assert np.allclose(ref[1], moments[1])
