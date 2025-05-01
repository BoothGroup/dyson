"""Tests for :class:`~dyson.expressions`."""

from __future__ import annotations

import itertools
import pytest
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


def test_init(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the instantiation of the expression from a mean-field object."""
    expression = expression_cls.from_mf(mf)
    assert expression.mol is mf.mol
    assert expression.nphys == mf.mol.nao
    assert expression.nocc == mf.mol.nelectron // 2
    assert expression.nvir == mf.mol.nao - mf.mol.nelectron // 2


def test_hamiltonian(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the Hamiltonian of the expression."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = expression.diagonal()
    hamiltonian = expression.build_matrix()

    assert np.allclose(np.diag(hamiltonian), diagonal)
    assert hamiltonian.shape == expression.shape
    assert (expression.nconfig + expression.nsingle) == diagonal.size


def test_gf_moments(mf: scf.hf.RHF, expression_cls: dict[str, type[BaseExpression]]) -> None:
    """Test the Green's function moments of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = expression.diagonal()
    hamiltonian = expression.build_matrix()

    # Construct the moments
    moments = np.zeros((2, expression.nphys, expression.nphys))
    for i, j in itertools.product(range(expression.nphys), repeat=2):
        bra = expression.get_state_bra(j)
        ket = expression.get_state_ket(i)
        moments[0, i, j] += bra.conj() @ ket
        moments[1, i, j] += np.einsum("j,i,ij->", bra.conj(), ket, hamiltonian)

    # Compare the moments to the reference
    ref = expression.build_gf_moments(2)

    assert np.allclose(ref[0], moments[0])
    assert np.allclose(ref[1], moments[1])
