"""Tests for :class:`~dyson.solvers.static.exact`."""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

import numpy as np

from dyson import util
from dyson.solvers import Exact

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression


def test_exact_solver(
    mf: scf.hf.RHF, expressions: dict[str, type[BaseExpression]], sector: str
) -> None:
    """Test the exact solver."""
    expression = expressions[sector].from_mf(mf)
    diagonal = expression.diagonal()
    if diagonal.size > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    hamiltonian = expression.build_matrix()

    solver = Exact(hamiltonian, expression.nphys, hermitian=expression.hermitian)
    solver.kernel()

    eigvals, eigvecs = util.eig_biorth(hamiltonian, hermitian=expression.hermitian)

    assert solver.matrix is hamiltonian
    assert solver.nphys == expression.nphys
    assert solver.hermitian == expression.hermitian
    assert np.allclose(solver.get_eigenfunctions(unpack=True)[0], eigvals)
    assert np.allclose(solver.get_eigenfunctions(unpack=True)[1][0], eigvecs[0])
    assert np.allclose(solver.get_eigenfunctions(unpack=True)[1][1], eigvecs[1])

    eigvals, eigvecs = solver.get_eigenfunctions(unpack=True)
    matrix_reconstructed = (eigvecs[1] * eigvals[None]) @ eigvecs[0].T.conj()

    assert np.allclose(hamiltonian, matrix_reconstructed)

    static = solver.get_static_self_energy()
    self_energy = solver.get_self_energy()
    eigvals, eigvecs = self_energy.diagonalise_matrix(static)

    print(solver.eigvals[:5])
    print(eigvals[:5])
    assert np.allclose(solver.eigvals, eigvals)

    if expression.hermitian:
        matrix_reconstructed = (eigvecs * eigvals[None]) @ eigvecs.T.conj()
    else:
        matrix_reconstructed = (eigvecs[1] * eigvals[None]) @ eigvecs[0].T.conj()

    eigvals, eigvecs = util.eig_biorth(matrix_reconstructed, hermitian=expression.hermitian)

    assert np.allclose(solver.eigvals, eigvals)
