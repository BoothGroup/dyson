"""Example of the exact diagonalisation solver.

This solver is a non-scalable solver that exactly diagonalises the dense Hamiltonian as a
demonstration for small systems. When constructing from an expression, it constructs the Hamiltonian
matrix using repeated applications of the matrix-vector product to unit vectors, which is also slow.
"""

import numpy
from pyscf import gto, scf

from dyson import FCI, Exact

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use an FCI one-hole expression for the Hamiltonian
exp = FCI.hole.from_mf(mf)

# Solve the Hamiltonian using the Exact solver, initialisation via either:

# 1) Create the solver from the expression
solver = Exact.from_expression(exp)
solver.kernel()

# 2) Create the solver from a self-energy
static = solver.result.get_static_self_energy()
self_energy = solver.result.get_self_energy()
overlap = solver.result.get_overlap()
solver = Exact.from_self_energy(static, self_energy, overlap=overlap)
solver.kernel()

# 3) Create the solver directly from the matrix and excitation vectors
solver = Exact(
    exp.build_matrix(),
    numpy.asarray(exp.get_excitation_bras()),
    numpy.asarray(exp.get_excitation_kets()),
    hermitian=exp.hermitian_upfolded,
)
solver.kernel()
