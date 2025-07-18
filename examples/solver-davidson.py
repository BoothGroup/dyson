"""Example of the Davidson eigenvalue solver.

This solver is a traditional iterative Jacobi--Davidson eigenvalue solver to find the eigenvalues
and eigenvectors of a Hamiltonian corresponding to the lowest-lying states. As such, it does not
fully calculate the Green's function, but targets the energy region close to the Fermi level.
"""

import numpy
from pyscf import gto, scf

from dyson import FCI, Davidson, Exact

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use an FCI one-hole expression for the Hamiltonian
exp = FCI.hole.from_mf(mf)

# Use the exact solver to get the self-energy for demonstration purposes
exact = Exact.from_expression(exp)
exact.kernel()
static = exact.result.get_static_self_energy()
self_energy = exact.result.get_self_energy()
overlap = exact.result.get_overlap()

# Solve the Hamiltonian using the Davidson solver, initialisation via either:

# 1) Create the solver from the expression
solver = Davidson.from_expression(exp, nroots=5)
solver.kernel()

# 2) Create the solver from a self-energy
solver = Davidson.from_self_energy(static, self_energy, overlap=overlap, nroots=5)
solver.kernel()

# 3) Create the solver directly from the matrix and excitation vectors
solver = Davidson(
    exp.apply_hamiltonian,
    exp.diagonal(),
    numpy.asarray(exp.get_excitation_bras()),
    numpy.asarray(exp.get_excitation_kets()),
    hermitian=exp.hermitian_upfolded,
    nroots=5,
)
solver.kernel()
