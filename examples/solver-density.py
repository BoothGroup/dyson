"""Example of the density relaxation solver.

This solver relaxes the density matrix of a system in the presence of a self-energy. Between
iterations, the self-energy is shifted in order to allow the Aufbau principle to assign a
chemical potential that best matches the particle number of the system. The resulting Green's
function is a minimum with respect to the self-consistent field.

The solvers require a function to evaluate the static part of the self-energy (i.e. the Fock matrix)
for a given density matrix. We provide a convenience function to get this from a PySCF RHF object.

Note that for some Hamiltonians, the relaxation of the density and the shifting of the self-energy
may not commute, i.e. their solutions cannot be obtain simultaneously. In this case, one solution
is favoured according to a parameter.
"""

from pyscf import gto, scf

from dyson import MBLSE, TDAGW, AufbauPrinciple, AuxiliaryShift, DensityRelaxation, Exact
from dyson.solvers.static.density import get_fock_matrix_function

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use a TDA-GW Dyson expression for the Hamiltonian
exp = TDAGW.dyson.from_mf(mf)

# Use the exact solver to get the self-energy for demonstration purposes
exact = Exact.from_expression(exp)
exact.kernel()
static = exact.result.get_static_self_energy()
self_energy = exact.result.get_self_energy()
overlap = exact.result.get_overlap()

# Solve the Hamiltonian using the density relaxation solver, initialisation via either:

# 1) Create the solver from a self-energy
solver = DensityRelaxation.from_self_energy(
    static,
    self_energy,
    overlap=overlap,
    get_static=get_fock_matrix_function(mf),
    nelec=mol.nelectron,
)
solver.kernel()

# 2) Create the solver directly from the self-energy
solver = DensityRelaxation(
    static,
    self_energy,
    overlap=overlap,
    get_static=get_fock_matrix_function(mf),
    nelec=mol.nelectron,
)
solver.kernel()

# Like the auxiliary shift solver, we can customise the solvers


class MyAufbauPrinciple(AufbauPrinciple):  # noqa: D101
    solver = MBLSE


class MyAuxiliaryShift(AuxiliaryShift):  # noqa: D101
    solver = MyAufbauPrinciple


solver = DensityRelaxation.from_self_energy(
    static,
    self_energy,
    overlap=overlap,
    get_static=get_fock_matrix_function(mf),
    nelec=mol.nelectron,
    solver_outer=MyAuxiliaryShift,
    solver_inner=MyAufbauPrinciple,
)
solver.kernel()

# By default, the non-commutative solutions favour the self-consistency in the density matrix,
# rather than the particle number. To favour the particle number, we can pass an additional
# parameter
solver = DensityRelaxation.from_self_energy(
    static,
    self_energy,
    overlap=overlap,
    get_static=get_fock_matrix_function(mf),
    nelec=mol.nelectron,
    solver_outer=MyAuxiliaryShift,
    solver_inner=MyAufbauPrinciple,
    favour_rdm=False,  # Favour the particle number over the density matrix
)
solver.kernel()
