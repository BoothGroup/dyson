"""Example of the Aufbau principle solver.

This solver applies another solver before filling the resulting solution according to the Aufbau
principle. This attaches a chemical potential to the solution, and the resulting self-energy and
Green's function.
"""

import numpy
from pyscf import gto, scf

from dyson import TDAGW, AufbauPrinciple, MBLGF, Exact
from dyson.solvers.static.chempot import search_aufbau_global

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

# Solve the Hamiltonian using the Aufbau solver, initialisation via either:

# 1) Create the solver from a self-energy
solver = AufbauPrinciple.from_self_energy(static, self_energy, overlap=overlap, nelec=mol.nelectron)
solver.kernel()

# 2) Create the solver directly from the self-energy
solver = AufbauPrinciple(
    static,
    self_energy,
    overlap=overlap,
    nelec=mol.nelectron,
)
solver.kernel()

# By default, this is solving the input self-energy using the Exact solver. To use another solver,
# e.g. MBLSE, as the base solver, you can specify it as the `solver` argument
solver = AufbauPrinciple.from_self_energy(
    static, self_energy, overlap=overlap, nelec=mol.nelectron, solver=MBLGF
)
solver.kernel()

# If you don't want to solve the self-energy at all and just want to find a chemical potential for
# an existing solution, you can pass the Green's function directly to the search functions
greens_function = solver.result.get_greens_function()
chempot, error = search_aufbau_global(greens_function, mol.nelectron)
