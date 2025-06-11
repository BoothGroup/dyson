"""Example of the auxiliary shift solver.

This solver applies another solver with a variable shift in the energies of the self-energy. This
shift is optimised to allow an Aufbau principle to arrive at the best possible solution with respect
to the particle number. This modifies the self-energy and attaches a chemical potential to the
solution, and the resulting self-energy and Green's function.
"""

from pyscf import gto, scf

from dyson import MBLSE, TDAGW, AufbauPrinciple, AuxiliaryShift, Exact

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

# Solve the Hamiltonian using the auxiliary shift solver, initialisation via either:

# 1) Create the solver from a self-energy
solver = AuxiliaryShift.from_self_energy(static, self_energy, overlap=overlap, nelec=mol.nelectron)
solver.kernel()

# 2) Create the solver directly from the self-energy
solver = AuxiliaryShift(
    static,
    self_energy,
    overlap=overlap,
    nelec=mol.nelectron,
)
solver.kernel()

# By default, this is solving the input self-energy using the default Aufbau solver. To use another
# solver, e.g. one that uses MBLSE for the base solver, you can specify it as the `solver` argument


class MyAufbauPrincple(AufbauPrinciple):  # noqa: D101
    solver = MBLSE


solver = AuxiliaryShift.from_self_energy(
    static, self_energy, overlap=overlap, nelec=mol.nelectron, solver=MyAufbauPrincple
)
solver.kernel()
