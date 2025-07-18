"""Example of the correction vector solver.

This solver is a dynamic iterative solver to build the downfolded frequency-dependent Green's
function. It uses a GMRES algorithm under the hood to iteratively improve the correction vector
and contract to the Green's function.
"""

import numpy
from pyscf import gto, scf

from dyson import FCI, CorrectionVector, Exact
from dyson.grids import RealFrequencyGrid

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use an FCI one-hole expression for the Hamiltonian
exp = FCI.hole.from_mf(mf)

# Initialise a real frequency grid for the correction vector solver
grid = RealFrequencyGrid.from_uniform(-3.0, 3.0, 128, eta=1e-2)

# Use the exact solver to get the self-energy for demonstration purposes
exact = Exact.from_expression(exp)
exact.kernel()
static = exact.result.get_static_self_energy()
self_energy = exact.result.get_self_energy()
overlap = exact.result.get_overlap()

# Solve the Hamiltonian using the correction vector solver, initialisation via either:

# 1) Create the solver from the expression
solver = CorrectionVector.from_expression(exp, grid=grid, ordering="ordered")
gf = solver.kernel()

# 2) Create the solver from a self-energy
solver = CorrectionVector.from_self_energy(
    static, self_energy, overlap=overlap, grid=grid, ordering="ordered"
)
gf = solver.kernel()

# 3) Create the solver directly from the matrix and excitation vectors
solver = CorrectionVector(
    exp.apply_hamiltonian,
    exp.diagonal(),
    exp.nphys,
    grid,
    exp.get_excitation_bra,
    exp.get_excitation_ket,
    ordering="ordered",
)
gf = solver.kernel()

# Compare to that of the Exact solver, by downfolding the Green's function corresponding to the
# exact result onto the same grid
gf_exact = grid.evaluate_lehmann(exact.result.get_greens_function(), ordering="ordered")
print("Correction vector error:", numpy.max(numpy.abs(gf - gf_exact)))
