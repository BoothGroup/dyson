"""Example of the Chebyshev polynomial Green's function solver.

This solver uses Chebyshev polynomials to evaluate the Green's function on a real frequency grid. It
is systematically improvable by increasing the order of the Chebyshev polynomial expansion. It is
related to MBLGF, however, it does not offer a static result, rather a dynamic Green's function.
"""

import numpy
from pyscf import gto, scf

from dyson import FCI, CPGF, Exact, util
from dyson.grids import RealFrequencyGrid

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use an FCI one-hole expression for the Hamiltonian
exp = FCI.hole.from_mf(mf)

# Initialise a real frequency grid for the correction vector solver
grid = RealFrequencyGrid.from_uniform(-3.0, 0.0, 128, eta=1e-2)

# Use the exact solver to get the self-energy for demonstration purposes
exact = Exact.from_expression(exp)
exact.kernel()
static = exact.result.get_static_self_energy()
self_energy = exact.result.get_self_energy()
overlap = exact.result.get_overlap()

# CPGF requires a pair of scaling parameters, which are used to scale the spectrum onto the range
# [-1, 1]. The scaling parameters can be obtained from the minimum and maximum eigenvalues of the
# Hamiltonian if they are known a priori, or they can be approximated from the diagonal of the
# expression. If the approximation is used, it is recommended to use an additional factor to avoid
# cases where the minimum and maximum of the diagonal are not representative of the spectrum.
energies, _ = self_energy.diagonalise_matrix(static, overlap=overlap)
scaling = util.get_chebyshev_scaling_parameters(energies.min(), energies.max())

# Solve the Hamiltonian using the CPGF solver, initialisation via either:

# 1) Create the solver from the expression
max_cycle = 1024
solver = CPGF.from_expression(exp, grid=grid, max_cycle=max_cycle, scaling=scaling, ordering="advanced")
gf = solver.kernel()

# 2) Create the solver from a self-energy
solver = CPGF.from_self_energy(
    static,
    self_energy,
    overlap=overlap,
    grid=grid,
    max_cycle=max_cycle,
    scaling=scaling,
    ordering="advanced",
)
gf = solver.kernel()

# 3) Create the solver directly from the matrix and excitation vectors
solver = CPGF(
    exp.build_gf_chebyshev_moments(max_cycle + 1, scaling=scaling),
    grid,
    max_cycle=max_cycle,
    scaling=scaling,
    ordering="advanced",
)
gf = solver.kernel()

# Compare to that of the Exact solver, by downfolding the Green's function corresponding to the
# exact result onto the same grid
gf_exact = grid.evaluate_lehmann(exact.result.get_greens_function(), ordering="advanced")
print("Correction vector error:", numpy.max(numpy.abs(gf - gf_exact)))
