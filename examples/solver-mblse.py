"""Example of the MBLSE solver.

This solver diagonalises the self-energy via conservation of its spectral moments, using recursion
relations of those moments. The resulting Green's function is approximate, and is systematically
improved by increasing the number of moments (maximum algorithm cycle) used in the calculation.
"""

from pyscf import gto, scf

from dyson import ADC2, MBLSE, MLSE, Exact

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use an ADC(2) one-hole expression for the Hamiltonian
exp = ADC2.hole.from_mf(mf)

# Use the exact solver to get the self-energy for demonstration purposes
exact = Exact.from_expression(exp)
exact.kernel()
static = exact.result.get_static_self_energy()
self_energy = exact.result.get_self_energy()
overlap = exact.result.get_overlap()

# Solve the Hamiltonian using the MBLSE solver, initialisation via either:

# 1) Create the solver from the expression
solver = MBLSE.from_expression(exp, max_cycle=1)
solver.kernel()

# 2) Create the solver from a self-energy
solver = MBLSE.from_self_energy(static, self_energy, overlap=overlap, max_cycle=1)
solver.kernel()

# 3) Create the solver directly from the moments
max_cycle = 1
solver = MBLSE(
    static,
    self_energy.moments(range(2 * max_cycle + 2)),
    overlap=overlap,
    hermitian=exp.hermitian_downfolded,
    max_cycle=max_cycle,
)
solver.kernel()

# One can also use the MLSE solver, which a specialised implementation of MBLSE for single elements
# of the static self-energy and self-energy moments. In a diagonal approximation, this can be used
# to reduce the scaling of MBLSE by treating each diagonal element separately.
solver = MLSE(
    static[0, 0],
    self_energy.moments(range(2 * max_cycle + 2))[:, 0, 0],
    overlap=overlap[0, 0],
    max_cycle=max_cycle,
)
solver.kernel()
