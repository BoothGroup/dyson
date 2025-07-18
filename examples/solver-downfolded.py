r"""Example of the downfolded eigenvalue solver.

This solver finds the eigenvalues of the frequency-dependent downfolded self-energy matrix in a
self-consistent manner. It converges on the pole of the Green's function closest to the initial
guess, but does not account for a fully featured Green's function. The eigenvalue problem also
maintains a dependency on the broadening parameter :math:`\eta`.
"""

import numpy
from pyscf import gto, scf

from dyson import FCI, Downfolded, Exact
from dyson.grids import GridRF

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

# Solve the Hamiltonian using the Downfolded solver, initialisation via either:

# 1) Create the solver from a self-energy
solver = Downfolded.from_self_energy(static, self_energy, overlap=overlap, eta=1e-2)
solver.kernel()

# 2) Create the solver directly from the generating function


def _function(freq: float) -> numpy.ndarray:
    """Evaluate the self-energy at the frequency."""
    grid = GridRF(1, buffer=numpy.array([freq]))
    grid.eta = 1e-2
    return grid.evaluate_lehmann(self_energy, ordering="ordered").array[0]


solver = Downfolded(
    static,
    _function,
    overlap=overlap,
    hermitian=exp.hermitian,
)
solver.kernel()
