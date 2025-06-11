"""Example of particle-hole separated calculations."""

import matplotlib.pyplot as plt
import numpy
from pyscf import gto, scf

from dyson import ADC2, MBLGF, Exact, Spectral
from dyson.grids import GridRF

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use FCI one-hole and one-particle expressions for the Hamiltonian
exp_h = ADC2.h.from_mf(mf)
exp_p = ADC2.p.from_mf(mf)

# Use MBLGF to solve the Hamiltonian for each case separately
solver_h = MBLGF.from_expression(exp_h, max_cycle=1)
solver_h.kernel()
solver_p = MBLGF.from_expression(exp_p, max_cycle=1)
solver_p.kernel()

# Combine the results -- this function operators by projecting the result back into a self-energy
# and combining the two self-energies, before diagonalising the combined self-energy to get a new
# result spectrum. This may have unwanted consequences for some methodology, so use with care.
result = Spectral.combine(solver_h.result, solver_p.result)

# Get the spectral functions
grid = GridRF.from_uniform(-3.0, 3.0, 1024, eta=0.05)
spectrum_h = -grid.evaluate_lehmann(
    solver_h.result.get_greens_function(), ordering="advanced", trace=True
).imag / numpy.pi
spectrum_p = -grid.evaluate_lehmann(
    solver_p.result.get_greens_function(), ordering="advanced", trace=True
).imag / numpy.pi
spectrum_combined = -grid.evaluate_lehmann(
    result.get_greens_function(), ordering="advanced", trace=True
).imag / numpy.pi

# Plot the spectra
plt.figure()
plt.plot(grid, spectrum_combined, "k-", label="Combined Spectrum")
plt.plot(grid, spectrum_h, "r--", label="Hole Spectrum")
plt.plot(grid, spectrum_p, "b--", label="Particle Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Spectral function")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
