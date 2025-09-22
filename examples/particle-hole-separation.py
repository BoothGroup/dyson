"""Example of particle-hole separated calculations."""

import matplotlib.pyplot as plt
import numpy
from pyscf import gto, scf

from dyson import ADC2, MBLGF, Spectral
from dyson.grids import GridRF
from dyson.plotting import format_axes_spectral_function, plot_dynamic

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
spectrum_h = (1 / numpy.pi) * grid.evaluate_lehmann(
    solver_h.result.get_greens_function(),
    ordering="advanced",
    reduction="trace",
    component="imag",
)
spectrum_p = (1 / numpy.pi) * grid.evaluate_lehmann(
    solver_p.result.get_greens_function(),
    ordering="advanced",
    reduction="trace",
    component="imag",
)
spectrum_combined = (1 / numpy.pi) * grid.evaluate_lehmann(
    result.get_greens_function(),
    ordering="advanced",
    reduction="trace",
    component="imag",
)

# Plot the spectra
fig, ax = plt.subplots()
plot_dynamic(spectrum_combined, fmt="k-", label="Combined Spectrum", energy_unit="eV", ax=ax)
plot_dynamic(spectrum_h, fmt="C0--", label="Hole Spectrum", energy_unit="eV", ax=ax)
plot_dynamic(spectrum_p, fmt="C1--", label="Particle Spectrum", energy_unit="eV", ax=ax)
format_axes_spectral_function(grid, ax=ax, energy_unit="eV")
plt.legend()
plt.show()
