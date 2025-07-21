"""Plot spectra in the style of the cover of Joy Division's 'Unknown Pleasures' album."""

import matplotlib.pyplot as plt
import numpy
from pyscf import gto, scf

from dyson import ADC2, MBLGF, Lehmann, quiet
from dyson.grids import GridRF
from dyson.plotting import unknown_pleasures

# Suppress output
quiet()

# Define a grid for the spectra
grid = GridRF.from_uniform(-5.0, 7.0, 128, eta=0.25)

# Generate random bond distances for a pair of nitrogen atoms
spectra = []
for _ in range(64):
    bond_distance = numpy.random.uniform(0.8, 2.5)
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {bond_distance}", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    adc2_h = ADC2.h.from_mf(mf)
    adc2_p = ADC2.p.from_mf(mf)

    # Solve the ADC(2) Green's function for the hole and particle sectors
    gf_h = MBLGF.from_expression(adc2_h, max_cycle=4).kernel().get_greens_function()
    gf_p = MBLGF.from_expression(adc2_p, max_cycle=4).kernel().get_greens_function()
    gf = Lehmann.concatenate(gf_h, gf_p)
    sf = grid.evaluate_lehmann(gf, ordering="advanced", reduction="trace", component="imag")
    spectra.append(sf)

# Plot the spectra in the style of 'Unknown Pleasures'
unknown_pleasures(spectra)
plt.show()
