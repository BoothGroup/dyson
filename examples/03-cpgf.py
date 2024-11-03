"""
Example of the Chebyshev polynomial Green's function method
(CPGF) solver, which is similar to KPMGF but more accurately
produces the correctly normalised Lorentzian spectral function.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, agf2, lib
from dyson import CPGF, util

ncheb = 100  # Number of Chebyshev moments

# Define a self-energy using PySCF
mol = gto.M(atom="O 0 0 0; O 0 0 1", basis="6-31g", verbose=0)
mf = scf.RHF(mol).run()
se_static = np.diag(mf.mo_energy)
se = agf2.AGF2(mf, nmom=(None, None)).build_se()

# Found the bounds of the self-energy - in practice this should be
# done using a lower-scaling solver.
gf = se.get_greens_function(se_static)
emin = gf.energy.min()
emax = gf.energy.max()
grid = np.linspace(emin, emax, 1024)

# Scale the energies of the Green's function
a = (emax - emin) / (2.0 - 1e-2)
b = (emax + emin) / 2.0
energy_scaled = (gf.energy - b) / a

# Compute the Chebyshev moments
c = np.zeros((ncheb, mol.nao, energy_scaled.size))
c[0] = gf.coupling
c[1] = gf.coupling * energy_scaled
for i in range(2, ncheb):
    c[i] = 2.0 * c[i-1] * energy_scaled - c[i-2]
moments = lib.einsum("qx,npx->npq", gf.coupling, c)

# Use the solver to get the spectral function
solver = CPGF(moments, grid, (a, b), eta=1.0)
sf = solver.kernel()

# Get a reference spectral function for comparison
sf_ref = util.build_spectral_function(gf.energy, gf.coupling, grid, eta=1.0)

# Plot the results
plt.plot(grid, sf_ref, "C0-", label="Reference")
ylim = plt.ylim()
plt.plot(grid, sf, "C1-", label="CPGF")
plt.legend()
plt.xlabel("Frequency (Ha)")
plt.ylabel("Spectral function")
plt.ylim(ylim)
plt.show()
