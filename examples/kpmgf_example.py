"""
Example of the Green's function moment kernel polynomial method
(KMPGF) solver, leveraging a Chebyshev moment representation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, agf2, lib
from dyson import KPMGF

ncheb = 50  # Number of Chebyshev moments
kernel_type = "lorentz"  # Kernel method

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
solver = KPMGF(moments, grid, (a, b), kernel_type=kernel_type)
sf = solver.kernel()

# Get a reference spectral function for comparison
sf_ref = -lib.einsum(
        "pk,qk,wk->wpq",
        gf.coupling,
        gf.coupling,
        1.0/(grid[:,None]-gf.energy[None]+1.0j),
).imag / np.pi

# Plot the results
plt.plot(grid, np.trace(sf_ref, axis1=1, axis2=2), "C0-", label="Reference")
plt.plot(grid, np.trace(sf, axis1=1, axis2=2), "C1-", label="KPMGF")
plt.legend()
plt.xlabel("Frequency (Ha)")
plt.ylabel("Spectral function")
plt.show()
