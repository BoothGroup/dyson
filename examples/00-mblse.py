"""
Example of the moment block Lanczos recursion for moments of the
self-energy (MBLSE) solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, agf2, lib
from dyson import MBLSE, util

niter = 1
grid = np.linspace(-40, 20, 1024)

# Define a self-energy using PySCF
mol = gto.M(atom="O 0 0 0; O 0 0 1", basis="6-31g", verbose=0)
mf = scf.RHF(mol).run()
se_static = np.diag(mf.mo_energy)
se = agf2.AGF2(mf, nmom=(None, None)).build_se()
se_moms = se.moment(range(2*niter+2))

# Use the solver to get the spectral function
solver = MBLSE(se_static, se_moms)
e, v = solver.kernel()
sf = util.build_spectral_function(e, v[:mol.nao], grid, eta=1.0)

# Get a reference spectral function for comparison
gf = se.get_greens_function(se_static)
sf_ref = util.build_spectral_function(gf.energy, gf.coupling, grid, eta=1.0)

# Plot the results
plt.plot(grid, sf_ref, "C0-", label="Reference")
plt.plot(grid, sf, "C1-", label="MBLSE")
plt.legend()
plt.xlabel("Frequency (Ha)")
plt.ylabel("Spectral function")
plt.show()
