"""
Example of mixing MBL solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, agf2, lib
from dyson import MBLSE, MixedMBL, util

niter_occ = 1
niter_vir = 2
grid = np.linspace(-40, 20, 1024)

# Define a self-energy using PySCF
mol = gto.M(atom="O 0 0 0; O 0 0 1", basis="6-31g", verbose=0)
mf = scf.RHF(mol).run()
se_static = np.diag(mf.mo_energy)
se = agf2.AGF2(mf, nmom=(None, None)).build_se()
se_occ = se.get_occupied()
se_vir = se.get_virtual()

# Apply a solver to the occupied sector
solver_occ = MBLSE(se_static, se_occ.moment(range(2*niter_occ+2)))
solver_occ.kernel()

# Apply a solver to the virtual sector
solver_vir = MBLSE(se_static, se_vir.moment(range(2*niter_vir+2)))
solver_vir.kernel()

# Mix the solvers
mix = MixedMBL(solver_occ, solver_vir)

# Use the mixed solver to get the spectral function
e, v = mix.get_dyson_orbitals()
sf = util.build_spectral_function(e, v, grid, eta=1.0)

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
