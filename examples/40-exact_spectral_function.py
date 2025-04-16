"""
Example showing the construction of exact spectral functions (i.e. no moment
approximation) for CCSD.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf
from dyson import MBLGF, MixedMBLGF, util
from dyson.expressions import CCSD

niter_max = 4
grid = np.linspace(-4, 4, 256)

# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()

# Get the expressions
ccsd_1h = CCSD["1h"](mf)
ccsd_1p = CCSD["1p"](mf)

# Use MBLGF
th = ccsd_1h.build_gf_moments(niter_max * 2 + 2)
tp = ccsd_1p.build_gf_moments(niter_max * 2 + 2)
solver_h = MBLGF(th)
solver_h.kernel()
solver_p = MBLGF(tp)
solver_p.kernel()
solver = MixedMBLGF(solver_h, solver_p)

# Use the solver to get the approximate spectral functions
sf_approx = []
for i in range(1, niter_max + 1):
    e, v = solver.get_dyson_orbitals(i)
    sf = util.build_spectral_function(e, v, grid, eta=0.1)
    sf_approx.append(sf)

# Get the exact spectral function. Note that the exact function is solved
# using a correction vector approach that may not be robust at all frequencies,
# and the user should check the output array for NaNs.
#
# The procedure to obtain the exact spectral function scales as O(N_freq) and
# therefore the size of the grid should be considered.
sf_exact_h = util.build_exact_spectral_function(ccsd_1h, grid, eta=0.1)
sf_exact_p = util.build_exact_spectral_function(ccsd_1p, grid, eta=0.1)
sf_exact = sf_exact_h + sf_exact_p

# Plot the results
plt.plot(grid, sf_exact, "k-", label="Exact")
for i in range(1, niter_max + 1):
    plt.plot(grid, sf_approx[i - 1], f"C{i-1}--", label=f"MBLGF (niter={i})")
plt.legend()
plt.xlabel("Frequency (Ha)")
plt.ylabel("Spectral function")
plt.show()
