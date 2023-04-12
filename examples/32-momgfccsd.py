"""
Example performaing a MomGF-CCSD calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, cc
from dyson import MBLGF, NullLogger, util
from dyson.expressions import CCSD
np.set_printoptions(edgeitems=1000, linewidth=1000, precision=3)

nmom = 4

# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol).run()

# Run a CCSD calculation
ccsd = cc.CCSD(mf)
ccsd.kernel()
ccsd.solve_lambda()

# Find the moments
expr = CCSD["1h"](mf, t1=ccsd.t1, t2=ccsd.t2, l1=ccsd.l1, l2=ccsd.l2)
th = expr.build_gf_moments(nmom)
expr = CCSD["1p"](mf, t1=ccsd.t1, t2=ccsd.t2, l1=ccsd.l1, l2=ccsd.l2)
tp = expr.build_gf_moments(nmom)

# Solve for the Green's function
solverh = MBLGF(th, hermitian=False)
solverh.kernel()
gfh = solverh.get_greens_function()
solverp = MBLGF(tp, hermitian=False)
solverp.kernel()
gfp = solverp.get_greens_function()
gf = gfh + gfp

# Get the spectrum
grid = np.linspace(-5, 5, 1024)
eta = 1e-1
sf = util.build_spectral_function(gf.energies, gf.couplings, grid, eta=eta)

# If PySCF version is new enough, plot a reference
try:
    momgfcc = cc.momgfccsd.MomGFCCSD(ccsd, ((nmom-2)//2, (nmom-2)//2))
    eh, vh, ep, vp = momgfcc.kernel()
    e = np.concatenate((eh, ep), axis=0)
    v = np.concatenate((vh[0], vp[0]), axis=1)
    u = np.concatenate((vh[1], vp[1]), axis=1)
    sf_ref = util.build_spectral_function(e, (v, u), grid, eta=eta)
    plt.plot(grid, sf_ref, "C1--", label="MomGF-CCSD (PySCF)", zorder=10)
except AttributeError:
    pass

# Plot the results
plt.plot(grid, sf, "C0-", label="MomGF-CCSD")
plt.legend()
plt.xlabel("Frequency (Ha)")
plt.ylabel("Spectral function")
plt.show()
