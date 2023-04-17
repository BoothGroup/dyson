"""
Example performing a momentGW calculation.
"""

import numpy as np
from pyscf import gto, dft, cc
from dyson import MBLSE, MixedMBLSE, NullLogger, util
from dyson.expressions import GW

nmom_max = 5

# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "hf"
mf.kernel()

# Find the moments
gw = GW["Dyson"](mf)
static = gw.get_static_part()
th, tp = gw.build_se_moments(nmom_max+1)

# Solve for the Green's function
solverh = MBLSE(static, th)
solverp = MBLSE(static, tp)
solver = MixedMBLSE(solverh, solverp)
solver.kernel()
gf = solver.get_greens_function()
gf = gf.physical()
solver.log.info("IP: %.8f", -gf.occupied().energies[-1])
solver.log.info("EA: %.8f", gf.virtual().energies[0])

# Compare to momentGW
import momentGW
gw_ref = momentGW.GW(mf)
conv, gf_ref, se_ref = gw_ref.kernel(nmom_max)
gf_ref.remove_uncoupled(tol=0.1)
solver.log.info("")
solver.log.info("IP (ref): %.8f", -gf_ref.get_occupied().energy[-1])
solver.log.info("EA (ref): %.8f", gf_ref.get_virtual().energy[0])
