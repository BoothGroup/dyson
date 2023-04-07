"""
Example performing an ADC(2) calculation, using the `Davidson`
solver and the `MP2` expressions.
"""

import numpy as np
from pyscf import gto, scf, lib
from dyson import Lehmann, NullLogger, Davidson
from dyson.expressions import MP2


# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol).run()

# Get the expressions
mp2 = MP2["1h"](mf, non_dyson=True)
static = mp2.get_static_part()
diag = mp2.diagonal(static=static)
matvec = lambda v: mp2.apply_hamiltonian(v, static=static)

# Run the Davidson algorithm
solver = Davidson(matvec, diag, nroots=5, nphys=mp2.nocc)
solver.conv_tol = 1e-10
solver.kernel()
solver.log.info("IP: %.8f", -solver.get_greens_function().occupied().energies[-1])

# Compare to PySCF
print("\nPySCF:")
from pyscf import adc
adc2 = adc.ADC(mf)
adc2.verbose = 4
adc2.kernel(nroots=5)
