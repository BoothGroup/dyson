"""
Example performing an EOM-CCSD calculation, using the `Davidson`
solver and the `CCSD` expressions.
"""

import numpy as np
from pyscf import gto, scf, lib
from dyson import Lehmann, NullLogger, Davidson
from dyson.expressions import CCSD


# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol).run()

# Get the expressions
ccsd = CCSD["1h"](mf)
diag = ccsd.diagonal()
matvec = ccsd.apply_hamiltonian

# Run the Davidson algorithm
solver = Davidson(matvec, diag, nroots=5, nphys=ccsd.nocc)
solver.conv_tol = 1e-10
solver.kernel()
solver.log.info("IP: %.8f", -solver.get_greens_function().occupied().energies[-1])

# Compare to PySCF
print("\nPySCF:")
from pyscf import cc
ccsd = cc.CCSD(mf)
ccsd.verbose = 4
ccsd.kernel()
ccsd.ipccsd(nroots=5)

