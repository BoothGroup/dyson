"""
Example performing an FCI calculation for the IP, using the `Davidson`
solver and the `FCI` expressions.
"""

import numpy as np
from pyscf import gto, scf, lib
from dyson import Lehmann, NullLogger, Davidson, MBLGF
from dyson.expressions import FCI


# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol).run()

# Get the expressions
fci = FCI["1h"](mf)
diag = fci.diagonal()
matvec = fci.apply_hamiltonian

# Run the Davidson algorithm
solver = Davidson(matvec, diag, nroots=5, nphys=fci.nocc)
solver.conv_tol = 1e-10
solver.kernel()
solver.log.info("IP: %.8f", -solver.get_greens_function().occupied().energies[-1])

# Use MBLGF
moments = fci.build_gf_moments(4)
solver = MBLGF(moments)
solver.kernel()
solver.log.info("IP: %.8f", -solver.get_greens_function().occupied().energies[-1])
