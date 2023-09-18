"""
Example showing the relationship between the FCI static self-energy
and the Green's function moments.
"""

import numpy as np
from pyscf import gto, scf, lib
from dyson import Lehmann, NullLogger, MBLGF, MixedMBLGF
from dyson.expressions import FCI


# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol).run()

# Get the expressions
fci_1h = FCI["1h"](mf)
fci_1p = FCI["1p"](mf)

# Use MBLGF
th = fci_1h.build_gf_moments(4)
tp = fci_1p.build_gf_moments(4)
solver_h = MBLGF(th)
solver_h.kernel()
solver_p = MBLGF(tp)
solver_p.kernel()
solver = MixedMBLGF(solver_h, solver_p)

# Get the Green's function
gf = solver.get_greens_function()

# Back-transform to the self-energy and use the sum of the first order
# moments as the static self-energy
se = solver.get_self_energy()
se_static = th[1] + tp[1]
gf_recov = Lehmann(*se.diagonalise_matrix_with_projection(se_static))
assert np.allclose(gf.moment(range(4)), gf_recov.moment(range(4)))

# This is equivalent to the Fock matrix evaluated at the FCI density
dm = th[0] * 2.0
dm = np.linalg.multi_dot((mf.mo_coeff, dm, mf.mo_coeff.T))
f = mf.get_fock(dm=dm)
f = np.linalg.multi_dot((mf.mo_coeff.T, f, mf.mo_coeff))
assert np.allclose(f, se_static)
