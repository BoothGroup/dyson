"""
Example of optimising the density matrix such that it is self-consistent
with the Fock matrix, in the presence of some self-energy, with
intermediate chemical potential optimisation.
"""

import numpy as np
from pyscf import gto, scf, agf2
from dyson import Lehmann, MBLSE, SelfConsistentField

# Define a self-energy using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="aug-cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
fock = np.diag(mf.mo_energy)
gf2 = agf2.AGF2(mf, nmom=(None, 0))
se = gf2.build_se()

# Define a function to obtain the Fock matrix in the MO basis
def get_fock(dm):
    dm_ao = np.linalg.multi_dot((mf.mo_coeff, dm, mf.mo_coeff.T))
    fock = mf.get_fock(dm=dm_ao)
    return np.linalg.multi_dot((mf.mo_coeff.T, fock, mf.mo_coeff))

# Use the SelfConsistentField class to relax the density matrix such
# that it is self-consistent with the Fock matrix, and the number of
# electrons is correct
solver = SelfConsistentField(get_fock, se, mol.nelectron)
solver.conv_tol = 1e-10
solver.max_cycle_inner = 30
solver.kernel()
