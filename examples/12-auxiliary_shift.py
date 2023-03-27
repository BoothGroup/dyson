"""
Example of applying auxiliary shifts to satisfy the number of
electrons when one has auxiliaries, and the Aufbau principle alone
cannot satisfy the number of electrons.
"""

import numpy as np
from dyson import Lehmann, MBLSE, AufbauPrinciple, AuxiliaryShift

np.random.seed(1)

# Define a Fock matrix
n = 10
fock = np.diag(np.random.random(n))

# Define a self-energy
moms = np.random.random((6, n, n))
moms = moms + moms.transpose(0, 2, 1)
mblse = MBLSE(fock, moms)
mblse.kernel()
se = mblse.get_self_energy()

# Define the number of electrons filling them, at double filling (i.e. RHF)
nelec = 6

# Use the AufbauPrinciple class to get the chemical potential - this
# won't be satisfied exactly
w, v = se.diagonalise_matrix(fock)
gf = Lehmann(w, v[:n])
solver = AufbauPrinciple(gf, nelec, occupancy=2)
solver.kernel()

# Use the AuxiliaryShift class to get the chemical potential more
# accurately, by shifting the self-energy poles with respect to those
# of the Green's function
solver = AuxiliaryShift(fock, se, nelec, occupancy=2)
solver.kernel()
