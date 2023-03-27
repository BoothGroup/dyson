"""
Example of applying Aufbau principle.
"""

import numpy as np
from dyson import Lehmann, AufbauPrinciple

# Define some energies
e = np.arange(10).astype(float)

# Put them into a Lehmann representation for a Green's function
c = np.eye(e.size)
gf = Lehmann(e, c)

# Define the number of electrons filling them, at double filling (i.e. RHF)
nelec = 6

# Use the AufbauPrinciple class to get the HOMO and LUMO, and therefore the
# chemical potential
solver = AufbauPrinciple(gf, nelec, occupancy=2)
solver.kernel()
