"""
Example performing an AGF2 calculation, using the `SelfConsistent`
and `DensityRelaxation` solvers, along with the `MP2` expressions,
leverage the plug-and-play callback.
"""

import numpy as np
from pyscf import gto, scf, lib
from dyson import Lehmann, NullLogger, MBLSE, MixedMBLSE, DensityRelaxation, SelfConsistent
from dyson.expressions import MP2

nmom = 2

# Define a system using PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto3g", verbose=0)
mf = scf.RHF(mol).run()

# Define a function to calculate the Fock matrix in the MO basis
def get_fock(rdm1_mo):
    rdm1_ao = np.linalg.multi_dot((mf.mo_coeff, rdm1_mo, mf.mo_coeff.T))
    fock_ao = mf.get_fock(dm=rdm1_ao)
    fock_mo = np.linalg.multi_dot((mf.mo_coeff.T, fock_ao, mf.mo_coeff))
    return fock_mo

# Define a function to calculate the self-energy - also uses DIIS
# to extrapolate those moments. Note that the `diis` object would
# need to be cleared between calculations in the same script.
diis = lib.diis.DIIS()
def get_se(gf, se_prev=None):
    mo_energy, mo_coeff, mo_occ = gf.as_orbitals(mo_coeff=mf.mo_coeff, occupancy=2)
    fock = get_fock(gf.occupied().moment(0) * 2)

    mp2h = MP2["1h"](mf, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    th = mp2h.build_se_moments(nmom)
    th = lib.einsum("...ij,pi,qj->...pq", th, gf.couplings, gf.couplings)

    mp2p = MP2["1p"](mf, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    tp = mp2p.build_se_moments(nmom)
    tp = lib.einsum("...ij,pi,qj->...pq", tp, gf.couplings, gf.couplings)

    th, tp = diis.update(np.array([th, tp]), xerr=None)

    solverh = MBLSE(fock, th, log=NullLogger())
    solverh.kernel()

    solverp = MBLSE(fock, tp, log=NullLogger())
    solverp.kernel()

    solver = MixedMBLSE(solverh, solverp)

    return solver.get_self_energy()

# Define the initial Green's function
gf = Lehmann(mf.mo_energy, np.eye(mf.mo_energy.size))

# Run the solver
solver = SelfConsistent(
        get_se,
        gf,
        relax_solver=DensityRelaxation,
        get_fock=get_fock,
        conv_tol=1e-10,
)
solver.kernel()
solver.log.info("IP: %.8f", -solver.get_greens_function().occupied().energies[-1])
solver.log.info("EA: %.8f", solver.get_greens_function().virtual().energies[0])

# Compare to PySCF
print("\nPySCF:")
from pyscf import agf2
gf2 = agf2.AGF2(mf)
gf2.verbose = 3
gf2.kernel()
