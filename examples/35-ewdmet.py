"""
Example performing an EwDMET calculation using `dyson` and `vayesta`.
"""

import sys
sys.argv.append("--quiet")  # For Vayesta
import numpy as np
import scipy.linalg
from pyscf import gto, scf, lib
from vayesta.ewf import EWF
from vayesta.core.util import fix_orbital_sign
from vayesta.lattmod import Hubbard1D, LatticeRHF
from vayesta.core.linalg import recursive_block_svd
from dyson import MBLGF, MixedMBLGF, DensityRelaxation, FCI, Lehmann, util, NullLogger
from dyson import DensityRelaxation, AufbauPrinciple, SelfConsistent
np.set_printoptions(edgeitems=1000, linewidth=1000, precision=3)

nsite = 20
u = 4.0
nmom_max_fci = 1
nmom_max_bath = 1
nfrag = 2

# Define a Hubbard model
hubbard = Hubbard1D(
        nsite=nsite,
        nelectron=nsite,
        hubbard_u=u,
        verbose=0,
)
mf = LatticeRHF(hubbard)
mf.kernel()

# Define the intial orbitals
mo_energy = mf.mo_energy
mo_coeff = mf.mo_coeff  # (site|MO)
mo_occ = mf.mo_occ
chempot = 0.5 * (mf.mo_energy[mo_occ > 0].max() + mf.mo_energy[mo_occ == 0].min())
gf = Lehmann(mo_energy, mo_coeff, chempot=chempot)

# Construct fragment and environment orbitals
c_frag, c_env = np.split(np.eye(nsite), [nfrag], axis=1)  # (site|frag), (site|env)

print(f"\nSystem")
print("-" * len(f"System"))
print(f"Number of sites: {nsite}")
print(f"U: {u}")
print(f"Maximum order FCI moments: {nmom_max_fci}")
print(f"Maximum order bath moments: {nmom_max_bath}")
print(f"Fragment size: {nfrag}")
print(f"Environment size: {nsite - nfrag}")
print(f"Bath size: {nfrag * (2*nmom_max_bath+1)}")
print(f"Cluster size: {nfrag * (2*nmom_max_bath+2)}")

print(f"\nHartree-Fock")
print("-" * len(f"Hartree-Fock"))
print("Nelec: {}".format(hubbard.nelectron))
print("E(tot) = {:.8f}".format(mf.e_tot))
print("IP = {:.8f}".format(-mf.mo_energy[nsite//2-1]))
print("EA = {:.8f}".format(mf.mo_energy[nsite//2]))
print("Gap = {:.8f}".format(mf.mo_energy[nsite//2] - mf.mo_energy[nsite//2-1]))

# Function to canonicalise cluster orbitals
def canonicalize_mo(mo_coeff, fock):
    """Canonicalise the orbitals.
    """
    fock = np.linalg.multi_dot((mo_coeff.T, fock, mo_coeff))
    mo_energy, rot = np.linalg.eigh(fock)
    mo_coeff_canon = np.dot(mo_coeff, rot)
    mo_coeff_canon, signs = fix_orbital_sign(mo_coeff_canon)
    rot = rot * signs[None]
    assert np.allclose(np.dot(mo_coeff, rot), mo_coeff_canon)
    assert np.allclose(np.dot(mo_coeff_canon, rot.T), mo_coeff)
    return mo_energy, mo_coeff_canon

# Function to build the bath
def construct_bath(mo_energy, mo_coeff, mo_occ, tol=1e-10):
    """Construct an EwDMET bath.
    """

    def part(s):
        en = np.power.outer(mo_energy[s], np.arange(nmom_max_bath+1)).T
        c_env_s = np.dot(c_env.T, mo_coeff[:, s])
        c_frag_s = np.dot(c_frag.T, mo_coeff[:, s])
        bath = lib.einsum("ai,ki,mi->akm", mo_coeff[nfrag:, s], mo_coeff[:nfrag, s], en)
        bath = bath.reshape(-1, nfrag * (nmom_max_bath+1))
        return bath

    bath = np.concatenate([part(mo_occ > 0), part(mo_occ == 0)], axis=-1)  # (env|bath)

    # Normalise (improves stability of orthogonalisation)
    norm = np.linalg.norm(bath, axis=0, keepdims=True)
    norm[np.abs(norm) < 1e-16] = 1e-16
    bath /= norm

    # Orthogonalise
    wt, bath = np.linalg.eigh(np.dot(bath, bath.T))
    bath = bath[:, wt > tol]

    # Include fragment degrees of freedom
    c_bath = np.zeros((nsite, bath.shape[-1]))
    c_bath[nfrag:] = bath

    return c_bath

# Define a function to update the self-energy
def get_se(gf):
    """Compute the self-energy for a given Green's function.
    """

    # Get the bath orbitals
    mo_energy, mo_coeff, mo_occ = gf.as_orbitals()  # (site|MO)
    c_bath = construct_bath(mo_energy, mo_coeff, mo_occ)  # (site|bath)

    # Get the cluster orbitals
    c_act = np.concatenate((c_frag, c_bath), axis=-1)  # (site|frag+bath)
    e_act, c_act = canonicalize_mo(c_act, mf.get_fock())  # (site|clst)
    occ_act = (e_act < chempot).astype(float) * 2

    # Calculate the moments in the cluster
    expr = FCI["1h"](mf, mo_energy=e_act, mo_coeff=c_act, mo_occ=occ_act)
    th = expr.build_gf_moments(nmom_max_fci+1)  # (clst|clst)
    expr = FCI["1p"](mf, mo_energy=e_act, mo_coeff=c_act, mo_occ=occ_act)
    tp = expr.build_gf_moments(nmom_max_fci+1)  # (clst|clst)

    # Run the solver
    solverh = MBLGF(th, log=NullLogger())
    solverp = MBLGF(tp, log=NullLogger())
    solver = MixedMBLGF(solverh, solverp)
    solver.kernel()

    # Get the auxiliaries in the site basis
    se = solver.get_self_energy()  # (clst|aux)
    se.couplings = np.dot(c_act, se.couplings)  # (site|aux)

    # Project out the environment couplings
    # FIXME Is this needed?
    se.couplings = lib.einsum("pi,qi,qj->pj", c_frag, c_frag, se.couplings)

    return se

# Define a function to update the Fock matrix
get_fock = lambda dm: mf.get_fock(dm=dm)

# Run the self-consistent loop
e_tot = mf.e_tot
for cycle in range(1, 21):
    print(f"\nIteration {cycle}")
    print("-" * len(f"Iteration {cycle}"))

    # Update the self-energy
    se = get_se(gf)

    # Update the Green's function, relaxing the density
    nelec = hubbard.nelectron
    solver = DensityRelaxation(get_fock, se, nelec, log=NullLogger())
    solver.kernel()
    gf = solver.get_greens_function()

    # Get the energy, IP, EA, gap
    e_prev = e_tot
    e_tot = util.greens_function_galitskii_migdal(gf.occupied().moment([0, 1]), mf.get_hcore())
    ip = -gf.physical().occupied().energies.max()
    ea = gf.physical().virtual().energies.min()
    gap = ip + ea
    print("E(tot) = {:.8f}".format(e_tot))
    print("IP = {:.8f}".format(ip))
    print("EA = {:.8f}".format(ea))
    print("Gap = {:.8f}".format(gap))

    if abs(e_tot - e_prev) < 1e-10:
        break
