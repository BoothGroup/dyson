"""
Example performing an EwDMET calculation using `dyson` and `vayesta`.
"""

import sys
sys.argv.append("--quiet")  # For Vayesta
import numpy as np
import scipy.linalg
from pyscf import gto, scf, lib, ao2mo
from vayesta.ewf import EWF
from vayesta.core.util import fix_orbital_sign
from vayesta.lattmod import Hubbard1D, LatticeRHF
from vayesta.core.linalg import recursive_block_svd
from dyson import MBLGF, MixedMBLGF, DensityRelaxation, FCI, Lehmann, util, NullLogger
from dyson import DensityRelaxation, AufbauPrinciple, SelfConsistent
np.set_printoptions(edgeitems=1000, linewidth=1000, precision=3)

nsite = 10
u = 4.0
nmom_max_fci = 1
nmom_max_bath = 1
nfrag = 2
tol = 1e-10

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

# Initialise fragment and environment orbitals
c_frag, c_env = np.split(np.eye(nsite), [nfrag], axis=1)  # (site|frag), (site|env)
p_mo_frag = np.linalg.multi_dot((mo_coeff[:nfrag].T, mo_coeff[:nfrag]))  # (MO|MO)
p_mo_env = np.linalg.multi_dot((mo_coeff[nfrag:].T, mo_coeff[nfrag:]))  # (MO|MO)

# Construct fragment Hamiltonian
h1e_frag = np.linalg.multi_dot((c_frag.T, mf.get_hcore(), c_frag))  # (frag|frag)
h2e_frag = ao2mo.kernel(mf._eri, c_frag)  # (frag,frag|frag,frag)

# Initialise a self-energy
se = Lehmann(np.zeros((0,)), np.zeros((nfrag, 0)), chempot=chempot)  # (frag|aux)

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

def tile_se(se, nimage):
    # Block diagonally tile a self-energy
    # TODO: Check 2D and other lattice models.
    e = np.concatenate([se.energies] * nimage, axis=0)
    c = scipy.linalg.block_diag(*([se.couplings] * nimage))
    return Lehmann(e, c, chempot=se.chempot)

# Run the EwDMET calculation
e_tot = mf.e_tot
for cycle in range(1, 11):
    print(f"\nIteration {cycle}")
    print("-" * len(f"Iteration {cycle}"))

    # Get the DM in the environment
    fock = mf.get_fock()
    se_full = tile_se(se, nsite//nfrag)
    fock_ext = se_full.matrix(fock)  # (site+aux|site+aux)
    e, c = np.linalg.eigh(fock_ext)  # (site+aux|QMO)
    # TODO chempot, fock opt over physical space
    # Construct QMO density matrix over full space
    dm = np.dot(c[:, e < se.chempot], c[:, e < se.chempot].T)  # (site+aux|site+aux)
    # Project out both fragment and fragment-local auxiliary degrees of freedom
    #dm[:nfrag, :] = dm[:, :nfrag] = 0
    #dm[nsite:nsite+se.naux, nsite:nsite+se.naux] = 0
    dm_env = np.zeros((nsite-nfrag+se_full.naux-se.naux,) * 2)
    dm_env[:nsite-nfrag, :nsite-nfrag] = dm[nfrag:nsite, nfrag:nsite]
    dm_env[nsite-nfrag:, nsite-nfrag:] = dm[nsite+se.naux:, nsite+se.naux:]
    dm_env[:nsite-nfrag, nsite-nfrag:] = dm[nfrag:nsite, nsite+se.naux:]
    dm_env[nsite-nfrag:, :nsite-nfrag] = dm[nsite+se.naux:, nfrag:nsite]

    # Build the DMET bath orbitals
    eig, r = np.linalg.eigh(dm_env)
    eig, r = eig[::-1], r[:, ::-1]
    c_all = r.copy()
    c_all = fix_orbital_sign(c_all)[0]
    c_dmet = c_all[:, np.logical_and(eig >= tol, eig <= 1-tol)]  # (envsite+envaux|bath)
    c_occenv = c_all[:, eig > 1-tol]  # (envsite+envaux|occenv)
    c_virenv = c_all[:, eig < tol]  # (envsite+envaux|virenv)
    print("DMET: n(bath) = {}, n(occ-env) = {}, n(vir-env) = {}".format(
        c_dmet.shape[1], c_occenv.shape[1], c_virenv.shape[1]))

    # Enlarge space again
    def enlarge(c):
        c_new = np.zeros((nsite+se_full.naux, c.shape[-1]))
        c_new[nfrag:nsite] = c[:(nsite-nfrag)]
        c_new[(nsite+se.naux):] = c[(nsite-nfrag):]
        return c_new
    c_dmet = enlarge(c_dmet)
    c_occenv = enlarge(c_occenv)
    c_virenv = enlarge(c_virenv)

    # Build the EwDMET bath orbitals
    def build_part(c_partenv):
        if not c_partenv.size:
            return np.zeros((nsite+se_full.naux, 0))
        # Span also the fragment space
        c_part = np.zeros((nsite+se_full.naux, nfrag+se.naux+c_partenv.shape[-1]))  # (site+aux|frag+aux+part-env)
        c_part[:nsite, :nfrag] = c_frag
        c_part[nsite:(nsite+se.naux), nfrag:(nfrag+se.naux)] = np.eye(se.naux)
        c_part[:, (nfrag+se.naux):] = c_partenv
        fock_ext_proj = np.linalg.multi_dot((c_part.T, fock_ext, c_part))  # (frag+aux+part-env|frag+aux+part-env)
        r_part, sv_part, orders_part = recursive_block_svd(fock_ext_proj, n=nfrag+se.naux, tol=tol, maxblock=nmom_max_bath)  # (part-env|ewdmet)
        print(sv_part, orders_part)
        c_ewdmet = np.dot(c_partenv, r_part[:, sv_part > tol])  # (site+aux|ewdmet)
        return c_ewdmet
    c_occewdmet = build_part(c_occenv)  # (site+aux|occewdmet)
    c_virewdmet = build_part(c_virenv)  # (site+aux|virewdmet)
    c_bath = np.hstack([c_dmet, c_occewdmet, c_virewdmet])  # (site+aux|bath)
    print("EwDMET: n(bath) = {} ({}o, {}v)".format(c_bath.shape[1], c_occewdmet.shape[1], c_virewdmet.shape[1]))

    # Build the cluster orbitals
    c_cls = np.zeros((nsite+se_full.naux, nfrag+c_bath.shape[-1]))  # (site+aux|frag+bath)
    c_cls[:nsite, :nfrag] = c_frag
    c_cls[:, nfrag:] = c_bath
    print("Cluster size: {}".format(c_cls.shape[-1]))

    # TODO: Optionally semi-canonicalize (after cluster hamiltonian construction?)
    #fock_ext_proj = np.linalg.multi_dot((c_cls.T, fock_ext, c_cls))  # (frag+bath|frag+bath)
    #e_cls, rot = np.linalg.eigh(fock_ext_proj)
    #c_cls = np.dot(c_cls, rot)  # (site+aux|cls)
    #c_cls, signs = fix_orbital_sign(c_cls)
    #c_cls = c_cls[:nsite]  # (site|cls)
    #o_cls = (e_cls < se.chempot).astype(float) * 2
    #print("Cluster size: {} ({}o, {}v)".format(c_cls.shape[-1], np.sum(o_cls > 0), np.sum(o_cls == 0)))

    # Get the Hamiltonian - extended fock matrix in bath, h_core elsewhere.
    h1e = np.linalg.multi_dot((c_cls.T, fock_ext, c_cls))
    h1e[:, :nfrag] = h1e[:nfrag, :] = 0.
    h1e[:nfrag, :nfrag] = np.linalg.multi_dot((c_frag.T, mf.get_hcore(), c_frag))
    h1e[:nfrag, nfrag:] = np.linalg.multi_dot((c_frag.T, mf.get_hcore(), c_bath[:nsite]))
    h1e[nfrag:, :nfrag] = h1e[:nfrag, nfrag:].T
    c_frag_cls = np.dot(c_frag.T, c_cls[:nsite])  # (frag|cls)
    h2e = ao2mo.kernel(h2e_frag, c_frag_cls)

    # Get number of electrons in the cluster.
    # To do this, we need to project dm into the cluster space, trace, x 2.
    # I (unfortunately) DO NOT THINK THIS WILL BE AN INTEGER (other than without auxiliaries - check this)
    # Choose the nearest integer number of electrons (eek...)
    dm_cls = np.linalg.multi_dot((c_cls.T, dm, c_cls))
    nelec_exact = np.trace(dm_cls) * 2
    nelec = int(np.rint(nelec_exact))
    print("Nelec in cluster: {:.6f} (rounded to {})".format(nelec_exact, nelec))

    # Optimize a chemical potential in the fragment space, such that the ground state FCI calculation
    # has the right number of electrons in it. This obviously might be fractional.

    # Get the FCI moments
    expr = FCI["1h"](h1e=h1e, h2e=h2e, nelec=nelec)
    th = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)
    expr = FCI["1p"](h1e=h1e, h2e=h2e, nelec=nelec)
    tp = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)

    # Run the solver
    solverh = MBLGF(th, log=NullLogger())
    solverp = MBLGF(tp, log=NullLogger())
    solver = MixedMBLGF(solverh, solverp)
    solver.kernel()

    # Get the auxiliaries in the site basis
    se = solver.get_self_energy(chempot=se.chempot)  # (cls|aux)
    se.couplings = np.dot(c_frag_cls, se.couplings)  # (frag|aux)

    # Get the energy, IP, EA, gap
    e_prev = e_tot
    e_tot = util.greens_function_galitskii_migdal(th, h1e)
    e, c = solver.get_dyson_orbitals()
    ip = -e[e < se.chempot].min()
    ea = e[e >= se.chempot].min()
    gap = ip + ea
    print("E(tot) = {:.8f}".format(e_tot))
    print("IP = {:.8f}".format(ip))
    print("EA = {:.8f}".format(ea))
    print("Gap = {:.8f}".format(gap))

    if abs(e_tot - e_prev) < 1e-10:
        break

