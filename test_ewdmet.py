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

nsite = 16
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
fock = np.linalg.multi_dot((c_frag.T, mf.get_fock(), c_frag))  # (frag frag)

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

# Run the EwDMET calculation
e_tot = mf.e_tot
for cycle in range(1, 11):
    print(f"\nIteration {cycle}")
    print("-" * len(f"Iteration {cycle}"))

    # Get the DM in the environment
    # TODO fock in full site basis, tile local SEs
    e, c = se.diagonalise_matrix(fock)  # (site+aux|QMO)
    # TODO chempot, fock opt over physical space
    c_moaux = np.linalg.multi_dot((mo_coeff, c[:nsite], c.T))  # (site|MO+aux)
    dm = np.dot(c[:, e < se.chempot], c[:, e < se.chempot].T)  # (site+aux|site+aux)
    # TODO project out fragment and local auxiliaries from dm

    # Build the DMET bath orbitals
    eig, r = np.linalg.eigh(dm)
    eig, r = eig[::-1], r[:, ::-1]
    c_all = r.copy()
    c_all = fix_orbital_sign(c_all)[0]
    c_dmet = c_all[:, np.logical_and(eig >= tol, eig <= 1-tol)]  # (site+aux|bath)
    c_occenv = c_all[:, eig > 1-tol]  # (site+aux|occ-env)
    c_virenv = c_all[:, eig < tol]  # (site+aux|vir-env)
    print("DMET: n(bath) = {}, n(occ-env) = {}, n(vir-env) = {}".format(
        c_dmet.shape[1], c_occenv.shape[1], c_virenv.shape[1]))

    # Build the EwDMET bath orbitals
    c_bath = [c_dmet]
    for c_partenv in [c_occenv, c_virenv]:
        if c_partenv.size:
            # Span also the fragment space
            c_part = np.zeros((c_partenv.shape[0], nsite + c_partenv.shape[1]))  # (MO+aux|MO+part-env)
            c_part[:nsite, :nsite] = p_mo_frag
            c_part[:, nsite:] = c_partenv
            fock_ext = se.matrix(fock)  # (MO+aux|MO+aux)
            fock_ext = np.linalg.multi_dot((c_part.T, fock_ext, c_part))  # (MO+part-env|MO+part-env)
            # TODO n should be frag+aux, fock_ext spans frag+aux+occenv
            r_part, sv_part, orders_part = recursive_block_svd(fock_ext, n=nsite, tol=tol, maxblock=nmom_max_bath)
            c_partewdmet = np.dot(c_partenv, r_part[:, sv_part > tol])  # (MO+aux|part-ewdmet)
            c_bath.append(c_partewdmet)
    c_bath = np.hstack(c_bath)  # (MO+aux|bath)
    c_bath = np.dot(c_moaux, c_bath)  # (site|bath)
    print("EwDMET: n(bath) = {}".format(c_bath.shape[1]))

    # Build the cluster orbitals
    c_cls = np.hstack((c_frag, c_bath))  # (site|cls)
    fock_proj = np.linalg.multi_dot((c_cls.T, fock, c_cls))  # (cls|cls)
    e_cls, rot = np.linalg.eigh(fock_proj)
    c_cls = np.dot(c_cls, rot)
    c_cls, signs = fix_orbital_sign(c_cls)
    o_cls = (e_cls < se.chempot).astype(float) * 2
    print("Cluster size: {} ({}o, {}v)".format(c_cls.shape[-1], np.sum(o_cls > 0), np.sum(o_cls == 0)))

    # Get the Hamiltonian  # TODO won't scale
    h1e = np.linalg.multi_dot((c_cls.T, mf.get_hcore(), c_cls))
    h2e = ao2mo.kernel(mf._eri, c_cls)

    # Get the FCI moments
    expr = FCI["1h"](h1e=h1e, h2e=h2e, nelec=np.sum(e_cls < se.chempot) * 2)
    th = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)
    expr = FCI["1p"](h1e=h1e, h2e=h2e, nelec=np.sum(e_cls < se.chempot) * 2)
    tp = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)

    # Run the solver
    solverh = MBLGF(th, log=NullLogger())
    solverp = MBLGF(tp, log=NullLogger())
    solver = MixedMBLGF(solverh, solverp)
    solver.kernel()

    # Get the auxiliaries in the site basis
    se = solver.get_self_energy(chempot=se.chempot)  # (cls|aux)
    se.couplings = np.linalg.multi_dot((mo_coeff.T, c_cls, se.couplings))  # (MO|aux)

    # Project out the environment couplings
    se.couplings = np.dot(p_mo_frag, se.couplings)  # (MO|aux)

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

