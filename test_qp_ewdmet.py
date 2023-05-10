"""
EwDMET with a quasiparticle approximation.
"""

import numpy as np
import scipy.linalg
from vayesta.lattmod import Hubbard1D, LatticeRHF
from vayesta.core.linalg import recursive_block_svd
from dyson import Lehmann, FCI, MBLGF, MixedMBLGF, NullLogger
from pyscf import ao2mo, lib


def qp_ewdmet_hubbard1d(
        mf,
        nmom_max_fci=1,
        nmom_max_bath=1,
        nfrag=2,
        bath_tol=1e-10,
        max_cycles=30,
        conv_tol=1e-7,
        diis_space=10,
        eta=1e-2,
        v_init=None,
):
    """
    Run the QP-EwDMET calculation for the 1D Hubbard model.

    Parameters
    ----------
    mf : LatticeRHF
        Mean-field object.
    nmom_max_fci : int, optional
        Maximum moment order to used for MBLGF. Default is 1.
    nmom_max_bath : int, optional
        Maximum EwDMET bath order. Default is 1.
    nfrag : int, optional
        Number of sites per fragment. Default is 2.
    bath_tol : float, optional
        Threshold for bath. Default is 1e-10.
    max_cycles : int, optional
        Maximum number of iterations. Default is 30.
    conv_tol : float, optional
        Convergence threshold in the total energy. Default is 1e-7.
    diis_space : int, optional
        Size of the DIIS space. Default is 10.
    eta : float, optional
        Broadening parameter for the static potential. Default is 1e-2.
    v_init : ndarray, optional
        Initial static potential. Default is None.
    """

    # Check arguments
    nsite = mf.mol.nao
    nelec = mf.mol.nelectron
    assert nsite % nfrag == 0
    assert nelec % 2 == 0

    # Initialise rotations into fragment and environment
    c_frag, c_env = np.split(np.eye(nsite), [nfrag], axis=1)  # (site|frag), (site|env)

    # Initialise a self-energy as a static potential
    v = v_init if v_init is not None else np.zeros((nsite, nsite))  # (site|site)

    # Initialise a DIIS object to aid convergence
    diis = lib.diis.DIIS()
    diis.max_space = diis_space

    # Function to canonicalise cluster orbitals
    def canonicalize_mo(mo_coeff, fock):
        fock = np.linalg.multi_dot((mo_coeff.T, fock, mo_coeff))
        mo_energy, rot = np.linalg.eigh(fock)
        mo_coeff_canon = np.dot(mo_coeff, rot)
        #mo_coeff_canon, signs = fix_orbital_sign(mo_coeff_canon)
        #rot = rot * signs[None]
        assert np.allclose(np.dot(mo_coeff, rot), mo_coeff_canon)
        assert np.allclose(np.dot(mo_coeff_canon, rot.T), mo_coeff)
        return mo_energy, mo_coeff_canon

    # Print stuff
    print(f"\nSystem")
    print("-" * len(f"System"))
    print(f"Number of sites: {nsite}")
    print(f"U: {mf.mol.hubbard_u}")
    print(f"Maximum order FCI moments: {nmom_max_fci}")
    print(f"Maximum order bath moments: {nmom_max_bath}")
    print(f"Fragment size: {nfrag}")
    print(f"Environment size: {nsite - nfrag}")
    print(f"Bath size: {nfrag * (2*nmom_max_bath+1)}")
    print(f"Cluster size: {nfrag * (2*nmom_max_bath+2)}")
    print(f"\nHartree-Fock")
    print("-" * len(f"Hartree-Fock"))
    print("Nelec: {}".format(nelec))
    print("E(tot) = {:.8f}".format(mf.e_tot))
    print("IP = {:.8f}".format(-mf.mo_energy[nelec//2-1]))
    print("EA = {:.8f}".format(mf.mo_energy[nelec//2]))
    print("Gap = {:.8f}".format(mf.mo_energy[nelec//2] - mf.mo_energy[nelec//2-1]))

    # Run the calculation
    e_tot = mf.e_tot
    for cycle in range(1, max_cycles+1):
        print(f"\nIteration {cycle}")
        print("-" * len(f"Iteration {cycle}"))

        # Get the Fock matrix and add the static potential
        fock = mf.get_fock()  # (site|site)
        fock += v

        # Find the corresponding density matrix
        e, c = np.linalg.eigh(fock)
        dm = np.dot(c[:, :nelec//2], c[:, :nelec//2].T)  # (site|site)

        # Build the DMET bath orbitals
        dm_env = np.linalg.multi_dot((c_env.T, dm, c_env))  # (env|env)
        eig, r = np.linalg.eigh(dm_env)
        r = np.dot(c_env, r)  # (site|env)
        c_dmet = r[:, np.logical_and(eig > bath_tol, eig < 1-bath_tol)]  # (site|dmet)
        c_occenv = r[:, eig > 1-bath_tol]  # (site|occenv)
        c_virenv = r[:, eig < bath_tol]  # (site|virenv)
        print("DMET: nbath = {}, nenv = {} ({}o, {}v)".format(c_dmet.shape[1], c_occenv.shape[1]+c_virenv.shape[1], c_occenv.shape[1], c_virenv.shape[1]))

        # Build the EwDMET bath orbitals
        def build_part(c_partenv):
            if not c_partenv.size:
                return np.zeros((nsite, 0))
            c = np.hstack((c_frag, c_partenv))
            fock_part = np.linalg.multi_dot((c.T, fock, c))  # (frag+partenv|frag+partenv)
            r, sv, orders = recursive_block_svd(fock_part, n=nfrag, tol=bath_tol, maxblock=nmom_max_bath)  # (partenv|partenv)
            c_ewdmet = np.dot(c_partenv, r[:, sv > bath_tol])  # (site|ewdmet)
            return c_ewdmet
        c_occewdmet = build_part(c_occenv)  # (site|occewdmet)
        c_virewdmet = build_part(c_virenv)  # (site|virewdmet)
        c_bath = np.hstack((c_dmet, c_occewdmet, c_virewdmet))  # (site|bath)
        print("EwDMET: nbath = {} ({}o, {}v)".format(c_occewdmet.shape[1]+c_virewdmet.shape[1], c_occewdmet.shape[1], c_virewdmet.shape[1]))
        print("Combined: nbath = {}".format(c_bath.shape[1]))

        # Build the cluster orbitals and canonicalise them
        c_cls = np.hstack((c_frag, c_bath))  # (site|cls)
        e_cls, c_cls = canonicalize_mo(c_cls, fock)
        dm_cls = np.linalg.multi_dot((c_cls.T, dm, c_cls))  # (cls|cls)
        nelec_cls = np.trace(dm_cls) * 2
        print("Cluster: ncls = {}, nelec = {:.6f}".format(c_cls.shape[1], nelec_cls))

        # Get the Hamiltonian in the cluster
        c = np.linalg.multi_dot((c_frag, c_frag.T, c_cls))  # (site|cls)
        p_bath = np.linalg.multi_dot((c.T, c_bath, c_bath.T, c))  # (cls|cls)
        h1e = np.linalg.multi_dot((c.T, mf.get_hcore(), c))  # (cls|cls)
        h1e += np.linalg.multi_dot((p_bath, c.T, fock-mf.get_hcore(), c, p_bath))  # (cls|cls)
        h2e = ao2mo.kernel(mf._eri, c)  # (cls,cls|cls,cls)

        # Get the FCI moments
        expr = FCI["1h"](h1e=h1e, h2e=h2e, nelec=int(np.rint(nelec_cls)))
        th = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)
        expr = FCI["1p"](h1e=h1e, h2e=h2e, nelec=int(np.rint(nelec_cls)))
        tp = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)

        # Run the solver
        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        print("Moment error: {:.2e}".format(solver._check_moment_error()))

        # Get the self-energy and convert to a static potential
        se = solver.get_self_energy()  # (cls|aux)
        v_cls = se.as_static_potential(e_cls, eta=eta)  # (cls|cls)

        # Rotate into the fragment basis and tile for all fragments
        c = np.linalg.multi_dot((c_frag.T, c_cls))  # (frag|cls)
        v_frag = np.linalg.multi_dot((c, v_cls, c.T))  # (frag|frag)
        v_frag = diis.update(v_frag)
        v = scipy.linalg.block_diag(*[v_frag] * (nsite // nfrag))  # (site|site)

        # Get the energy, IP, EA, gap
        fock = mf.get_fock() + v
        e, c = np.linalg.eigh(fock)
        dm = np.dot(c[:, :nelec//2], c[:, :nelec//2].T)
        e_prev = e_tot
        e_tot = mf.energy_tot(dm=dm, vhf=fock)
        ip = -e[nelec//2-1]
        ea = e[nelec//2]
        gap = ip + ea
        print("E(tot) = {:.8f} (ΔE = {:.2e})".format(e_tot, e_tot - e_prev))
        print("IP = {:.8f}, EA = {:.8f}, Gap = {:.8f}".format(ip, ea, gap))

        # Check for convergence
        if abs(e_tot - e_prev) < conv_tol:
            break


if __name__ == "__main__":
    nsite = 10  # Number of sites
    nelec = 10  # Number of electrons
    u = 2.0  # Hubbard U parameter
    nfrag = 2  # Number of sites per fragment

    # Define the model
    hubbard = Hubbard1D(
            nsite=nsite,
            nelectron=nelec,
            hubbard_u=u,
            verbose=0,
    )
    mf = LatticeRHF(hubbard)
    mf.kernel()

    # Run the EwDMET calculation
    qp_ewdmet_hubbard1d(mf, nfrag=nfrag)
