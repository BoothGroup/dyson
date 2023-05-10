"""
EwDMET with a quasiparticle approximation to the self-consistent self-energy.
"""

import numpy as np
import scipy.linalg
import scipy.optimize
from vayesta.lattmod import Hubbard1D, LatticeRHF
from vayesta.core.linalg import recursive_block_svd
from dyson import Lehmann, FCI, MBLGF, MixedMBLGF, NullLogger, AuxiliaryShift
from pyscf import ao2mo, lib


def tile_se(se, nimage):
    # Block diagonally tile a self-energy
    e = np.concatenate([se.energies] * nimage, axis=0)
    c = scipy.linalg.block_diag(*([se.couplings] * nimage))
    return Lehmann(e, c, chempot=se.chempot)


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
        interacting_bath=False,
        trans_sym=True
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
    interacting_bath : bool, optional
        Whether to include interactions in the bath space or just fragment. Default False.
    trans_sym : bool, optional
        Whether the system has translational symmetry (and therefore we know the fragment occupation a priori).
        Default True.
    """

    # Check arguments
    nsite = mf.mol.nao
    nelec = mf.mol.nelectron
    assert nsite % nfrag == 0
    assert nelec % 2 == 0

    if trans_sym:
        nelec_frag_target = nfrag * nelec / nsite
    else:
        nelec_frag_target = None

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

        # Get the Fock matrix and add the static potential over the lattice
        fock = mf.get_fock()  # (site|site)
        fock += v

        # Find the corresponding density matrix
        e, c = np.linalg.eigh(fock)
        dm = np.dot(c[:, :nelec//2], c[:, :nelec//2].T)  # (site|site)

        nelec_frag = np.trace(np.linalg.multi_dot((c_frag.T, dm, c_frag))) * 2.
        print("Number of fragment electrons on the lattice state: {}".format(nelec_frag))

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
        e_cls, c_cls_canon = canonicalize_mo(c_cls, fock) # (site|cls_canon)
        dm_cls = np.linalg.multi_dot((c_cls_canon.T, dm, c_cls_canon))  # (cls|cls)
        nelec_cls = np.trace(dm_cls) * 2
        print("Cluster: ncls = {}, nelec = {:.6f}".format(c_cls.shape[1], nelec_cls))
        assert(np.isclose(nelec_cls, np.rint(nelec_cls)))

        p_bath = np.linalg.multi_dot((c_cls_canon.T, c_bath, c_bath.T, c_cls_canon))  # (cls|cls)
        c = np.linalg.multi_dot((c_frag, c_frag.T, c_cls_canon))  # (site|cls)

        # Get the Hamiltonian in the cluster
        h1e = np.linalg.multi_dot((c_cls_canon.T, mf.get_hcore(), c_cls_canon))  # (cls|cls)
        if interacting_bath:
            # full interactions and h_core everywhere.
            h2e = ao2mo.kernel(mf._eri, c_cls_canon)
        else:
            # Fock+qp_se in bath, h_core everywhere else. Interactions only in fragment.
            h1e += np.linalg.multi_dot((p_bath, c_cls_canon.T, fock-mf.get_hcore(), c_cls_canon, p_bath))  # (cls|cls)
            h2e = ao2mo.kernel(mf._eri, c)  # (cls,cls|cls,cls)

        # Optimize a chemical potential in the fragment space, such that the ground
        # state *FCI* calculation has the right number of electrons in it. This
        # obviously might be fractional.

        # Define function to get the moments for a given chemical potential
        def get_moments(chempot, dm_only=True):
            # Apply the chemical potential in the bath (this is done in a bit of a silly way!)
            mu = np.diag([np.array(chempot).ravel()[0]] * c_bath.shape[1])  # (bath|bath)
            mu = np.linalg.multi_dot((c_cls_canon.T, c_bath, mu, c_bath.T, c_cls_canon))  # (cls|cls)
            h1e_mu = h1e - mu

            # Get the FCI moments
            if dm_only:
                expr = FCI["1h"](h1e=h1e_mu, h2e=h2e, nelec=int(np.rint(nelec_cls)))
                return expr.build_gf_moments(1).squeeze()  # (cls|cls)
            else:
                expr = FCI["1h"](h1e=h1e_mu, h2e=h2e, nelec=int(np.rint(nelec_cls)))
                th = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)
                expr = FCI["1p"](h1e=h1e_mu, h2e=h2e, nelec=int(np.rint(nelec_cls)))
                tp = expr.build_gf_moments(nmom_max_fci+1)  # (cls|cls)
                return th, tp

        # Define objective function for optimisation
        def obj(chempot):
            # Project the zeroth hole moment into the fragment
            dm = get_moments(chempot, dm_only=True)
            c_frag_canon = np.linalg.multi_dot((c_frag.T, c_cls_canon)) # (frag|cls)
            nelec = np.trace(np.linalg.multi_dot((c, dm, c.T))) * 2
            if nelec_frag_target is None:
                # Get it to equal the number of fragment electrons from lattice description.
                nelec_target = np.trace(np.linalg.multi_dot((c_frag.T, dm, c_frag)))
            else:
                nelec_target = nelec_frag_target

            # Return the squared difference
            return (nelec - nelec_target)**2

        # Optimise
        opt = scipy.optimize.minimize(obj, x0=0.0, method="BFGS")
        th, tp = get_moments(opt.x[0], dm_only=False)
        print("Chemical potential for FCI ground state: {:.6f}".format(opt.x[0]))
        print("Error in nelec in cluster: {:.2e}".format(np.trace(th[0])*2 - nelec_cls))

        # Check number of electrons in the DM
        c_frag_canon = np.linalg.multi_dot((c_frag.T, c_cls_canon))  # (frag|cls)
        nelec_frag_cls = np.trace(np.linalg.multi_dot((c_frag_canon, th[0], c_frag_canon.T))) * 2
        print("Number of fragment electrons from FCI cluster: {}".format(nelec_frag_cls))

        # Run the solver
        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        print("Moment error: {:.2e}".format(solver._check_moment_error()))

        # Get the dynamical self-energy and convert to a static potential
        se = solver.get_self_energy()  # (cls|aux)
        v_cls = se.as_static_potential(e_cls, eta=eta)  # (cls|cls)

        # Rotate into the fragment basis and tile for all fragments
        v_frag = np.linalg.multi_dot((c_frag_canon, v_cls, c_frag_canon.T))  # (frag|frag)
        v_frag = diis.update(v_frag)
        # TODO: Will this only work for the 1D model? Check tiling with other models.
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

    # Get the tiled self-energy
    c = np.linalg.multi_dot((c_frag.T, c_cls_canon))  # (frag|cls)
    se.couplings = np.dot(c, se.couplings)  # (frag|aux)
    # TODO: Will this only work for the 1D model? Check tiling with other models.
    se = tile_se(se, nsite//nfrag)  # (site|aux)

    return se, v


if __name__ == "__main__":
    nsite = 128 # Number of sites
    nelec = 128 # Number of electrons
    u = 4.0  # Hubbard U parameter
    nfrag = 2  # Number of sites per fragment

    # Define the model
    hubbard = Hubbard1D(
            nsite=nsite,
            nelectron=nelec,
            hubbard_u=u,
            verbose=0
    )
    mf = LatticeRHF(hubbard)
    mf.kernel()

    # Run the EwDMET calculation. Return the (dynamic) self-energy, and its qp-approx on the lattice
    se, v = qp_ewdmet_hubbard1d(mf, nfrag=nfrag)
    print()

    # Shift final auxiliaries to ensure right particle number
    shift = AuxiliaryShift(mf.get_fock(), se, nelec, occupancy=2, log=NullLogger())
    shift.kernel()
    se_shifted = shift.get_self_energy()
    print('Final (shifted) auxiliaries: {} ({}o, {}v)'.format(se_shifted.naux, se_shifted.occupied().naux, se_shifted.virtual().naux))

    # Find the Green's function
    gf = Lehmann(*se_shifted.diagonalise_matrix_with_projection(mf.get_fock()), chempot=se_shifted.chempot)
    dm = gf.occupied().moment(0) * 2.0
    nelec_gf = np.trace(dm)
    assert(np.isclose(nelec_gf, gf.occupied().weights(occupancy=2).sum()))
    print('Number of electrons in final (shifted) GF with dynamical self-energy: {}'.format(nelec_gf))
    assert(np.isclose(nelec_gf, float(nelec)))

    # Find qp-Green's function (used to define the self-consistent bath space
    # (and bath effective interactions if not using an interacting bath)
    qp_ham = mf.get_fock() + v
    qp_e, qp_c = np.linalg.eigh(qp_ham)
    qp_mu = (qp_e[nelec//2-1] + qp_e[nelec//2] ) / 2
    gf_qp = Lehmann(qp_e, qp_c, chempot=qp_mu)

    # Plot the spectrum
    from dyson.util import build_spectral_function
    import matplotlib.pyplot as plt
    grid = np.linspace(-5, 5, 1024)
    sf_hf = build_spectral_function(mf.mo_energy, np.eye(mf.mo_occ.size), grid, eta=0.1)
    sf_dynamic = build_spectral_function(gf.energies, gf.couplings, grid, eta=0.1)
    sf_static = build_spectral_function(gf_qp.energies, gf_qp.couplings, grid, eta=0.1)
    plt.plot(grid, sf_hf, "C0-", label="HF")
    plt.plot(grid, sf_dynamic, "C1-", label="QP-EwDMET (dynamic)")
    plt.plot(grid, sf_static, "C2-", label="QP-EwDMET (static)")
    plt.xlabel("Frequency")
    plt.ylabel("Spectral function")
    plt.legend()
    plt.tight_layout()
    plt.show()
