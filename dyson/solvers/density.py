"""
Relax the density matrix in the presence of a self-energy.
"""

import numpy as np
from pyscf import lib

from dyson import NullLogger
from dyson.lehmann import Lehmann
from dyson.solvers import AufbauPrinciple, AuxiliaryShift, BaseSolver


class DensityRelaxation(BaseSolver):
    """
    Relax the density matrix in the presence of a self-energy.

    Parameters
    ----------
    get_fock : callable
        Callable that returns the Fock matrix in the MO basis.  Takes
        a density matrix as input.
    se : dyson.lehmann.Lehmann
        Lehmann representation of the self-energy.
    nelec : int
        Number of electrons.
    occupancy : int, optional
        Occupancy of each state, i.e. `2` for a restricted reference
        and `1` for other references.  Default value is `2`.
    chempot_solver : BaseSolver, optional
        Solver for the chemical potential.  One of
        {`dyson.solvers.AufbauPrinciple`,
        `dyson.solvers.AuxiliaryShift`}.  Default value is
        `dyson.solvers.AuxiliaryShift`.
    diis_space : int, optional
        Size of the DIIS space.  Default value is `8`.
    diis_min_space : int, optional
        Minimum size of the DIIS space.  Default value is `2`.
    max_cycle_outer : int, optional
        Maximum number of outer iterations.  Default value is `20`.
    max_cycle_inner : int, optional
        Maximum number of inner iterations.  Default value is `50`.
    conv_tol : float, optional
        Threshold for convergence in the change in the density matrix.
        Default value is `1e-8`.
    """

    def __init__(self, get_fock, se, nelec, **kwargs):
        # Input:
        self._get_fock = get_fock
        self.se = Lehmann.from_pyscf(se)
        self.nelec = nelec

        # Parameters:
        self.occupancy = kwargs.pop("occupancy", 2)
        self.chempot_solver = kwargs.pop("chempot_solver", AuxiliaryShift)
        self.diis_space = kwargs.pop("diis_space", 8)
        self.diis_min_space = kwargs.pop("diis_min_space", 2)
        self.max_cycle_outer = kwargs.pop("max_cycle_outer", 20)
        self.max_cycle_inner = kwargs.pop("max_cycle_inner", 50)
        self.conv_tol = kwargs.pop("conv_tol", 1e-8)

        # Base class:
        super().__init__(**kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > occupancy:  %s", self.occupancy)
        self.log.info(" > chempot_solver:  %s", self.chempot_solver)
        self.log.info(" > diis_space:  %s", self.diis_space)
        self.log.info(" > diis_min_space:  %s", self.diis_min_space)
        self.log.info(" > max_cycle_outer:  %s", self.max_cycle_outer)
        self.log.info(" > max_cycle_inner:  %s", self.max_cycle_inner)
        self.log.info(" > conv_tol:  %s", self.conv_tol)

        # Caching:
        self.converged = False
        self.se_res = None
        self.gf_res = None

    def get_fock(self, rdm1):
        """
        Get the Fock matrix in the MO basis.

        Parameters
        ----------
        rdm1 : numpy.ndarray
            One-particle reduced density matrix in the MO basis.

        Returns
        -------
        fock : numpy.ndarray
            Fock matrix.
        """

        return self._get_fock(rdm1)

    def optimise_chempot(self, se, fock):
        """
        Optimise the chemical potential.

        Parameters
        ----------
        se : dyson.lehmann.Lehmann
            Lehmann representation of the self-energy.
        fock : numpy.ndarray
            Fock matrix.

        Returns
        -------
        se : dyson.lehmann.Lehmann
            Lehmann representation of the self-energy, with the
            chemical potential optimised.
        error : float
            Error in the chemical potential.
        """

        if self.chempot_solver:
            solver = self.chempot_solver(fock, se, self.nelec, guess=se.chempot, log=NullLogger())
            solver.kernel()

            se = solver.get_self_energy()
            error = solver.error
            converged = solver.converged

        else:
            error = 0.0
            converged = True

        return se, error, converged

    def _kernel_rhf(self):
        """
        Perform the self-consistent field for a restricted reference.
        """

        se = self.se
        nocc = self.nelec // self.occupancy
        rdm1 = np.zeros((se.nphys, se.nphys))
        rdm1[:nocc, :nocc] = np.eye(nocc) * self.occupancy
        rdm1_prev = rdm1.copy()
        fock = self.get_fock(rdm1)

        self.log.info("-" * 47)
        self.log.info(
            "{:^6s} {:^6s} {:^16s} {:^16s}".format(
                "Iter",
                "DM iter",
                "DM error",
                "Chempot error",
            )
        )
        self.log.info("%6s %6s %16s %16s" % ("-" * 6, "-" * 6, "-" * 16, "-" * 16))

        for niter_outer in range(1, self.max_cycle_outer + 1):
            se, error_chempot, converged_chempot = self.optimise_chempot(se, fock)

            diis = lib.diis.DIIS()
            diis.space = self.diis_space
            diis.min_space = self.diis_min_space
            diis.verbose = 0

            for niter_inner in range(1, self.max_cycle_inner + 1):
                w, v = se.diagonalise_matrix_with_projection(fock)
                gf = Lehmann(w, v, chempot=se.chempot)

                aufbau = AufbauPrinciple(gf, self.nelec, log=NullLogger())
                aufbau.kernel()
                se.chempot = gf.chempot = aufbau.chempot

                rdm1 = gf.occupied().moment(0) * self.occupancy
                fock = self.get_fock(rdm1)

                try:
                    fock = diis.update(fock, xerr=None)
                except np.linalg.LinAlgError:
                    pass

                error_rdm1 = np.max(np.abs(rdm1 - rdm1_prev))
                self.log.debug("%6d %6d %16.5g", niter_outer, niter_inner, error_rdm1)
                if error_rdm1 < self.conv_tol:
                    break

                rdm1_prev = rdm1.copy()

            self.log.info(
                "%6d %6d %16.5g %16.5g",
                niter_outer,
                niter_inner,
                error_rdm1,
                error_chempot,
            )

            if error_rdm1 < self.conv_tol and abs(aufbau.error) < self.chempot_solver.conv_tol:
                self.converged = True
                break

        self.log.info("-" * 47)

        self.flag_convergence(self.converged)

        self.se_res = se
        self.gf_res = gf

        return gf, se, self.converged

    def _kernel(self):
        """
        Perform the self-consistent field.
        """

        if not isinstance(self.nelec, tuple):
            return self._kernel_rhf()
        else:
            raise NotImplementedError("UHF not implemented.")

    def get_auxiliaries(self):
        return self.se_res.energies, self.se_res.couplings

    def get_dyson_orbitals(self):
        return self.gf_res.energies, self.gf_res.couplings

    def get_self_energy(self):
        return self.se_res

    def get_greens_function(self):
        return self.gf_res
