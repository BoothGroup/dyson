"""
Self-consistent solution to the Dyson equation.
"""

import numpy as np

from dyson import Lehmann, NullLogger
from dyson.solvers import AufbauPrinciple, AuxiliaryShift, BaseSolver, DensityRelaxation


class SelfConsistent(BaseSolver):
    """
    Self-consistent solution to the Dyson equation.

    Parameters
    ----------
    get_se : callable
        Callable that returns the self-energy.  Takes a Green's
        function in the format of a `Lehmann` object as input, which
        provides the basis in which the self-energy is to be
        constructed.
    get_fock : callable
        Callable that returns the Fock matrix.  Takes a density matrix
        in the MO basis as input.  Default value is `None`.
    gf_init : dyson.Lehmann
        Initial guess for the Green's function.
    nelec : int, optional
        Number of electrons.  If not provided, the number is inferred
        from the initial guess for the Green's function.  Default
        value is `None`.
    occupancy : int, optional
        Occupancy of each state, i.e. `2` for a restricted reference
        and `1` for other references.  Default value is `2`.
    relax_solver : dyson.solvers.BaseSolver, optional
        Solver for relaxing the density matrix or chemical potential.
        Must be one of {`None`, `dyson.solvers.AufbauPrinciple`,
        `dyson.solvers.AuxiliaryShift`,
        `dyson.solvers.DensityRelaxation`}.  If provided, the
        `get_fock` argument must be provided.  Default value is
        `None`.
    max_cycle : int, optional
        Maximum number of iterations.  Default value is `50`.
    conv_tol : float, optional
        Convergence threshold in the first moment of the Green's
        function.  Default value is `1e-8`.
    """

    def __init__(self, get_se, get_fock, gf_init, **kwargs):
        # Input:
        self._get_se = get_se
        self._get_fock = get_fock
        self.gf_init = gf_init

        # Parameters:
        self._nelec = kwargs.pop("nelec", None)
        self.occupancy = kwargs.pop("occupancy", 2)
        self.relax_solver = kwargs.pop("relax_solver", None)
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.conv_tol = kwargs.pop("conv_tol", 1e-8)

        # Base class:
        super().__init__(**kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > nelec:  %s", self.nelec)
        self.log.info(" > occupancy:  %s", self.occupancy)
        self.log.info(" > relax_solver:  %s", self.relax_solver)
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > conv_tol:  %s", self.conv_tol)

        # Caching:
        self.converged = False
        self.se_res = None
        self.gf_res = None

    def get_se(self, gf, se_prev=None):
        """
        Update the self-energy using a particular Green's function.

        Parameters
        ----------
        gf : dyson.Lehmann
            Green's function.
        se_prev : dyson.Lehmann, optional
            Previous self-energy.  Default value is `None`.

        Returns
        -------
        se : dyson.Lehmann
            Self-energy.
        """

        return self._get_se(gf)

    def get_fock(self, rdm1):
        """
        Update the Fock matrix using a particular density matrix.

        Parameters
        ----------
        rdm1 : numpy.ndarray
            Density matrix in the MO basis.

        Returns
        -------
        fock : numpy.ndarray
            Fock matrix.
        """

        return self._get_fock(rdm1)

    def _kernel(self):
        """
        Perform the self-consistent solution of the Dyson equation.
        """

        gf = self.gf_init
        gf_prev = gf
        se = self.get_se(gf)
        se_prev = None
        gap = gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]

        self.log.info("-" * 58)
        self.log.info(
            "{:^6s} {:^12s} {:^12s} {:^12s} {:^12s}".format(
                "Iter",
                "Gap",
                "Gap error",
                "Nelec error",
                "Chempot",
            )
        )
        self.log.info(
            "%6s %12s %12s %12s %12s",
            "-" * 6,
            "-" * 12,
            "-" * 12,
            "-" * 12,
            "-" * 12,
        )

        for i in range(1, self.max_cycle + 1):
            gf_prev = gf
            gap_prev = gap

            if self.relax_solver:
                if self.relax_solver is DensityRelaxation:
                    fock = self.get_fock
                else:
                    rdm1 = gf.occupied().moment(0) * self.occupancy
                    fock = self.get_fock(rdm1)

                solver = self.relax_solver(fock, se, self.nelec, log=NullLogger())
                solver.kernel()

                gf = solver.get_greens_function()
                se = solver.get_self_energy()

            else:
                rdm1 = gf.occupied().moment(0) * self.occupancy
                fock = self.get_fock(rdm1)

                w, v = se.diagonalise_matrix_with_projection(fock)
                gf = Lehmann(w, v, chempot=se.chempot)

            se_prev = se.copy()
            se = self.get_se(gf, se_prev=se_prev)

            gap_prev = gap
            ip = -gf.physical().occupied().energies[-1]
            ea = gf.physical().virtual().energies[0]
            gap = ip + ea

            n_error = abs(np.trace(gf.occupied().moment(0)) * self.occupancy - self.nelec)
            gap_error = abs(gap - gap_prev)

            self.log.info(
                "{:6d} {:12.8f} {:12.6g} {:12.6g} {:12.6f}".format(
                    i,
                    gap,
                    gap_error,
                    n_error,
                    gf.chempot,
                )
            )

            if gap_error < self.conv_tol:
                self.converged = True
                break

        self.log.info("-" * 58)

        self.flag_convergence(self.converged)

        self.se_res = se
        self.gf_res = gf

        return gf, se, self.converged

    @property
    def nelec(self):
        """
        Number of electrons.
        """

        if self._nelec is None:
            rdm1 = self.gf_init.occupied().moment(0) * self.occupancy
            self._nelec = int(np.round(np.trace(rdm1)))

        return self._nelec

    def get_auxiliaries(self):
        return self.se_res.energies, self.se_res.couplings

    def get_dyson_orbitals(self):
        return self.gf_res.energies, self.gf_res.couplings

    def get_self_energy(self):
        return self.se_res

    def get_greens_function(self):
        return self.gf_res
