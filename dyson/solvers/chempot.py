"""
Chemical potential optimisation.
"""

import numpy as np
import scipy.optimize

from dyson import NullLogger
from dyson.lehmann import Lehmann
from dyson.solvers import MBLGF, MBLSE, BaseSolver


class AufbauPrinciple(BaseSolver):
    """
    Fill a series of orbitals according to the Aufbau principle.

    Parameters
    ----------
    *args : tuple
        Input arguments.  Either `(gf, nelec)` or `(fock, se, nelec)`
        where `gf` is the Lehmann representation of the Green's
        function, `fock` is the Fock matrix, `se` is the Lehmann
        representation of the self-energy and `nelec` is the number
        of electrons in the physical space.
    occupancy : int, optional
        Occupancy of each state, i.e. `2` for a restricted reference
        and `1` for other references.  Default value is `2`.
    """

    # Default parameters:
    occupancy = 2

    def __init__(self, *args, **kwargs):
        # Input:
        if len(args) == 2:
            gf, nelec = args
            self.se = None
            self.gf = Lehmann.from_pyscf(gf)
            self.nelec = nelec
        else:
            fock, se, nelec = args
            self.se = Lehmann.from_pyscf(se)
            w, v = self.se.diagonalise_matrix_with_projection(fock)
            self.gf = Lehmann(w, v, chempot=se.chempot)
            self.nelec = nelec

        # Parameters:
        self.occupancy = kwargs.pop("occupancy", self.occupancy)

        # Base class:
        super().__init__(self, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > occupancy:  %s", self.occupancy)

        # Caching:
        self.converged = False
        self.homo = None
        self.lumo = None
        self.chempot = None
        self.error = None

    def _kernel(self):
        energies = self.gf.energies
        couplings_l, couplings_r = self.gf._unpack_couplings()

        sum0 = sum1 = 0.0
        for i in range(self.gf.naux):
            n = np.dot(couplings_l[:, i], couplings_r[:, i].conj()).real
            n *= self.occupancy
            sum0, sum1 = sum1, sum1 + n

            self.log.debug("Number of electrons [0:%d] = %.6f", i + 1, sum1)

            if i:
                if sum0 <= self.nelec and self.nelec <= sum1:
                    break

        if abs(sum0 - self.nelec) < abs(sum1 - self.nelec):
            homo = i - 1
            error = self.nelec - sum0
        else:
            homo = i
            error = self.nelec - sum1

        try:
            lumo = homo + 1
            chempot = 0.5 * (energies[homo] + energies[lumo])
        except:
            raise ValueError("Failed to find Fermi energy.")

        self.log.info("HOMO = %.6f", energies[homo])
        self.log.info("LUMO = %.6f", energies[lumo])
        self.log.info("Chemical potential = %.6f", chempot)
        self.log.info("Error in nelec = %.3g", error)

        self.converged = True
        self.homo = energies[homo]
        self.lumo = energies[lumo]
        self.chempot = chempot
        self.error = error

        return chempot, error

    def get_auxiliaries(self):
        if self.se is None:
            raise ValueError("`AufbauPrinciple` was initialised with a Green's function.")
        return self.se.energies, self.se.couplings

    def get_dyson_orbitals(self):
        return self.gf.energies, self.gf.couplings

    def get_self_energy(self):
        if self.se is None:
            raise ValueError("`AufbauPrinciple` was initialised with a Green's function.")
        return self.se.copy(chempot=self.chempot, deep=False)

    def get_greens_function(self):
        return self.gf.copy(chempot=self.chempot, deep=False)


class AufbauPrincipleBisect(AufbauPrinciple):

    def _kernel(self):
        energies = self.gf.energies
        couplings_l, couplings_r = self.gf._unpack_couplings()
        weights = self.gf.weights()
        low, high = 0, self.gf.naux
        mid = high // 2
        for iter in range(100):
            sum = self.occupancy * weights[:mid].sum()
            self.log.debug("Number of electrons [0:%d] = %.6f", iter + 1, sum)
            if sum < self.nelec:
                low = mid
                mid = mid + (high - low) // 2
            else:
                high = mid
                mid = mid - (high - low) // 2

            if low == mid or high == mid:
                break

        n_low, n_high = self.occupancy * weights[:low].sum(), self.occupancy * weights[:high].sum()

        if abs(n_low - self.nelec) < abs(n_high - self.nelec):
            homo = low - 1
            error = self.nelec - n_low
        else:
            homo = high - 1
            error = self.nelec - n_high

        try:
            lumo = homo + 1
            chempot = 0.5 * (energies[homo] + energies[lumo])
        except:
            raise ValueError("Failed to find Fermi energy.")
        self.log.info("HOMO LUMO %s %s" % (homo, lumo))
        self.log.info("HOMO = %.6f", energies[homo])
        self.log.info("LUMO = %.6f", energies[lumo])
        self.log.info("Chemical potential = %.6f", chempot)
        self.log.info("Error in nelec = %.3g", error)

        self.converged = True
        self.homo = energies[homo]
        self.lumo = energies[lumo]
        self.chempot = chempot
        self.error = error

        return chempot, error


class AuxiliaryShift(BaseSolver):
    """
    Shift the self-energy auxiliaries with respect to the Green's
    function, operating on a MBLSE or MBLGF solver.

    Parameters
    ----------
    fock : numpy.ndarray
        Fock matrix.
    se : dyson.lehmann.Lehmann
        Lehmann representation of the self-energy.
    nelec : int
        Number of electrons in the physical space.
    occupancy : int, optional
        Occupancy of each state, i.e. `2` for a restricted reference
        and `1` for other references.  Default value is `2`.
    max_cycle : float, optional
        Maximum number of iterations.  Default value is `200`.
    conv_tol : float, optional
        Threshold for convergence in the number of electrons.  Default
        value is `1e-8`.
    guess : float, optional
        Initial guess for the shift.  Default value is 0.0.
    """

    # Default parameters:
    occupancy = 2
    max_cycle = 200
    conv_tol = 1e-8
    guess = 0.0

    def __init__(self, fock, se, nelec, **kwargs):
        # Input:
        self.fock = fock
        self.se = Lehmann.from_pyscf(se)
        self.nelec = nelec

        # Parameters:
        self.occupancy = kwargs.pop("occupancy", self.occupancy)
        self.max_cycle = kwargs.pop("max_cycle", self.max_cycle)
        self.conv_tol = kwargs.pop("conv_tol", self.conv_tol)
        self.guess = kwargs.pop("guess", self.guess)

        # Base class:
        super().__init__(self, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > occupancy:  %s", self.occupancy)
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > conv_tol:  %s", self.conv_tol)
        self.log.info(" > guess:  %s", self.guess)

        # Caching:
        self.iteration = 0
        self.converged = False
        self.shift = 0.0
        self.chempot = None
        self.error = None

    def objective(self, x, fock=None, out=None):
        """Objective function."""

        if fock is None:
            fock = self.fock

        e, c = self.se.diagonalise_matrix_with_projection(fock, chempot=np.ravel(x)[0], out=out)
        gf = Lehmann(e, c)

        aufbau = AufbauPrinciple(gf, self.nelec, occupancy=self.occupancy, log=NullLogger())
        aufbau.conv_tol = self.conv_tol
        aufbau.kernel()

        return aufbau.error**2

    def gradient(self, x, fock=None, out=None):
        """Gradient of the objective function."""

        if fock is None:
            fock = self.fock

        e, c = self.se.diagonalise_matrix(fock, chempot=np.ravel(x)[0], out=out)
        if self.se.hermitian:
            c_phys = c[: self.se.nphys]
            gf = Lehmann(e, c_phys)
        else:
            c_phys = (
                c[: self.se.nphys],
                np.linalg.inv(c).T.conj()[: self.se.nphys],
            )
            gf = Lehmann(e, c_phys)

        aufbau = AufbauPrinciple(gf, self.nelec, occupancy=self.occupancy, log=NullLogger())
        aufbau.conv_tol = self.conv_tol
        gf.chempot, error = aufbau.kernel()

        gf_occ = gf.occupied()
        gf_vir = gf.virtual()

        nphys = gf.nphys
        nocc = np.sum(gf.energies < gf.chempot)

        h1 = -np.dot(c[gf.nphys :, gf_occ.naux :].conj().T, c[gf.nphys :, : gf_occ.naux])
        z = h1 / (gf_vir.energies[:, None] - gf_occ.energies[None])

        c_occ = np.dot(gf_vir.couplings, z)
        d_rdm1 = np.dot(c_occ, c_occ.T.conj()) * 4.0

        dif = np.trace(d_rdm1).real * error * self.occupancy

        return error**2, dif

    def callback(self, xk):
        self.log.info("Iteration %d: Chemical potential = %.6f", self.iteration, xk)
        self.iteration += 1

    def _kernel(self):
        opt = scipy.optimize.minimize(
            self.gradient,
            x0=self.guess,
            method="TNC",
            jac=True,
            options=dict(
                maxfun=self.max_cycle,
                ftol=self.conv_tol**2,
                xtol=0,
                gtol=0,
            ),
            callback=self.callback,
        )

        shift = -opt.x
        se = self.se.copy()
        se.energies += shift

        aufbau = AufbauPrinciple(
            self.fock, se, self.nelec, occupancy=self.occupancy, log=NullLogger()
        )
        aufbau.conv_tol = self.conv_tol
        aufbau.kernel()

        self.log.info("Auxiliary shift = %.6f", shift)
        self.log.info("Chemical potential = %.6f", aufbau.chempot)
        self.log.info("Error in nelec = %.3g", aufbau.error)
        self.flag_convergence(opt.success)

        self.converged = opt.success
        self.shift = shift
        self.chempot = aufbau.chempot
        self.error = aufbau.error

        return self.chempot, aufbau.error

    def get_auxiliaries(self):
        return self.se.energies, self.se.couplings

    def get_dyson_orbitals(self):
        return self.gf.energies, self.gf.couplings

    def get_self_energy(self):
        se = self.se.copy(chempot=self.chempot, deep=False)
        se.energies = se.energies.copy() + self.shift
        return se

    def get_greens_function(self):
        se = self.get_self_energy()
        w, v = se.diagonalise_matrix_with_projection(self.fock)
        gf = Lehmann(w, v, chempot=self.chempot)
        return gf
