"""
Solver base class.
"""

import numpy as np

from dyson import default_log, init_logging
from dyson import Lehmann


class BaseSolver:
    """
    Base class for all solvers.
    """

    def __init__(self, *args, **kwargs):
        self.log = kwargs.pop("log", default_log)
        init_logging(self.log)
        self.log.info("")
        self.log.info("%s", self.__class__.__name__)
        self.log.info("%s", "*" * len(self.__class__.__name__))

        # Check all the arguments have now been consumed:
        if len(kwargs):
            for key, val in kwargs.items():
                self.log.warn("Argument `%s` invalid" % key)

    def kernel(self, *args, **kwargs):
        """
        Driver function. Classes inheriting the `BaseSolver` should
        implement `_kernel`, which is called by this function. If
        the solver has a `_cache`, this function clears it.
        """

        out = self._kernel(*args, **kwargs)

        # Clear the cache if it is used:
        if hasattr(self, "_cache"):
            self._cache.clear()

        return out

    def flag_convergence(self, converged):
        """Preset logging for convergence message."""

        if converged:
            self.log.info("Successfully converged.")
        else:
            self.log.info("Failed to converge.")

    def get_auxiliaries(self, *args, **kwargs):
        """
        Return the auxiliary energies and couplings.
        """

        raise NotImplementedError

    def get_dyson_orbitals(self, *args, **kwargs):
        """
        Return the Dyson orbitals and their energies.
        """

        eigvals, eigvecs = self.get_eigenfunctions(*args, **kwargs)

        if self.hermitian:
            eigvecs = eigvecs[: self.nphys]
        elif isinstance(eigvecs, tuple):
            eigvecs = (eigvecs[0][: self.nphys], eigvecs[1][: self.nphys])
        else:
            eigvecs = (eigvecs[: self.nphys], np.linalg.inv(eigvecs).T.conj()[: self.nphys])

        return eigvals, eigvecs

    def get_eigenfunctions(self, *args, **kwargs):
        """
        Return the eigenvalues and eigenfunctions.
        """

        return self.eigvals, self.eigvecs

    def get_self_energy(self, *args, chempot=0.0, **kwargs):
        """
        Get the self-energy in the format of `pyscf.agf2`.
        """

        return Lehmann(*self.get_auxiliaries(*args, **kwargs), chempot=chempot)

    def get_greens_function(self, *args, chempot=0.0, **kwargs):
        """
        Get the Green's function in the format of `pyscf.agf2`.
        """

        return Lehmann(*self.get_dyson_orbitals(*args, **kwargs), chempot=chempot)
