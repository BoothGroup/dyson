"""
Solver base class.
"""

import numpy as np

from dyson import default_log, init_logging


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
