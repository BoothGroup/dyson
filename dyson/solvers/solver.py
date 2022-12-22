"""
Solver base class.
"""

import numpy as np

from dyson import default_log


class BaseSolver:
    """
    Base class for all solvers.
    """

    def __init__(self, *args, **kwargs):
        self.log = kwargs.pop("log", default_log)

        # Check all the arguments have now been consumed:
        if len(kwargs):
            for key, val in kwargs.items():
                self.log.warn("Argument `%s` invalid" % key)

    def kernel(self, *args, **kwargs):
        """
        Driver function. Classes inheriting the `BaseSolver` should
        implement `_kernel`, which is called by this function.
        """

        out = self._kernel(self, *args, **kwargs)

        return out

    def flag_converged(self, converged):
        """Preset logging for convergence message.
        """

        if converged:
            self.log.info("Successfully converged.")
        else:
            self.log.info("Failed to converge.")

