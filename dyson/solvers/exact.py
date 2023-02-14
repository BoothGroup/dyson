"""
Exact eigensolver on the dense upfolded self-energy.
"""

import numpy as np
import scipy.linalg

from dyson import util
from dyson.solvers import BaseSolver


class Exact(BaseSolver):
    """
    Exact eigensolver on the dense upfolded self-energy.

    Input
    -----
    matrix : numpy.ndarray (n, n)
        Dense representation of the upfolded self-energy matrix.

    Parameters
    ----------
    hermitian : bool, optional
        If `True`, the input matrix is assumed to be hermitian,
        otherwise it is assumed to be non-hermitian. Default value
        is `True`.
    overlap : numpy.ndarray, optional
        If provided, use as part of a generalised eigenvalue problem.
        Default value is `None`.

    Returns
    -------
    eigvals : numpy.ndarray (n,)
        Eigenvalues of the matrix, representing the energies of the
        Green's function.
    eigvecs : numpy.ndarray (n, n)
        Eigenvectors of the matrix, which provide the Dyson orbitals
        once projected into the physical space.
    """

    def __init__(self, matrix, **kwargs):
        # Input:
        self.matrix = matrix

        # Parameters:
        self.hermitian = kwargs.pop("hermitian", True)
        self.overlap = kwargs.pop("overlap", None)

        # Base class:
        super().__init__(matrix, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > hermitian:  %s", self.hermitian)
        self.log.info(" > overlap:  %s", None if not self.generalised else type(self.overlap))

    def _kernel(self):
        if self.hermitian:
            eigvals, eigvecs = self._kernel_hermitian()
        else:
            eigvals, eigvecs = self._kernel_nonhermitian()

        self.log.info(util.print_eigenvalues(eigvals))

        return eigvals, eigvecs

    def _kernel_hermitian(self):
        if self.generalised:
            return np.linalg.eigh(self.matrix)
        else:
            return scipy.linalg.eigh(self.matrix, b=self.overlap)

    def _kernel_nonhermitian(self):
        if self.generalised:
            return np.linalg.eig(self.matrix)
        else:
            return scipy.linalg.eig(self.matrix, b=self.overlap)

    @property
    def generalised(self):
        return self.overlap is not None
