"""
Exact eigensolver on the dense upfolded matrix.
"""

import numpy as np

from dyson import util
from dyson.solvers import BaseSolver


class Exact(BaseSolver):
    """
    Exact eigensolver on the dense upfolded matrix.

    Input
    -----
    matrix : numpy.ndarray (n, n)
        Dense representation of the self-energy matrix.

    Parameters
    ----------
    hermitian : bool, optional
        If `True`, the input matrix is assumed to be hermitian,
        otherwise it is assumed to be non-hermitian. Default value
        is `True`.

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

        # Base class:
        super().__init__(matrix, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > hermitian:  %s", self.hermitian)

    def _kernel(self):
        if self.hermitian:
            eigvals, eigvecs = self._kernel_hermitian()
        else:
            eigvals, eigvecs = self._kernel_nonhermitian()

        self.log.info(util.print_eigenvalues(eigvals))

        return eigvals, eigvecs

    def _kernel_hermitian(self):
        return np.linalg.eigh(self.matrix)

    def _kernel_nonhermitian(self):
        return np.linalg.eig(self.matrix)
