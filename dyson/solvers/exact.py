"""
Exact eigensolver on the dense upfolded matrix.
"""

import numpy as np

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

    class __init__(self, matrix, **kwargs):
        # Input:
        self.matrix = matrix

        # Parameters:
        self.hermitian = kwargs.pop("hermitian", True)

        # Base class:
        super().__init__(*args, **kwargs)

    def _kernel(self):
        if self.hermitian:
            return self._kernel_hermitian()
        else:
            return self._kernel_nonhermitian()

    def _kernel_hermitian(self):
        return np.linalg.eigh(self.matrix)

    def _kernel_nonhermitian(self):
        return np.linalg.eig(self.matrix)
