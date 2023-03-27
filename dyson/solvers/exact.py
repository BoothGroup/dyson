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
        self.nphys = kwargs.pop("nphys", None)

        # Base class:
        super().__init__(matrix, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > hermitian:  %s", self.hermitian)
        self.log.info(" > overlap:  %s", None if not self.generalised else type(self.overlap))
        self.log.info(" > nphys:  %s", self.nphys)

        # Caching:
        self.eigvals = None
        self.eigvecs = None

    def _kernel(self):
        if self.hermitian:
            eigvals, eigvecs = self._kernel_hermitian()
        else:
            eigvals, eigvecs = self._kernel_nonhermitian()

        self.eigvals = eigvals
        self.eigvecs = eigvecs

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

    def get_dyson_orbitals(self):
        if self.nphys is None:
            raise ValueError("`nphys` must be set to use `Exact.get_dyson_orbitals`")

        return super().get_dyson_orbitals()

    def get_auxiliaries(self):
        if self.nphys is None:
            raise ValueError("`nphys` must be set to use `Exact.get_dyson_orbitals`")

        energies = self.matrix[:self.nphys, self.nphys:]

        if self.hermitian:
            couplings = self.matrix[:self.nphys, self.nphys:]
        else:
            couplings_l = self.matrix[:self.nphys, self.nphys:]
            couplings_r = self.matrix[self.nphys:, :self.nphys].conj().T
            couplings = (couplings_l, couplings_r)

        return energies, couplings

    @property
    def generalised(self):
        return self.overlap is not None
