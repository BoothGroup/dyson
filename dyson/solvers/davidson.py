"""
Davidson eigensolver using the matrix-vector operation on the
upfolded self-energy.

Interfaces pyscf.lib.
"""

try:
    from pyscf import lib
except ImportError:
    lib = None
import numpy as np

from dyson import util
from dyson.solvers import BaseSolver

# TODO abs picker


class Davidson(BaseSolver):
    """
    Davidson eigensolver using the matrix-vector operation on the
    upfolded self-energy.

    Input
    -----
    matvec : callable
        Function returning the result of the dot-product of the
        upfolded self-energy with an arbitrary state vector. Input
        arguments are `vector`.
    diagonal : numpy.ndarray (n,)
        Diagonal entries of the upfolded self-energy to precondition
        the solver.

    Parameters
    ----------
    nroots : int, optional
        Number of roots to solve for. Default value is `5`.
    picker : callable, optional
        Function to pick eigenvalues. Input arguments are `eigvals`,
        `eigvecs`, `nroots`, `**env`. Default value is
        `pyscf.lib.pick_real_eigs`.
    guess : numpy.ndarray, optional
        Guess vector. If not `None`, the diagonal is used to construct
        a guess based on `diag`. Default value is `None`.
    max_cycle : int, optional
        Maximum number of iterations. Default value is `50`.
    max_space : int, optional
        Maximum number of trial vectors to store. Default value is
        `12`.
    conv_tol : float, optional
        Threshold for convergence. Default value is `1e-12`.
    hermitian : bool, optional
        If `True`, the input matrix is assumed to be hermitian,
        otherwise it is assumed to be non-hermitian. Default value
        is `False`.

    Returns
    -------
    eigvals : numpy.ndarray (nroots,)
        Eigenvalues of the matrix, representing the energies of the
        Green's function.
    eigvecs : numpy.ndarray (n, nroots)
        Eigenvectors of the matrix, which provide the Dyson orbitals
        once projected into the physical space.
    """

    def __init__(self, matvec, diagonal, **kwargs):
        # Input:
        self.matvec = matvec
        self.diagonal = diagonal

        if lib is None:
            raise ImportError("PySCF installation required for %s." % self.__class__)

        # Parameters
        self.nroots = kwargs.pop("nroots", 5)
        self.picker = kwargs.pop("picker", lib.pick_real_eigs)
        self.guess = kwargs.pop("guess", None)
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.max_space = kwargs.pop("max_space", 12)
        self.conv_tol = kwargs.pop("conv_tol", 1e-8)
        self.hermitian = kwargs.pop("hermitian", True)

        # Base class:
        super().__init__(matvec, diagonal, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > nroots:  %s", self.nroots)
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > max_space:  %s", self.max_space)
        self.log.info(" > conv_tol:  %s", self.conv_tol)
        self.log.info(" > hermitian:  %s", self.hermitian)

    def _kernel(self):
        if self.hermitian:
            convs, eigvals, eigvecs = self._kernel_hermitian()
        else:
            convs, eigvals, eigvecs = self._kernel_nonhermitian()

        self.log.info(util.print_eigenvalues(eigvals, nroots=self.nroots))

        return eigvals, eigvecs

    def _kernel_hermitian(self):
        matvecs = lambda vs: [self.matvec(v) for v in vs]

        guess = self.guess
        if guess is None:
            args = np.argsort(np.abs(self.diagonal))
            guess = np.zeros((self.nroots, self.diagonal.size))
            for root, idx in enumerate(args[: self.nroots]):
                guess[root, idx] = 1.0

        convs, eigvals, eigvecs = lib.davidson1(
            lambda vs: [self.matvec(v) for v in vs],
            guess,
            self.diagonal,
            pick=self.picker,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=self.nroots,
            verbose=0,
        )
        eigvals = np.array(eigvals)
        eigvecs = np.array(eigvecs).T

        return convs, eigvals, eigvecs

    def _kernel_nonhermitian(self):
        matvecs = lambda vs: [self.matvec(v) for v in vs]

        guess = self.guess
        if guess is None:
            args = np.argsort(np.abs(self.diagonal))
            guess = np.zeros((self.nroots, self.diagonal.size))
            for root, idx in enumerate(args[: self.nroots]):
                guess[root, idx] = 1.0

        convs, eigvals, eigvecs = lib.davidson_nosym1(
            lambda vs: [self.matvec(v) for v in vs],
            guess,
            self.diagonal,
            pick=self.picker,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=self.nroots,
            verbose=0,
        )
        eigvals = np.array(eigvals)
        eigvecs = np.array(eigvecs).T

        return convs, eigvals, eigvecs
