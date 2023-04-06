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


def pick_real_eigs(w, v, nroots, env, threshold=1e-3):
    """Pick real eigenvalues, sorting by absolute value.
    """

    iabs = np.abs(w.imag)
    tol = max(threshold, np.sort(iabs)[min(w.size, nroots) - 1])
    idx = np.where(iabs <= tol)[0]
    num = np.count_nonzero(iabs[idx] < threshold)

    if num < nroots and w.size >= nroots:
        warnings.warn(
                "Only %d eigenvalues (out of %3d requested roots) with imaginary part < %4.3g.\n"
                % (num, min(w.size, nroots), threshold),
        )

    real_eigenvectors = env.get("dtype") == np.float64
    w, v, idx = lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_eigenvectors=real_eigenvectors)

    mask = np.argsort(np.abs(w))
    w = w[mask]
    v = v[:, mask]

    return w, v, 0


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
        self.picker = kwargs.pop("picker", pick_real_eigs)
        self.guess = kwargs.pop("guess", None)
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.max_space = kwargs.pop("max_space", 12)
        self.conv_tol = kwargs.pop("conv_tol", 1e-8)
        self.tol_residual = kwargs.pop("tol_residual", 1e-6)
        self.hermitian = kwargs.pop("hermitian", True)
        self.nphys = kwargs.pop("nphys", None)

        # Base class:
        super().__init__(matvec, diagonal, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > nroots:  %s", self.nroots)
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > max_space:  %s", self.max_space)
        self.log.info(" > conv_tol:  %s", self.conv_tol)
        self.log.info(" > hermitian:  %s", self.hermitian)

        # Caching:
        self.converged = []
        self.eigvals = None
        self.eigvecs = None

    def _kernel(self):
        #if self.hermitian:
        #    convs, eigvals, eigvecs = self._kernel_hermitian()
        #else:
        #    convs, eigvals, eigvecs = self._kernel_nonhermitian()

        # Sometimes Hermitian theories may have non-Hermitian matrices,
        # i.e. perturbation theories, so always use the non-Hermitian
        # solver.
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
            tol_residual=self.tol_residual,
            nroots=self.nroots,
            verbose=0,
        )
        eigvals = np.array(eigvals)
        eigvecs = np.array(eigvecs).T

        mask = np.argsort(eigvals)
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.converged = convs

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

        mask = np.argsort(eigvals)
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.converged = convs

        return convs, eigvals, eigvecs

    def get_dyson_orbitals(self):
        if self.nphys is None:
            raise ValueError("`nphys` must be set to use `Exact.get_dyson_orbitals`")

        return super().get_dyson_orbitals()

    def get_auxiliaries(self):
        raise ValueError("Cannot determine auxiliaries using `Davidson`.")
