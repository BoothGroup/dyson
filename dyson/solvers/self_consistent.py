"""
Self-consistent eigensolver on the downfolded matrix.
"""

import numpy as np

from dyson import util
from dyson.solvers import BaseSolver


class SelfConsistent(BaseSolver):
    """
    Self-consistent eigensolver on the downfolded matrix.

    Input
    -----
    static : numpy.ndarray
        Static part of the matrix (i.e. self-energy).
    function : callable
        Function returning the matrix (i.e. self-energy) at a given
        argument (i.e. frequency). Input arguments are `argument`.

    Parameters
    ----------
    guess : float or numpy.ndarray, optional
        Initial guess for the argument entering `function`. A single
        float uses the same guess for every index in `orbitals`,
        whilst a list allows different initial guesses per orbital.
        Default value is `0.0`.
    target : int or str, optional
        Method used to target a particular root. If input is of type
        `int`, take the eigenvalue at this index. Otherwise one of
        `{"min", "max", "mindif"}`. The first two take the minimnum
        and maximum eigenvalues, and `"mindif"` takes the eigenvalue
        closest to the guess (and then closest to the previous one at
        each subsequent iteration).
    max_cycle : int, optional
        Maximum number of iterations. Default value is `50`.
    conv_tol : float, optional
        Threshold for convergence. Default value is `1e-8`.
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
        Eigenvectors of the matrix, which are proportional to the
        Dyson orbitals.
    """

    # TODO: Can probably use a newton solver as C* Σ(w) C - w = 0

    def __init__(self, static, function, **kwargs):
        # Input:
        self.static = static
        self.function = function

        # Parameters:
        self.guess = kwargs.pop("guess", 0.0)
        self.target = kwargs.pop("target", "mindif")
        self.max_cycle = kwargs.pop("max_cycle", 50)
        self.conv_tol = kwargs.pop("conv_tol", 1e-8)
        self.hermitian = kwargs.pop("hermitian", True)

        # Base class:
        super().__init__(function, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > guess:  %s", self.guess)
        self.log.info(" > target:  %s", self.target)
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > conv_tol:  %s", self.conv_tol)
        self.log.info(" > hermitian:  %s", self.hermitian)

    def picker(self, roots):
        if isinstance(self.target, int):
            root = roots[self.target]
        else:
            if self.target == "min":
                root = np.min(roots)
            elif self.target == "max":
                root = np.max(roots)
            elif self.target == "mindif":
                root = np.min(np.abs(roots - self.guess))
            else:
                raise ValueError("`target = %s`" % self.target)

        return root

    def eig(self, matrix):
        if self.hermitian:
            return np.linalg.eigh(matrix)
        else:
            return np.linalg.eig(matrix)

    def eigvals(self, matrix):
        if self.hermitian:
            return np.linalg.eigvalsh(matrix)
        else:
            return np.linalg.eigvals(matrix)

    def _kernel(self):
        root = self.guess
        root_prev = None

        self.log.info("-" * 38)
        self.log.info("%4s %16s %16s", "Iter", "Root", "Delta")
        self.log.info("%4s %16s %16s", "-" * 4, "-" * 16, "-" * 16)

        for cycle in range(1, self.max_cycle + 1):
            matrix = self.static + self.function(root)
            roots = self.eigvals(matrix)
            root = self.picker(roots)

            if cycle > 1:
                self.log.info("%4d %16.8f %16.3g", cycle, root, abs(root - root_prev))
                if abs(root - root_prev) < self.conv_tol:
                    break
            else:
                self.log.info("%4d %16.8f", cycle, root)

            root_prev = root

        self.log.info("%4s %16s %16s", "-" * 4, "-" * 16, "-" * 16)

        converged = abs(root - root_prev) < self.conv_tol
        self.flag_convergence(converged)

        matrix = self.static + self.function(root)
        eigvals, eigvecs = self.eig(matrix)

        self.log.info(util.print_eigenvalues(eigvals))

        return eigvals, eigvecs


class DiagonalSelfConsistent(BaseSolver):
    """
    Diagonal self-consistent eigensolver on the downfolded matrix.

    Input
    -----
    function : callable
        Function returning elements of the matrix (i.e. self-energy)
        at a given argument (i.e. frequency). Input arguments are
        `argument`, `orbital1`, `orbital2`.
    derivative : callable, optional
        Function returning elements of the derivative of the matrix
        (i.e. self-energy) at a given argument (i.e. frequency), with
        the derivative being with respect to the variable of said
        argument. Input arguments are the same a `function`.
    orbitals : list, optional
        Orbital indices to solve for eigenvalues and eigenvectors at.
        Default value solves for every orbital the result of function
        spans.

    Parameters
    ----------
    method : str, optional
        Method used to minimise the solution. One of `{"newton"}.
        Default value is `"newton"`.
    guess : float or numpy.ndarray, optional
        Initial guess for the argument entering `function`. A single
        float uses the same guess for every index in `orbitals`,
        whilst a list allows different initial guesses per orbital.
        Default value is `0.0`.
    linearised : bool, optional
        Linearise the problem using the derivative. If `True` then
        `derivative` must be provided, and `diagonal=True`. Default
        value is `False`.
    diagonal : bool, optional
        Apply a diagonal approximation to the input. Default value is
        `False`.

    Returns
    -------
    eigvals : numpy.ndarray (n,)
        Eigenvalues of the matrix, representing the energies of the
        Green's function.
    eigvecs : numpy.ndarray (n, n)
        Eigenvectors of the matrix, which are proportional to the
        Dyson orbitals.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError  # TODO
