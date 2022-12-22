"""
Self-consistent eigensolver on the downfolded matrix.
"""

import numpy as np

from dyson.solvers import BaseSolver


class SelfConsistent(BaseSolver):
    """
    Self-consistent eigensolver on the downfolded matrix.

    Input
    -----
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

    def __init__(self, function, **kwargs):
        # Input:
        self.function = function

        # Parameters:
        self.guess = kwargs.pop("guess", 0.0)
        self.target = kwargs.pop("target", "mindif")

        # Base class:
        super().__init__(*args, **kwargs)

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

        for cycle in range(1, self.max_cycle+1):
            matrix = self.function(root)
            roots = self.eigvals(matrix)

            root_prev = root
            root = self.picker(roots)

            if abs(root - root_prev) < self.conv_tol:
                break

        converged = abs(root - root_prev) < conv_tol
        self.flag_convergence(converged)

        matrix = self.function(root)
        eigvals, eigvecs = self.eig(matrix)

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

    pass
