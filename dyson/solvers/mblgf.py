"""
Moment-conserving block Lanczos eigensolver, conserving moments
of the Green's function.
"""

import warnings

import numpy as np

from dyson import util
from dyson.solvers import BaseSolver

# TODO inherit things from MBLSE or vice versa?


class RecurrenceCoefficients:
    """
    Recurrence coefficients container.
    """

    def __init__(self, shape, hermitian=True, force_orthogonality=True, dtype=np.float64):
        self.hermitian = hermitian
        self.zero = np.zeros(shape, dtype=dtype)
        self.data = {}

    def __getitem__(self, key):
        i, j = key

        if i == j == 1:
            return np.eye(self.zero.shape[0])

        if i < 1 or j < 1 or i < j:
            # Zero order Lanczos vectors are zero
            return self.zero
        else:
            # Return ∑ Σ^{j-1} Q_{1} C_{i,j}
            return self.data[i, j]

    def __setitem__(self, key, val):
        i, j = key

        self.data[i, j] = val


class MBLGF_Symm(BaseSolver):
    """
    Moment-conserving block Lanczos eigensolver, conserving the
    moments of the Green's function, for a Hermitian Green's function.

    Input
    -----
    moments : numpy.ndarray
        Moments of the Green's function.

    Parameters
    ----------
    max_cycle : int, optional
        Maximum number of iterations. If `None`, perform as many as
        the inputted number of moments permits. Default value is
        `None`.

    Returns
    -------
    eigvals : numpy.ndarray
        Eigenvalues of the matrix, representing the energies of the
        Green's function.
    eigvecs : numpy.ndarray
        Eigenvectors of the matrix, which provide the Dyson orbitals
        once projected into the physical space.
    """

    def __init__(self, moments, **kwargs):
        # Input:
        self.moments = moments

        # Parameters:
        self.max_cycle = kwargs.pop("max_cycle", None)
        self.hermitian = True

        max_cycle_limit = (len(moments) - 2) // 2
        if self.max_cycle is None:
            self.max_cycle = max_cycle_limit
        if self.max_cycle > max_cycle_limit:
            raise ValueError(
                "`max_cycle` cannot be more than (M-2)/2, where "
                "M is the number of inputted moments."
            )

        # Base class:
        super().__init__(moments, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > hermitian:  %s", self.hermitian)

        # Caching:
        self._cache = {}
        self.coefficients = RecurrenceCoefficients(
            self.moments[0].shape,
            hermitian=self.hermitian,
            dtype=np.result_type(*self.moments),
        )
        self.on_diagonal = {}
        self.off_diagonal = {}
        self.orth = None
        self.iteration = None

    @util.cache
    def orthogonalised_moment(self, n):
        """
        Compute an orthogonalised moment.
        """

        orth = self.orth
        if orth is None:
            orth = util.matrix_power(self.moments[0], -0.5, hermitian=self.hermitian)

        return np.linalg.multi_dot(
            (
                orth,
                self.moments[n],
                orth,
            )
        )

    def initialise_iteration_table(self):
        """
        Print the header for the table summarising the iterations.
        """

        self.log.info("-" * 89)
        self.log.info(
            "{:^4s} {:^16s} {:^33s} {:^33}".format(
                "",
                "",
                "Norm of matrix",
                "Norm of removed space",
            )
        )
        self.log.info(
            "{:^4s} {:^16s} {:^33s} {:^33}".format(
                "Iter",
                "Moment error",
                "-" * 33,
                "-" * 33,
            )
        )
        self.log.info(
            "%4s %16s %16s %16s %16s %16s",
            "",
            "",
            "On-diagonal",
            "Off-diagonal",
            "Square root",
            "Inv. square root",
        )
        self.log.info(
            "%4s %16s %16s %16s %16s %16s",
            "-" * 4,
            "-" * 16,
            "-" * 16,
            "-" * 16,
            "-" * 16,
            "-" * 16,
        )

    def _check_moment_error(self, iteration=None):
        """
        Check the error in the moments at a given iteration.
        """

        if iteration is None:
            iteration = self.iteration

        energies, dyson_orbitals = self.get_dyson_orbitals(iteration=iteration)

        left = dyson_orbitals.copy()
        moments_recovered = []
        for n in range(2 * iteration + 2):
            moments_recovered.append(np.dot(left, dyson_orbitals.T.conj()))
            left = left * energies[None]

        error_moments = sum(
            util.scaled_error(a, b)
            for a, b in zip(moments_recovered, self.moments[: 2 * iteration + 2])
        )

        return error_moments

    def initialise_recurrence(self):
        """
        Initialise the recurrences - performs the 'zeroth' iteration.

        This iteration is essentially equivalent to solving a generalised
        eigenvalue problem on the Fock matrix in the physical space.
        """

        # Initialise the table
        self.initialise_iteration_table()

        self.iteration = 0

        # Calculate the orthogonalisation matrix
        self.orth, error_inv_sqrt = util.matrix_power(
            self.moments[0],
            -0.5,
            hermitian=self.hermitian,
            return_error=True,
        )

        # Add zero matrix to out-of-bounds off-diagonal to simplify logic
        self.off_diagonal[-1] = self.coefficients.zero

        # Zeroth order on-diagonal block is the orthogonalised first
        # moment (equal to the orthogonalised static part of the
        # matrix corresponding to the solution moments)
        self.on_diagonal[0] = self.orthogonalised_moment(1)

        # Check the error in the moments up to this iteration
        error_moments = self._check_moment_error()

        # Logging
        self.log.info(
            "%4d %16.3g %16.3g %16s %16s %16.3g",
            0,
            error_moments,
            np.linalg.norm(self.on_diagonal[0]),
            "",
            "",
            error_inv_sqrt,
        )

    def recurrence_iteration(self):
        """
        Perform an iteration of the recurrence.
        """

        self.iteration += 1
        i = self.iteration - 1

        if self.iteration > self.max_cycle:
            raise ValueError(
                "Cannot perform more iterations than permitted "
                "by `max_cycle` or (M-2)/2 where M is the number "
                "of inputted moments."
            )

        # Find the square of the next off-diagonal block
        off_diagonal_squared = self.coefficients.zero.copy()
        for j in range(i + 2):
            for k in range(i + 1):
                off_diagonal_squared += np.linalg.multi_dot(
                    (
                        self.coefficients[i + 1, k + 1].T.conj(),
                        self.orthogonalised_moment(j + k + 1),
                        self.coefficients[i + 1, j],
                    )
                )

        off_diagonal_squared -= np.dot(
            self.on_diagonal[i],
            self.on_diagonal[i],
        )
        if i:
            off_diagonal_squared -= np.dot(
                self.off_diagonal[i - 1],
                self.off_diagonal[i - 1],
            )

        # Get the next off-diagonal block
        self.off_diagonal[i], error_sqrt = util.matrix_power(
            off_diagonal_squared,
            0.5,
            hermitian=self.hermitian,
            return_error=True,
        )

        # Get the inverse of the off-diagonal block
        off_diagonal_inv, error_inv_sqrt = util.matrix_power(
            off_diagonal_squared,
            -0.5,
            hermitian=self.hermitian,
            return_error=True,
        )

        for j in range(i + 2):
            residual = (
                +self.coefficients[i + 1, j]
                - np.dot(self.coefficients[i + 1, j + 1], self.on_diagonal[i])
                - np.dot(self.coefficients[i, j + 1], self.off_diagonal[i - 1])
            )
            self.coefficients[i + 2, j + 1] = np.dot(residual, off_diagonal_inv)

        self.on_diagonal[i + 1] = self.coefficients.zero.copy()
        for j in range(i + 2):
            for k in range(i + 2):
                self.on_diagonal[i + 1] += np.linalg.multi_dot(
                    (
                        self.coefficients[i + 2, k + 1].T.conj(),
                        self.orthogonalised_moment(j + k + 1),
                        self.coefficients[i + 2, j + 1],
                    )
                )

        # Check the error in the moments up to this iteration
        error_moments = self._check_moment_error()

        # Logging
        self.log.info(
            "%4d %16.3g %16.3g %16.3g %16.3g %16.3g",
            self.iteration,
            error_moments,
            np.linalg.norm(self.on_diagonal[i + 1]),
            np.linalg.norm(self.off_diagonal[i]),
            error_sqrt,
            error_inv_sqrt,
        )
        if self.iteration == self.max_cycle:
            self.log.info("-" * 89)

    def get_eigenfunctions(self, iteration=None):
        """
        Return the eigenfunctions.
        """

        if iteration is None:
            iteration = self.iteration

        h_tri = util.build_block_tridiagonal(
            [self.on_diagonal[i] for i in range(iteration + 1)],
            [self.off_diagonal[i] for i in range(iteration)],
        )

        eigvals, eigvecs = np.linalg.eigh(h_tri)
        eigvecs[: self.nphys] = np.dot(self.orth, eigvecs[: self.nphys])

        return eigvals, eigvecs

    def get_dyson_orbitals(self, iteration=None):
        """
        Return the Dyson orbitals and their energies.
        """

        eigvals, eigvecs = self.get_eigenfunctions(iteration=iteration)

        return eigvals, eigvecs[: self.nphys]

    def get_auxiliaries(self, iteration=None):
        """
        Return the self-energy auxiliaries.
        """

        if iteration is None:
            iteration = self.iteration

        h_tri = util.build_block_tridiagonal(
            [self.on_diagonal[i] for i in range(iteration + 1)],
            [self.off_diagonal[i] for i in range(iteration)],
        )

        energies, rotated_couplings = np.linalg.eigh(h_tri[self.nphys :, self.nphys :])
        couplings = np.dot(self.off_diagonal[0].T.conj(), rotated_couplings[: self.nphys])

        return energies, couplings

    def _kernel(self, iteration=None):
        if self.iteration is None:
            self.initialise_recurrence()
        if iteration is None:
            iteration = self.max_cycle
        while self.iteration < iteration:
            self.recurrence_iteration()

        self.log.info("Block Lanczos moment recurrence completed to iteration %d.", self.iteration)

        if iteration is None:
            iteration = self.max_cycle

        eigvals, eigvecs = self.get_eigenfunctions(iteration=iteration)

        self.log.info(util.print_dyson_orbitals(eigvals, eigvecs, self.nphys))

        return eigvals, eigvecs

    @property
    def nphys(self):
        return self.moments[0].shape[0]


class MBLGF_NoSymm(MBLGF_Symm):
    """
    Moment-conserving block Lanczos eigensolver, conserving the
    moments of the Green's function, for a non-Hermitian Green's
    function.

    Input
    -----
    moments : numpy.ndarray
        Moments of the Green's function.

    Parameters
    ----------
    max_cycle : int, optional
        Maximum number of iterations. If `None`, perform as many as
        the inputted number of moments permits. Default value is
        `None`.

    Returns
    -------
    eigvals : numpy.ndarray
        Eigenvalues of the matrix, representing the energies of the
        Green's function.
    eigvecs : tuple of numpy.ndarray
        Left- and right-hand eigenvectors of the matrix, which provide
        the Dyson orbitals once projected into the physical space.
    """

    def __init__(self, moments, **kwargs):
        # Input:
        self.moments = moments

        # Parameters:
        self.max_cycle = kwargs.pop("max_cycle", None)
        self.hermitian = False

        max_cycle_limit = (len(moments) - 2) // 2
        if self.max_cycle is None:
            self.max_cycle = max_cycle_limit
        if self.max_cycle > max_cycle_limit:
            raise ValueError(
                "`max_cycle` cannot be more than (M-2)/2, where "
                "M is the number of inputted moments."
            )

        # Base class:
        BaseSolver.__init__(self, moments, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > hermitian:  %s", self.hermitian)

        # Caching:
        self._cache = {}
        self.coefficients = [
            RecurrenceCoefficients(
                self.moments[0].shape,
                hermitian=self.hermitian,
                dtype=np.result_type(*self.moments),
            ),
            RecurrenceCoefficients(
                self.moments[0].shape,
                hermitian=self.hermitian,
                dtype=np.result_type(*self.moments),
            ),
        ]
        self.on_diagonal = {}
        self.off_diagonal = [{}, {}]
        self.orth = None
        self.iteration = None

    def initialise_iteration_table(self):
        """
        Print the header for the table sumarising the iterations.
        """

        self.log.info("-" * 106)
        self.log.info(
            "{:^4s} {:^16s} {:^50s} {:^33}".format(
                "",
                "",
                "Norm of matrix",
                "Norm of removed space",
            )
        )
        self.log.info(
            "{:^4s} {:^16s} {:^50s} {:^33}".format(
                "Iter",
                "Moment error",
                "-" * 50,
                "-" * 33,
            )
        )
        self.log.info(
            "%4s %16s %16s %16s %16s %16s %16s",
            "",
            "",
            "On-diagonal",
            "Off-diagonal ↑",
            "Off-diagonal ↓",
            "Square root",
            "Inv. square root",
        )
        self.log.info(
            "%4s %16s %16s %16s %16s %16s %16s",
            "-" * 4,
            "-" * 16,
            "-" * 16,
            "-" * 16,
            "-" * 16,
            "-" * 16,
            "-" * 16,
        )

    def _check_moment_error(self, iteration=None):
        """
        Check the error in the moments at a given iteration.
        """

        if iteration is None:
            iteration = self.iteration

        energies, (eigvecs_l, eigvecs_r) = self.get_dyson_orbitals(iteration=iteration)

        left = eigvecs_l[: self.nphys]
        right = eigvecs_r[: self.nphys]
        moments_recovered = []
        for n in range(2 * iteration + 2):
            moments_recovered.append(np.dot(left, right.T.conj()))
            left = left * energies[None]

        error_moments = util.scaled_error(
            np.array(moments_recovered),
            self.moments[: 2 * iteration + 2],
        )

        return error_moments

    def initialise_recurrence(self):
        """
        Initialise the recurrences - performs the 'zeroth' iteration.

        This iteration is essentially equivalent to solving a generalised
        eigenvalue problem on the Fock matrix in the physical space.
        """

        # Initialise the table
        self.initialise_iteration_table()

        self.iteration = 0

        # Calculate the orthogonalisation matrix
        self.orth, error_inv_sqrt = util.matrix_power(
            self.moments[0],
            -0.5,
            hermitian=self.hermitian,
            return_error=True,
        )

        # Add zero matrix to out-of-bounds off-diagonal to simplify logic
        self.off_diagonal[0][-1] = self.coefficients[0].zero
        self.off_diagonal[1][-1] = self.coefficients[1].zero

        # Zeroth order on-diagonal block is the orthogonalised first
        # moment (equal to the orthogonalised static part of the
        # matrix corresponding to the solution moments)
        self.on_diagonal[0] = self.orthogonalised_moment(1)

        # Check the error in the moments up to this iteration
        error_moments = self._check_moment_error()

        # Logging
        self.log.info(
            "%4d %16.3g %16.3g %16s %16s %16s %16.3g",
            0,
            error_moments,
            np.linalg.norm(self.on_diagonal[0]),
            "",
            "",
            "",
            error_inv_sqrt,
        )

    def recurrence_iteration(self):
        """
        Perform an iteration of the recurrence.
        """

        self.iteration += 1
        i = self.iteration - 1

        if self.iteration > self.max_cycle:
            raise ValueError(
                "Cannot perform more iterations than permitted "
                "by `max_cycle` or (M-2)/2 where M is the number "
                "of inputted moments."
            )

        # Find the square of the next off-diagonal blocks
        off_diagonal_squared = [
            self.coefficients[0].zero.astype(complex).copy(),
            self.coefficients[1].zero.astype(complex).copy(),
        ]
        for j in range(i + 2):
            for k in range(i + 1):
                off_diagonal_squared[0] += np.linalg.multi_dot(
                    (
                        self.coefficients[1][i + 1, k + 1],
                        self.orthogonalised_moment(j + k + 1),
                        self.coefficients[0][i + 1, j],
                    )
                )
                off_diagonal_squared[1] += np.linalg.multi_dot(
                    (
                        self.coefficients[1][i + 1, j],
                        self.orthogonalised_moment(j + k + 1),
                        self.coefficients[0][i + 1, k + 1],
                    )
                )

        off_diagonal_squared[0] -= np.dot(
            self.on_diagonal[i],
            self.on_diagonal[i],
        )
        off_diagonal_squared[1] -= np.dot(
            self.on_diagonal[i],
            self.on_diagonal[i],
        )
        if i:
            off_diagonal_squared[0] -= np.dot(
                self.off_diagonal[1][i - 1],
                self.off_diagonal[1][i - 1],
            )
            off_diagonal_squared[1] -= np.dot(
                self.off_diagonal[0][i - 1],
                self.off_diagonal[0][i - 1],
            )

        # Get the next off-diagonal blocks
        self.off_diagonal[0][i], error_sqrt_upper = util.matrix_power(
            off_diagonal_squared[0],
            0.5,
            hermitian=self.hermitian,
            return_error=True,
        )
        self.off_diagonal[1][i], error_sqrt_lower = util.matrix_power(
            off_diagonal_squared[1],
            0.5,
            hermitian=self.hermitian,
            return_error=True,
        )
        error_sqrt = np.sqrt(error_sqrt_upper**2 + error_sqrt_lower**2)

        # Get the inverse of the off-diagonal blocks
        off_diagonal_inv_upper, error_inv_sqrt_upper = util.matrix_power(
            off_diagonal_squared[0],
            -0.5,
            hermitian=self.hermitian,
            return_error=True,
        )
        off_diagonal_inv_lower, error_inv_sqrt_lower = util.matrix_power(
            off_diagonal_squared[1],
            -0.5,
            hermitian=self.hermitian,
            return_error=True,
        )
        error_inv_sqrt = np.sqrt(error_inv_sqrt_upper**2 + error_inv_sqrt_lower**2)

        for j in range(i + 2):
            residual = (
                +self.coefficients[0][i + 1, j]
                - np.dot(self.coefficients[0][i + 1, j + 1], self.on_diagonal[i])
                - np.dot(self.coefficients[0][i, j + 1], self.off_diagonal[0][i - 1])
            )
            self.coefficients[0][i + 2, j + 1] = np.dot(residual, off_diagonal_inv_lower)

            residual = (
                +self.coefficients[1][i + 1, j]
                - np.dot(self.on_diagonal[i], self.coefficients[1][i + 1, j + 1])
                - np.dot(self.off_diagonal[1][i - 1], self.coefficients[1][i, j + 1])
            )
            self.coefficients[1][i + 2, j + 1] = np.dot(off_diagonal_inv_upper, residual)

        self.on_diagonal[i + 1] = self.coefficients[0].zero.astype(complex).copy()
        for j in range(i + 2):
            for k in range(i + 2):
                self.on_diagonal[i + 1] += np.linalg.multi_dot(
                    (
                        self.coefficients[1][i + 2, k + 1],
                        self.orthogonalised_moment(j + k + 1),
                        self.coefficients[0][i + 2, j + 1],
                    )
                )

        # Check the error in the moments up to this iteration
        error_moments = self._check_moment_error()

        # Logging
        self.log.info(
            "%4d %16.3g %16.3g %16.3g %16.3g %16.3g %16.3g",
            self.iteration,
            error_moments,
            np.linalg.norm(self.on_diagonal[i + 1]),
            np.linalg.norm(self.off_diagonal[0][i]),
            np.linalg.norm(self.off_diagonal[1][i]),
            error_sqrt,
            error_inv_sqrt,
        )
        if self.iteration == self.max_cycle:
            self.log.info("-" * 106)

    def get_eigenfunctions(self, iteration=None):
        """
        Return the eigenfunctions.
        """

        if iteration is None:
            iteration = self.iteration

        h_tri = util.build_block_tridiagonal(
            [self.on_diagonal[i] for i in range(iteration + 1)],
            [self.off_diagonal[0][i] for i in range(iteration)],
            [self.off_diagonal[1][i] for i in range(iteration)],
        )

        eigvals, eigvecs = np.linalg.eig(h_tri)
        mask = np.argsort(eigvals.real)
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]

        eigvecs_l = eigvecs
        eigvecs_r = np.linalg.inv(eigvecs).T.conj()

        eigvecs_l[: self.nphys] = np.dot(self.orth, eigvecs_l[: self.nphys])
        eigvecs_r[: self.nphys] = np.dot(self.orth, eigvecs_r[: self.nphys])
        eigvecs = (eigvecs_l, eigvecs_r)

        return eigvals, eigvecs

    def get_dyson_orbitals(self, iteration=None):
        """
        Return the Dyson orbitals and their energies.
        """

        eigvals, eigvecs = self.get_eigenfunctions(iteration=iteration)

        return eigvals, (eigvecs[0][: self.nphys], eigvecs[1][: self.nphys])

    def get_auxiliaries(self, iteration=None):
        raise NotImplementedError  # TODO


def MBLGF(moments, **kwargs):
    """
    Wrapper to construct a solver based on the Hermiticity of the
    input, either by the `hermitian` keyword argument or by the
    structure of the input matrices.
    """

    if "hermitian" in kwargs:
        hermitian = kwargs.pop("hermitian")
    else:
        hermitian = all(np.allclose(m, m.T.conj()) for m in moments)

    if hermitian:
        return MBLGF_Symm(moments, **kwargs)
    else:
        return MBLGF_NoSymm(moments, **kwargs)
