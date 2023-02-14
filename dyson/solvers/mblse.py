"""
Moment-conserving block Lanczos eigensolver, conserving moments of
the self-energy.
"""

import warnings

import numpy as np

from dyson import util
from dyson.solvers import BaseSolver


class RecurrenceCoefficients:
    """
    Recurrence coefficient container.
    """

    def __init__(self, shape, hermitian=True, force_orthogonality=True, dtype=np.float64):
        self.hermitian = hermitian
        self.force_orthogonality = force_orthogonality
        self.zero = np.zeros(shape, dtype=dtype)
        self.data = {}

    def __getitem__(self, key):
        i, j, n = key

        if i == 0 or j == 0:
            # Zeroth order Lanczos vectors are zero
            return self.zero
        else:
            # Return Q_{i}^† H^{n} Q_{j}
            if (not self.hermitian) or i >= j:
                return self.data[i, j, n]
            else:
                return self.data[j, i, n].T.conj()

    def __setitem__(self, key, val):
        i, j, n = key

        if n == 0 and self.force_orthogonality:
            val = np.eye(self.zero.shape[0])

        if self.hermitian and i == j:
            val = 0.5 * util.hermi_sum(val)

        # Set Q_{i}^† H^{n} Q_{j}
        if (not self.hermitian) or i >= j:
            self.data[i, j, n] = val
        else:
            self.data[j, i, n] = val.T.conj()


class MBLSE_Symm(BaseSolver):
    """
    Moment-conserving block Lanczos eingsolver, conserving the
    moments of the self-energy, for a Hermitian self-energy.

    Input
    -----
    static : numpy.ndarray
        Static part of the self-energy.
    moments : numpy.ndarray
        Moments of the self-energy.

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

    def __init__(self, static, moments, **kwargs):
        # Input:
        self.static = static
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
        super().__init__(static, moments, **kwargs)

        # Logging:
        self.log.info("Options:")
        self.log.info(" > max_cycle:  %s", self.max_cycle)
        self.log.info(" > hermitian:  %s", self.hermitian)

        # Caching:
        self._cache = {}
        self.coefficients = RecurrenceCoefficients(
            static.shape,
            hermitian=True,
            dtype=np.result_type(self.static, *self.moments),
        )
        self.on_diagonal = {}
        self.off_diagonal = {}
        self.iteration = None

    @util.cache
    def coefficient_times_off_diagonal(self, i, j, n):
        """
        Compute Q_{i}^† H^{n} Q_{j} B_{j}^†
        """

        return np.dot(
            self.coefficients[i, j, n],
            self.off_diagonal[j],
        )

    @util.cache
    def coefficient_times_on_diagonal(self, i, j, n):
        """
        Compute Q_{i}^† H^{n} Q_{j} A_{j}
        """

        return np.dot(
            self.coefficients[i, j, n],
            self.on_diagonal[j],
        )

    def orthogonalised_moment(self, n):
        """
        Compute an orthogonalised moment.
        """

        orth = util.matrix_power(self.moments[0], -0.5, hermitian=True)

        return np.linalg.multi_dot(
            (
                orth,
                self.moments[n],
                orth,
            )
        )

    def _check_moment_error(self, iteration=None):
        """
        Check the error in the moments at a given iteration.
        """

        if iteration is None:
            iteration = self.iteration

        energies, couplings = self.get_auxiliaries(iteration=iteration)

        left = couplings.copy()
        moments_recovered = []
        for n in range(2 * iteration + 2):
            moments_recovered.append(np.dot(left, couplings.T.conj()))
            left = left * energies[None]

        error_moments = util.scaled_error(
            np.array(moments_recovered),
            self.moments[: 2 * iteration + 2],
        )

        return error_moments

    def initialise_recurrence(self):
        """
        Initialise the recurrences - performs the 'zeroth' iteration.
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

        self.iteration = 0

        # Zeroth order on-diagonal block is the static self-energy
        self.on_diagonal[0] = self.static

        # Zeroth order off-diagonal block is the square-root of the
        # zeroth order moment
        self.off_diagonal[0], error_sqrt = util.matrix_power(
            self.moments[0],
            0.5,
            hermitian=True,
            return_error=True,
        )

        # Populate the other orthogonalised moments
        orth, error_inv_sqrt = util.matrix_power(
            self.moments[0],
            -0.5,
            hermitian=True,
            return_error=True,
        )
        for n in range(2 * self.max_cycle + 2):
            # FIXME orth recalculated n+1 times
            self.coefficients[1, 1, n] = self.orthogonalised_moment(n)

        # First order on-diagonal block is the orthogonalised first
        # order moment
        self.on_diagonal[1] = self.coefficients[1, 1, 1]

        # Check the error in the moments up to this iteration
        error_moments = self._check_moment_error(iteration=0)

        # Logging
        self.log.info(
            "%4d %16.3g %16.3g %16.3g %16.3g %16.3g",
            0,
            error_moments,
            np.linalg.norm(self.on_diagonal[1]),
            np.linalg.norm(self.off_diagonal[0]),
            error_sqrt,
            error_inv_sqrt,
        )

    def recurrence_iteration(self):
        """
        Perform an iteration of the recurrence for hermitian systems.
        """

        self.iteration += 1
        i = self.iteration

        if self.iteration > self.max_cycle:
            raise ValueError(
                "Cannot perform more iterations than permitted "
                "by `max_cycle` or (M-2)/2 where M is the number "
                "of inputted moments."
            )

        # Find the square of the next off-diagonal block
        off_diagonal_squared = (
            +self.coefficients[i, i, 2]
            - util.hermi_sum(self.coefficient_times_off_diagonal(i, i - 1, 1))
            - np.dot(self.coefficients[i, i, 1], self.coefficients[i, i, 1])
        )
        if self.iteration > 1:
            off_diagonal_squared += np.dot(self.off_diagonal[i - 1], self.off_diagonal[i - 1])

        # Get the next off-diagonal block
        self.off_diagonal[i], error_sqrt = util.matrix_power(
            off_diagonal_squared,
            0.5,
            hermitian=True,
            return_error=True,
        )

        # Get the inverse of the off-diagonal block
        off_diagonal_inv, error_inv_sqrt = util.matrix_power(
            off_diagonal_squared,
            -0.5,
            hermitian=True,
            return_error=True,
        )

        for n in range(2 * (self.max_cycle - self.iteration + 1)):
            residual = (
                +self.coefficients[i, i, n + 1]
                - self.coefficient_times_off_diagonal(i, i - 1, n).T.conj()
                - self.coefficient_times_on_diagonal(i, i, n).T.conj()
            )
            self.coefficients[i + 1, i, n] = np.dot(off_diagonal_inv, residual)

            residual = (
                +self.coefficients[i, i, n + 2]
                - util.hermi_sum(self.coefficient_times_off_diagonal(i, i - 1, n + 1))
                - util.hermi_sum(self.coefficient_times_on_diagonal(i, i, n + 1))
                + util.hermi_sum(
                    np.dot(
                        self.coefficients[i, i, 1],
                        self.coefficient_times_off_diagonal(i, i - 1, n),
                    )
                )
                + np.dot(
                    self.off_diagonal[i - 1],
                    self.coefficient_times_off_diagonal(i - 1, i - 1, n),
                )
                + np.dot(
                    self.coefficients[i, i, 1],
                    self.coefficient_times_on_diagonal(i, i, n),
                )
            )
            self.coefficients[i + 1, i + 1, n] = np.linalg.multi_dot(
                (
                    off_diagonal_inv,
                    residual,
                    off_diagonal_inv,
                )
            )

        # Extract the next on-diagonal block
        self.on_diagonal[i + 1] = self.coefficients[i + 1, i + 1, 1]

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

    def get_auxiliaries(self, iteration=None):
        """
        Return the compressed self-energy auxiliaries.
        """

        if iteration is None:
            iteration = self.iteration

        h_tri = util.build_block_tridiagonal(
            [self.on_diagonal[i] for i in range(iteration + 2)],
            [self.off_diagonal[i] for i in range(iteration + 1)],
        )

        energies, rotated_couplings = np.linalg.eigh(h_tri[self.nphys :, self.nphys :])
        couplings = np.dot(self.off_diagonal[0].T.conj(), rotated_couplings[: self.nphys])

        return energies, couplings

    def get_eigenfunctions(self, iteration=None):
        """
        Return the eigenfunctions.
        """

        if iteration is None:
            iteration = self.iteration

        energies, couplings = self.get_auxiliaries(iteration=iteration)
        h_aux = np.block(
            [
                [self.static, couplings],
                [couplings.T.conj(), np.diag(energies)],
            ]
        )

        eigvals, eigvecs = np.linalg.eigh(h_aux)

        return eigvals, eigvecs

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
        return self.static.shape[0]


def MBLSE(static, moments, **kwargs):
    """
    Wrapper to construct a solver based on the Hermiticity of the
    input, either by the `hermitian` keyword argument or by the
    structure of the input matrices.
    """

    if "hermitian" in kwargs:
        hermitian = kwargs.pop("hermitian")
    else:
        hermitian = all(np.allclose(m, m.T.conj()) for m in [static, *moments])

    if hermitian:
        return MBLSE_Symm(static, moments, **kwargs)
    else:
        raise NotImplementedError