"""Moment block Lanczos for moments of the Green's function."""

from __future__ import annotations

from abc import abstractmethod
import functools
from typing import TYPE_CHECKING

from dyson import numpy as np, util
from dyson.solvers.static._mbl import BaseRecursionCoefficients, BaseMBL

if TYPE_CHECKING:
    from typing import Any, TypeAlias

    from dyson.typing import Array
    from dyson.lehmann import Lehmann

    Couplings: TypeAlias = Array | tuple[Array, Array]

# TODO: Use solvers for diagonalisation?


class RecursionCoefficients(BaseRecursionCoefficients):
    """Recursion coefficients for the moment block Lanczos algorithm for the Green's function.

    Args:
        nphys: Number of physical degrees of freedom.
    """

    def __getitem__(self, key: tuple[int, ...]) -> Array:
        """Get the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.

        Returns:
            The recursion coefficients.
        """
        i, j = key
        if i == j == 1:
            return np.eye(self.nphys, dtype=self.dtype)
        if i < 1 or j < 1 or i < j:
            return self._zero
        return self._data[i, j]

    def __setitem__(self, key: tuple[int, ...], value: Array) -> None:
        """Set the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.
            value: The recursion coefficients.
        """
        self._data[key] = value


def _infer_max_cycle(moments: Array) -> int:
    """Infer the maximum number of cycles from the moments."""
    return (moments.shape[0] - 2) // 2


class MBLGF(BaseMBL):
    """Moment block Lanczos for moments of the Green's function.

    Args:
        moments: Moments of the Green's function.
    """

    Coefficients = RecursionCoefficients

    def __init__(
        self,
        moments: Array,
        max_cycle: int | None = None,
        hermitian: bool = True,
        force_orthogonality: bool = True,
        calculate_errors: bool = True,
    ) -> None:
        """Initialise the solver.

        Args:
            moments: Moments of the Green's function.
            max_cycle: Maximum number of cycles.
            hermitian: Whether the Green's function is hermitian.
            force_orthogonality: Whether to force orthogonality of the recursion coefficients.
            calculate_errors: Whether to calculate errors.
        """
        self._moments = moments
        self.max_cycle = max_cycle if max_cycle is not None else _infer_max_cycle(moments)
        self.hermitian = hermitian
        self.force_orthogonality = force_orthogonality
        self.calculate_errors = calculate_errors
        if self.hermitian:
            self._coefficients = (
                self.Coefficients(
                    self.nphys,
                    hermitian=self.hermitian,
                    dtype=moments.dtype.name,
                    force_orthogonality=self.force_orthogonality,
                ),
            ) * 2
        else:
            self._coefficients = (
                self.Coefficients(
                    self.nphys,
                    hermitian=self.hermitian,
                    dtype=moments.dtype.name,
                    force_orthogonality=self.force_orthogonality,
                ),
                self.Coefficients(
                    self.nphys,
                    hermitian=self.hermitian,
                    dtype=moments.dtype.name,
                    force_orthogonality=self.force_orthogonality,
                ),
            )
        self._on_diagonal: dict[int, Array] = {}
        self._off_diagonal_upper: dict[int, Array] = {}
        self._off_diagonal_lower: dict[int, Array] = {}

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> MBLGF:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        max_cycle = kwargs.get("max_cycle", 0)
        energies, couplings = self_energy.diagonalise_matrix_with_projection(static)
        greens_function = self_energy.__class__(energies, couplings, chempot=self_energy.chempot)
        moments = greens_function.moments(range(2 * max_cycle + 2))
        return cls(moments, hermitian=greens_function.hermitian, **kwargs)

    def reconstruct_moments(self, iteration: int) -> Array:
        """Reconstruct the moments.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed moments.
        """
        greens_function = self.get_greens_function(iteration=iteration)
        energies = greens_function.energies
        left, right = greens_function.unpack_couplings()

        # Construct the recovered moments
        left_factored = left.copy()
        moments: list[Array] = []
        for order in range(2 * iteration + 2):
            moments.append(left_factored @ right.T.conj())
            left_factored = left_factored * energies[None]

        return np.array(moments)

    def initialise_recurrence(self) -> tuple[float | None, float | None, float | None]:
        """Initialise the recurrence (zeroth iteration).

        Returns:
            If :attr:`calculate_errors`, the error metrics in the square root of the off-diagonal
            block, the inverse square root of the off-diagonal block, and the error in the
            recovered moments. If not, all three are `None`.
        """
        # Get the inverse square-root error
        error_inv_sqrt: float | None = None
        if self.calculate_errors:
            _, error_inv_sqrt = util.matrix_power(
                self.moments[0], -0.5, hermitian=self.hermitian, return_error=True
            )

        # Initialise the blocks
        self.off_diagonal_upper[-1] = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        self.off_diagonal_lower[-1] = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        self.on_diagonal[0] = self.orthogonalised_moment(1)
        error_sqrt = 0.0

        # Get the error in the moments
        error_moments: float | None = None
        if self.calculate_errors:
            error_moments = self.moment_error(iteration=0)

        return error_sqrt, error_inv_sqrt, error_moments

    def _recurrence_iteration_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a Hermitian Green's function."""
        i = iteration + 1
        coefficients = self.coefficients[0]
        on_diagonal = self.on_diagonal
        off_diagonal = self.off_diagonal_upper

        # Find the squre of the off-diagonal block
        off_diagonal_squared = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        for j in range(i + 2):
            for k in range(i + 1):
                off_diagonal_squared += (
                    coefficients[i + 1, k + 1].T.conj()
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[i + 1, j]
                )
        off_diagonal_squared -= on_diagonal[i] @ on_diagonal[i]
        if i:
            off_diagonal_squared -= off_diagonal[i - 1] @ off_diagonal[i - 1]

        # Get the off-diagonal block
        off_diagonal[i], error_sqrt = util.matrix_power(
            off_diagonal_squared, 0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )
        self.off_diagonal_lower[i] = off_diagonal[i].T.conj()

        # Invert the off-diagonal block
        off_diagonal_inv, error_inv_sqrt = util.matrix_power(
            off_diagonal_squared, -0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )

        for j in range(i + 2):
            # Horizontal recursion
            residual = coefficients[i + 1, j].copy()
            residual -= coefficients[i + 1, j + 1], on_diagonal[i]
            residual -= coefficients[i, j + 1] @ off_diagonal[i - 1]
            coefficients[i + 2, j + 1] = residual @ off_diagonal_inv

        # Calculate the on-diagonal block
        on_diagonal[i + 1] = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        for j in range(i + 2):
            for k in range(i + 2):
                on_diagonal[i + 1] = (
                    coefficients[i + 2, k + 1].T.conj()
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[i + 2, j + 1]
                )

        # Get the error in the moments
        error_moments: float | None = None
        if self.calculate_errors:
            error_moments = self.moment_error(iteration=iteration)

        return error_sqrt, error_inv_sqrt, error_moments

    def _recurrence_iteration_non_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a non-Hermitian Green's function."""
        i = iteration + 1
        coefficients = self.coefficients
        on_diagonal = self.on_diagonal
        off_diagonal_upper = self.off_diagonal_upper
        off_diagonal_lower = self.off_diagonal_lower

        # Find the square of the off-diagonal blocks
        off_diagonal_upper_squared = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        off_diagonal_lower_squared = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        for j in range(i + 2):
            for k in range(i + 1):
                off_diagonal_upper_squared += (
                    coefficients[0][i + 1, k + 1]
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[1][i + 1, j]
                )
                off_diagonal_lower_squared += (
                    coefficients[1][i + 1, j]
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[0][i + 1, k + 1]
                )
        off_diagonal_upper_squared -= on_diagonal[i] @ on_diagonal[i]
        off_diagonal_lower_squared -= on_diagonal[i] @ on_diagonal[i]
        if i:
            off_diagonal_upper_squared -= off_diagonal_lower[i - 1] @ off_diagonal_lower[i - 1]
            off_diagonal_lower_squared -= off_diagonal_upper[i - 1] @ off_diagonal_upper[i - 1]

        # Get the off-diagonal blocks
        off_diagonal_upper[i], error_sqrt_upper = util.matrix_power(
            off_diagonal_upper_squared,
            0.5,
            hermitian=self.hermitian,
            return_error=self.calculate_errors,
        )
        off_diagonal_lower[i], error_sqrt_lower = util.matrix_power(
            off_diagonal_lower_squared,
            0.5,
            hermitian=self.hermitian,
            return_error=self.calculate_errors,
        )
        error_sqrt: float | None = None
        if self.calculate_errors:
            assert error_sqrt_upper is not None and error_sqrt_lower is not None
            error_sqrt = np.sqrt(error_sqrt_upper**2 + error_sqrt_lower**2)

        # Invert the off-diagonal blocks
        off_diagonal_upper_inv, error_inv_sqrt_upper = util.matrix_power(
            off_diagonal_upper_squared,
            -0.5,
            hermitian=self.hermitian,
            return_error=self.calculate_errors,
        )
        off_diagonal_lower_inv, error_inv_sqrt_lower = util.matrix_power(
            off_diagonal_lower_squared,
            -0.5,
            hermitian=self.hermitian,
            return_error=self.calculate_errors,
        )
        error_inv_sqrt: float | None = None
        if self.calculate_errors:
            assert error_inv_sqrt_upper is not None and error_inv_sqrt_lower is not None
            error_inv_sqrt = np.sqrt(error_inv_sqrt_upper**2 + error_inv_sqrt_lower**2)

        for j in range(i + 2):
            # Horizontal recursion
            residual = coefficients[0][i + 1, j].copy()
            residual -= coefficients[0][i + 1, j + 1] @ on_diagonal[i]
            residual -= coefficients[0][i, j + 1] @ off_diagonal_upper[i - 1]
            coefficients[0][i + 2, j + 1] = residual @ off_diagonal_lower_inv

            # Vertical recursion
            residual = coefficients[1][i + 1, j].copy()
            residual -= on_diagonal[i] @ coefficients[1][i + 1, j + 1]
            residual -= off_diagonal_lower[i - 1] @ coefficients[1][i, j + 1]
            coefficients[1][i + 2, j + 1] = residual @ off_diagonal_upper_inv

        # Calculate the on-diagonal block
        on_diagonal[i + 1] = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        for j in range(i + 2):
            for k in range(i + 2):
                on_diagonal[i + 1] = (
                    coefficients[1][i + 2, k + 1]
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[0][i + 2, j + 1]
                )

        # Get the error in the moments
        error_moments: float | None = None
        if self.calculate_errors:
            error_moments = self.moment_error(iteration=iteration)

        return error_sqrt, error_inv_sqrt, error_moments

    def get_auxiliaries(
        self, iteration: int | None = None, **kwargs: Any
    ) -> tuple[Array, Couplings]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Args:
            iteration: The iteration to get the auxiliary energies and couplings for.

        Returns:
            Auxiliary energies and couplings.
        """
        if iteration is None:
            iteration = self.max_cycle
        if kwargs:
            raise TypeError(
                f"get_auxiliaries() got unexpected keyword argument {next(iter(kwargs))}"
            )

        # Get the block tridiagonal Hamiltonian
        hamiltonian = util.build_block_tridiagonal(
            [self.on_diagonal[i] for i in range(iteration + 2)],
            [self.off_diagonal_upper[i] for i in range(iteration + 1)],
            [self.off_diagonal_lower[i] for i in range(iteration + 1)],
        )

        # Return early if there are no auxiliaries
        couplings: Couplings
        if hamiltonian.shape == (self.nphys, self.nphys):
            energies = np.zeros((0,), dtype=hamiltonian.dtype)
            couplings = np.zeros((self.nphys, 0), dtype=hamiltonian.dtype)
            return energies, couplings

        # Diagonalise the subspace to get the energies and basis for the couplings
        subspace = hamiltonian[self.nphys :, self.nphys :]
        energies, rotated = util.eig(subspace, hermitian=self.hermitian)

        # Project back to the couplings
        if self.hermitian:
            couplings = self.off_diagonal_upper[0].T.conj() @ rotated[: self.nphys]
        else:
            couplings = (
                self.off_diagonal_upper[0].T.conj() @ rotated[: self.nphys],
                self.off_diagonal_lower[0].T.conj() @ np.linalg.inv(rotated).T.conj()[: self.nphys],
            )

        return energies, couplings

    def get_eigenfunctions(
        self, unpack: bool = False, iteration: int | None = None, **kwargs: Any
    ) -> tuple[Array, Couplings]:
        """Get the eigenfunction at a given iteration.

        Args:
            unpack: Whether to unpack the eigenvectors into left and right components, regardless
                of the hermitian property.
            iteration: The iteration to get the eigenfunction for.

        Returns:
            The eigenfunction.
        """
        if iteration is None:
            iteration = self.max_cycle
        if kwargs:
            raise TypeError(
                f"get_auxiliaries() got unexpected keyword argument {next(iter(kwargs))}"
            )

        # Get the eigenvalues and eigenvectors
        eigvecs: Couplings
        if iteration == self.max_cycle and self.eigvals is not None and self.eigvecs is not None:
            eigvals = self.eigvals
            eigvecs = self.eigvecs
        else:
            # Diagonalise the block tridiagonal Hamiltonian
            hamiltonian = util.build_block_tridiagonal(
                [self.on_diagonal[i] for i in range(iteration + 2)],
                [self.off_diagonal_upper[i] for i in range(iteration + 1)],
                [self.off_diagonal_lower[i] for i in range(iteration + 1)],
            )
            eigvals, eigvecs = util.eig(hamiltonian, hermitian=self.hermitian)

            # Unorthogonalise the eigenvectors
            metric_inv = self.orthogonalisation_metric_inv
            if self.hermitian:
                eigvecs[: self.nphys] = metric_inv @ eigvecs[: self.nphys]  # type: ignore[index]
            else:
                left = eigvecs
                right = np.linalg.inv(eigvecs).T.conj()
                left[: self.nphys] = metric_inv @ left[: self.nphys]  # type: ignore[index]
                right[: self.nphys] = metric_inv.T.conj() @ right[: self.nphys]
                eigvecs = (left, right)  # type: ignore[assignment]

        if unpack:
            # Unpack the eigenvectors
            if self.hermitian:
                if isinstance(eigvecs, tuple):
                    raise ValueError("Hermitian solver should not get a tuple of eigenvectors.")
                return eigvals, (eigvecs, eigvecs)
            elif isinstance(eigvecs, tuple):
                return eigvals, eigvecs
            else:
                return eigvals, (eigvecs, np.linalg.inv(eigvecs).T.conj())

        return eigvals, eigvecs

    @property
    def coefficients(self) -> tuple[BaseRecursionCoefficients, BaseRecursionCoefficients]:
        """Get the recursion coefficients."""
        return self._coefficients

    @property
    def on_diagonal(self) -> dict[int, Array]:
        """Get the on-diagonal blocks of the self-energy."""
        return self._on_diagonal

    @property
    def off_diagonal_upper(self) -> dict[int, Array]:
        """Get the upper off-diagonal blocks of the self-energy."""
        return self._off_diagonal_upper

    @property
    def off_diagonal_lower(self) -> dict[int, Array]:
        """Get the lower off-diagonal blocks of the self-energy."""
        return self._off_diagonal_lower
