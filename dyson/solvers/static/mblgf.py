"""Moment block Lanczos for moments of the Green's function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import console, printing, util
from dyson import numpy as np
from dyson.solvers.static._mbl import BaseMBL, BaseRecursionCoefficients
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from typing import Any

    from dyson.expressions.expression import BaseExpression
    from dyson.lehmann import Lehmann
    from dyson.typing import Array


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

    def __init__(  # noqa: D417
        self,
        moments: Array,
        **kwargs: Any,
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
        self.max_cycle = kwargs["max_cycle"] if "max_cycle" in kwargs else _infer_max_cycle(moments)
        self.set_options(**kwargs)

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

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.moments.ndim != 3 or self.moments.shape[1] != self.moments.shape[2]:
            raise ValueError(
                "moments must be a 3D array with the second and third dimensions equal."
            )
        if _infer_max_cycle(self.moments) < self.max_cycle:
            raise ValueError("not enough moments provided for the specified max_cycle.")

        # Print the input information
        cond = printing.format_float(
            np.linalg.cond(self.moments[0]), threshold=1e10, scientific=True, precision=4
        )
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        console.print(f"Number of moments: [input]{self.moments.shape[0]}[/input]")
        console.print(f"Overlap condition number: {cond}")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> MBLGF:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        max_cycle = kwargs.get("max_cycle", 0)
        energies, couplings = self_energy.diagonalise_matrix_with_projection(
            static, overlap=overlap
        )
        greens_function = self_energy.__class__(energies, couplings, chempot=self_energy.chempot)
        moments = greens_function.moments(range(2 * max_cycle + 2))
        return cls(moments, hermitian=greens_function.hermitian, **kwargs)

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> MBLGF:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        moments = expression.build_gf_moments(2 * kwargs.get("max_cycle", 0) + 2)
        return cls(moments, hermitian=expression.hermitian, **kwargs)

    def reconstruct_moments(self, iteration: int) -> Array:
        """Reconstruct the moments.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed moments.
        """
        greens_function = self.solve(iteration=iteration).get_greens_function()
        return greens_function.moments(range(2 * iteration + 2))

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
        i = iteration - 1
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

        # Invert the off-diagonal block
        off_diagonal_inv, error_inv_sqrt = util.matrix_power(
            off_diagonal_squared, -0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )

        for j in range(i + 2):
            # Horizontal recursion
            residual = coefficients[i + 1, j].copy()
            residual -= coefficients[i + 1, j + 1] @ on_diagonal[i]
            residual -= coefficients[i, j + 1] @ off_diagonal[i - 1]
            coefficients[i + 2, j + 1] = residual @ off_diagonal_inv

        # Calculate the on-diagonal block
        on_diagonal[i + 1] = np.zeros((self.nphys, self.nphys), dtype=self.moments.dtype)
        for j in range(i + 2):
            for k in range(i + 2):
                on_diagonal[i + 1] += (
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
        i = iteration - 1
        coefficients = self.coefficients
        on_diagonal = self.on_diagonal
        off_diagonal_upper = self.off_diagonal_upper
        off_diagonal_lower = self.off_diagonal_lower

        # Find the square of the off-diagonal blocks
        dtype = np.result_type(self.moments.dtype, self.on_diagonal[0].dtype)
        off_diagonal_upper_squared = np.zeros((self.nphys, self.nphys), dtype=dtype)
        off_diagonal_lower_squared = np.zeros((self.nphys, self.nphys), dtype=dtype)
        for j in range(i + 2):
            for k in range(i + 1):
                off_diagonal_upper_squared += (
                    coefficients[1][i + 1, k + 1]
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[0][i + 1, j]
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
            residual = coefficients[0][i + 1, j].astype(dtype, copy=True)
            residual -= coefficients[0][i + 1, j + 1] @ on_diagonal[i]
            residual -= coefficients[0][i, j + 1] @ off_diagonal_upper[i - 1]
            coefficients[0][i + 2, j + 1] = residual @ off_diagonal_lower_inv

            # Vertical recursion
            residual = coefficients[1][i + 1, j].astype(dtype, copy=True)
            residual -= on_diagonal[i] @ coefficients[1][i + 1, j + 1]
            residual -= off_diagonal_lower[i - 1] @ coefficients[1][i, j + 1]
            coefficients[1][i + 2, j + 1] = off_diagonal_upper_inv @ residual

        # Calculate the on-diagonal block
        on_diagonal[i + 1] = np.zeros((self.nphys, self.nphys), dtype=dtype)
        for j in range(i + 2):
            for k in range(i + 2):
                on_diagonal[i + 1] += (
                    coefficients[1][i + 2, k + 1]
                    @ self.orthogonalised_moment(j + k + 1)
                    @ coefficients[0][i + 2, j + 1]
                )

        # Get the error in the moments
        error_moments: float | None = None
        if self.calculate_errors:
            error_moments = self.moment_error(iteration=iteration)

        return error_sqrt, error_inv_sqrt, error_moments

    def solve(self, iteration: int | None = None) -> Spectral:
        """Solve the eigenvalue problem at a given iteration.

        Args:
            iteration: The iteration to get the results for.

        Returns:
            The :cls:`Spectral` object.
        """
        if iteration is None:
            iteration = self.max_cycle

        # Check if we're just returning the result
        if iteration == self.max_cycle and self.result is not None:
            return self.result

        # Diagonalise the block tridiagonal Hamiltonian
        on_diag = [self.on_diagonal[i] for i in range(iteration + 1)]
        off_diag_upper = [self.off_diagonal_upper[i] for i in range(iteration)]
        off_diag_lower = (
            [self.off_diagonal_lower[i] for i in range(iteration)] if not self.hermitian else None
        )
        hamiltonian = util.build_block_tridiagonal(on_diag, off_diag_upper, off_diag_lower)
        if self.hermitian:
            eigvals, eigvecs = util.eig(hamiltonian, hermitian=self.hermitian)
        else:
            eigvals, eigvecs_tuple = util.eig_lr(hamiltonian, hermitian=self.hermitian)
            eigvecs = np.array(eigvecs_tuple)

        # Unorthogonalise the eigenvectors
        metric_inv = self.orthogonalisation_metric_inv
        if self.hermitian:
            eigvecs[: self.nphys] = metric_inv @ eigvecs[: self.nphys]
        else:
            eigvecs[:, : self.nphys] = np.array(
                [
                    metric_inv.T.conj() @ eigvecs[0, : self.nphys],
                    metric_inv @ eigvecs[1, : self.nphys],
                ],
            )

        return Spectral(eigvals, eigvecs, self.nphys)

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
