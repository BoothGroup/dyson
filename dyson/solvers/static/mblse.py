"""Moment block Lanczos for moments of the self-energy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util, console, printing
from dyson.lehmann import Lehmann
from dyson.solvers.static._mbl import BaseMBL, BaseRecursionCoefficients
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from typing import Any, TypeVar

    from dyson.expressions.expression import BaseExpression
    from dyson.typing import Array

    T = TypeVar("T", bound="BaseMBL")


class RecursionCoefficients(BaseRecursionCoefficients):
    """Recursion coefficients for the moment block Lanczos algorithm for the self-energy.

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
        i, j, order = key
        if i == 0 or j == 0:
            return self._zero
        if i < j and self.hermitian:
            return self._data[j, i, order].T.conj()
        return self._data[i, j, order]

    def __setitem__(self, key: tuple[int, ...], value: Array) -> None:
        """Set the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.
            value: The recursion coefficients.
        """
        i, j, order = key
        if order == 0 and self.force_orthogonality:
            value = np.eye(self.nphys, dtype=self.dtype)
        if self.hermitian and i == j:
            value = 0.5 * util.hermi_sum(value)
        if i < j and self.hermitian:
            self._data[j, i, order] = value.T.conj()
        else:
            self._data[i, j, order] = value


def _infer_max_cycle(moments: Array) -> int:
    """Infer the maximum number of cycles from the moments."""
    return (moments.shape[0] - 2) // 2


class MBLSE(BaseMBL):
    """Moment block Lanczos for moments of the self-energy.

    Args:
        static: Static part of the self-energy.
        moments: Moments of the self-energy.
    """

    Coefficients = RecursionCoefficients

    def __init__(  # noqa: D417
        self,
        static: Array,
        moments: Array,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            moments: Moments of the self-energy.
            overlap: Overlap matrix for the physical space.
            max_cycle: Maximum number of cycles.
            hermitian: Whether the self-energy is hermitian.
            force_orthogonality: Whether to force orthogonality of the recursion coefficients.
            calculate_errors: Whether to calculate errors.
        """
        self._static = static
        self._moments = moments
        self._overlap = overlap
        self.max_cycle = kwargs["max_cycle"] if "max_cycle" in kwargs else _infer_max_cycle(moments)
        self.set_options(**kwargs)

        self._coefficients = self.Coefficients(
            self.nphys,
            hermitian=self.hermitian,
            dtype=np.result_type(static.dtype, moments.dtype).name,
            force_orthogonality=self.force_orthogonality,
        )
        self._on_diagonal: dict[int, Array] = {}
        self._off_diagonal: dict[int, Array] = {}

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.static.ndim != 2 or self.static.shape[0] != self.static.shape[1]:
            raise ValueError("static must be a square matrix.")
        if self.moments.ndim != 3 or self.moments.shape[1] != self.moments.shape[2]:
            raise ValueError(
                "moments must be a 3D array with the second and third dimensions equal."
            )
        if self.moments.shape[1] != self.static.shape[0]:
            raise ValueError(
                "moments must have the same shape as static in the last two dimensions."
            )
        if _infer_max_cycle(self.moments) < self.max_cycle:
            raise ValueError("not enough moments provided for the specified max_cycle.")

        # Print the input information
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        if self.overlap is not None:
            cond = printing.format_float(np.linalg.cond(self.overlap), threshold=1e10, scientific=True, precision=4)
            console.print(f"Overlap condition number: {cond}")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> MBLSE:
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
        moments = self_energy.moments(range(2 * max_cycle + 2))
        return cls(static, moments, hermitian=self_energy.hermitian, overlap=overlap, **kwargs)

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> MBLSE:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            "Cannot instantiate MBLSE from expression, use from_self_energy instead."
        )

    def reconstruct_moments(self, iteration: int) -> Array:
        """Reconstruct the moments.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed moments.
        """
        self_energy = self.solve(iteration=iteration).get_self_energy()
        return self_energy.moments(range(2 * iteration))

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

        # Initialise the coefficients
        for n in range(2 * self.max_cycle + 2):
            self.coefficients[1, 1, n] = self.orthogonalised_moment(n)

        # Initialise the blocks
        self.off_diagonal[0], error_sqrt = util.matrix_power(
            self.moments[0], 0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )
        self.on_diagonal[0] = self.static
        self.on_diagonal[1] = self.coefficients[1, 1, 1]

        # Get the error in the moments
        error_moments: float | None = None
        if self.calculate_errors:
            error_moments = self.moment_error(iteration=0)

        return error_sqrt, error_inv_sqrt, error_moments

    def _recurrence_iteration_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a Hermitian self-energy."""
        i = iteration
        coefficients = self.coefficients
        on_diagonal = self.on_diagonal
        off_diagonal = self.off_diagonal

        # Find the squre of the off-diagonal block
        off_diagonal_squared = coefficients[i, i, 2].copy()
        off_diagonal_squared -= util.hermi_sum(coefficients[i, i - 1, 1] @ off_diagonal[i - 1])
        off_diagonal_squared -= coefficients[i, i, 1] @ coefficients[i, i, 1]
        if iteration > 1:
            off_diagonal_squared += off_diagonal[i - 1].T.conj() @ off_diagonal[i - 1]

        # Get the off-diagonal block
        off_diagonal[i], error_sqrt = util.matrix_power(
            off_diagonal_squared, 0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )

        # Invert the off-diagonal block
        off_diagonal_inv, error_inv_sqrt = util.matrix_power(
            off_diagonal_squared, -0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )

        for n in range(2 * (self.max_cycle - iteration + 1)):
            # Horizontal recursion
            residual = coefficients[i, i, n + 1].copy()
            residual -= off_diagonal[i - 1].T.conj() @ coefficients[i - 1, i, n]
            residual -= on_diagonal[i] @ coefficients[i, i, n]
            coefficients[i + 1, i, n] = off_diagonal_inv @ residual

            # Diagonal recursion
            residual = coefficients[i, i, n + 2].copy()
            residual -= util.hermi_sum(coefficients[i, i - 1, n + 1] @ off_diagonal[i - 1])
            residual -= util.hermi_sum(coefficients[i, i, n + 1] @ on_diagonal[i])
            residual += util.hermi_sum(
                on_diagonal[i] @ coefficients[i, i - 1, n] @ off_diagonal[i - 1]
            )
            residual += util.hermi_sum(
                off_diagonal[i - 1].T.conj() @ coefficients[i - 1, i - 1, n] @ off_diagonal[i - 1]
            )
            residual += on_diagonal[i] @ coefficients[i, i, n] @ on_diagonal[i]
            coefficients[i + 1, i + 1, n] = off_diagonal_inv @ residual @ off_diagonal_inv.T.conj()

        # Extract the on-diagonal block
        on_diagonal[i + 1] = coefficients[i + 1, i + 1, 1].copy()

        # Get the error in the moments
        error_moments: float | None = None
        if self.calculate_errors:
            error_moments = self.moment_error(iteration=iteration)

        return error_sqrt, error_inv_sqrt, error_moments

    def _recurrence_iteration_non_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a non-Hermitian self-energy."""
        i = iteration
        coefficients = self.coefficients
        on_diagonal = self.on_diagonal
        off_diagonal = self.off_diagonal

        # Find the squre of the off-diagonal block
        off_diagonal_squared = coefficients[i, i, 2].copy()
        off_diagonal_squared -= coefficients[i, i, 1] @ coefficients[i, i, 1]
        off_diagonal_squared -= coefficients[i, i - 1, 1] @ off_diagonal[i - 1]
        off_diagonal_squared -= off_diagonal[i - 1] @ coefficients[i, i - 1, 1]
        if iteration > 1:
            off_diagonal_squared += off_diagonal[i - 1] @ off_diagonal[i - 1]

        # Get the off-diagonal block
        off_diagonal[i], error_sqrt = util.matrix_power(
            off_diagonal_squared, 0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )

        # Invert the off-diagonal block
        off_diagonal_inv, error_inv_sqrt = util.matrix_power(
            off_diagonal_squared, -0.5, hermitian=self.hermitian, return_error=self.calculate_errors
        )

        for n in range(2 * (self.max_cycle - iteration + 1)):
            # Horizontal recursion
            residual = coefficients[i, i, n + 1].copy()
            residual -= off_diagonal[i - 1] @ coefficients[i - 1, i, n]
            residual -= on_diagonal[i] @ coefficients[i, i, n]
            coefficients[i + 1, i, n] = off_diagonal_inv @ residual

            # Vertical recursion
            residual = coefficients[i, i, n + 1].copy()
            residual -= coefficients[i, i - 1, n] @ off_diagonal[i - 1]
            residual -= coefficients[i, i, n] @ on_diagonal[i]
            coefficients[i, i + 1, n] = residual @ off_diagonal_inv

            # Diagonal recursion
            residual = coefficients[i, i, n + 2].copy()
            residual -= coefficients[i, i - 1, n + 1] @ off_diagonal[i - 1]
            residual -= coefficients[i, i, n + 1] @ on_diagonal[i]
            residual -= off_diagonal[i - 1] @ coefficients[i - 1, i, n + 1]
            residual += off_diagonal[i - 1] @ coefficients[i - 1, i - 1, n] @ off_diagonal[i - 1]
            residual += off_diagonal[i - 1] @ coefficients[i - 1, i, n] @ on_diagonal[i]
            residual -= on_diagonal[i] @ coefficients[i, i, n + 1]
            residual += on_diagonal[i] @ coefficients[i, i - 1, n] @ off_diagonal[i - 1]
            residual += on_diagonal[i] @ coefficients[i, i, n] @ on_diagonal[i]
            coefficients[i + 1, i + 1, n] = off_diagonal_inv @ residual @ off_diagonal_inv

        # Extract the on-diagonal block
        on_diagonal[i + 1] = coefficients[i + 1, i + 1, 1].copy()

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
        # TODO inherit
        if iteration is None:
            iteration = self.max_cycle

        # Check if we're just returning the result
        if iteration == self.max_cycle and self.result is not None:
            return self.result

        # Get the supermatrix
        on_diag = [self.on_diagonal[i] for i in range(iteration + 2)]
        off_diag_upper = [self.off_diagonal[i] for i in range(iteration + 1)]
        off_diag_lower = (
            [self.off_diagonal[i] for i in range(iteration + 1)] if not self.hermitian else None
        )
        hamiltonian = util.build_block_tridiagonal(on_diag, off_diag_upper, off_diag_lower)

        # Diagonalise the subspace
        subspace = hamiltonian[self.nphys :, self.nphys :]
        energies, rotated = util.eig_lr(subspace, hermitian=self.hermitian)
        if self.hermitian:
            couplings = self.off_diagonal[0] @ rotated[0][: self.nphys]
        else:
            couplings = np.array(
                [
                    self.off_diagonal[0].T.conj() @ rotated[0][: self.nphys],
                    self.off_diagonal[0] @ rotated[1][: self.nphys],
                ]
            )

        return Spectral.from_self_energy(self.static, Lehmann(energies, couplings), overlap=self.overlap)

    @property
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        return self._static

    @property
    def overlap(self) -> Array | None:
        """Get the overlap matrix for the physical space."""
        return self._overlap

    @property
    def coefficients(self) -> BaseRecursionCoefficients:
        """Get the recursion coefficients."""
        return self._coefficients

    @property
    def on_diagonal(self) -> dict[int, Array]:
        """Get the on-diagonal blocks of the self-energy."""
        return self._on_diagonal

    @property
    def off_diagonal(self) -> dict[int, Array]:
        """Get the off-diagonal blocks of the self-energy."""
        return self._off_diagonal
