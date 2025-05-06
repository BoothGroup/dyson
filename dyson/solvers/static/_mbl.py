"""Common functionality for moment block Lanczos solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import TYPE_CHECKING, overload
import warnings

from dyson import numpy as np, util
from dyson.solvers.solver import StaticSolver

if TYPE_CHECKING:
    from typing import TypeAlias, Literal

    from dyson.typing import Array

# TODO: reimplement caching


class BaseRecursionCoefficients(ABC):
    """Base class for recursion coefficients for the moment block Lanczos algorithms.

    Args:
        nphys: Number of physical degrees of freedom.
    """

    def __init__(
        self,
        nphys: int,
        hermitian: bool = True,
        force_orthogonality: bool = True,
        dtype: str = "float64",
    ):
        """Initialise the recursion coefficients."""
        self._nphys = nphys
        self._dtype = dtype
        self._zero = np.zeros((nphys, nphys), dtype=dtype)
        self._data: dict[tuple[int, ...], Array] = {}
        self.hermitian = hermitian
        self.force_orthogonality = force_orthogonality

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys

    @property
    def dtype(self) -> str:
        """Get the data type of the recursion coefficients."""
        return self._dtype

    @abstractmethod
    def __getitem__(self, key: tuple[int, ...]) -> Array:
        """Get the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.

        Returns:
            The recursion coefficients.
        """
        pass

    @abstractmethod
    def __setitem__(self, key: tuple[int, ...], value: Array) -> None:
        """Set the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.
            value: The recursion coefficients.
        """
        pass


class BaseMBL(StaticSolver):
    """Base class for moment block Lanczos solvers."""

    Coefficients: type[BaseRecursionCoefficients]

    _moments: Array

    max_cycle: int
    force_orthogonality: bool
    calculate_errors: bool

    def kernel(self) -> None:
        """Run the solver."""
        # pylint: disable=unused-variable

        # Run the solver
        for iteration in range(self.max_cycle + 1):  # TODO: check
            error_sqrt, error_inv_sqrt, error_moments = self.recurrence_iteration(iteration)

            error_decomp = max(error_sqrt, error_inv_sqrt) if self.calculate_errors else 0.0
            if error_decomp > 1e-10 and self.hermitian:
                warnings.warn(
                    f"Space contributing non-zero weight to the moments ({error_decomp}) was "
                    f"removed during iteration {iteration}. Allowing complex eigenvalues by "
                    "setting hermitian=False may help resolve this.",
                    UserWarning,
                    2,
                )
            elif error_decomp > 1e-10:
                warnings.warn(
                    f"Space contributing non-zero weight to the moments ({error_decomp}) was "
                    f"removed during iteration {iteration}. Since hermitian=False was set, this "
                    "likely indicates singularities which may indicate convergence of the moments.",
                    UserWarning,
                    2,
                )

        # Diagonalise the compressed self-energy
        self.eigvals, self.eigvecs = self.get_eigenfunctions(iteration=self.max_cycle)

    @functools.cached_property
    def orthogonalisation_metric(self) -> Array:
        """Get the orthogonalisation metric."""
        return util.matrix_power(self.moments[0], -0.5, hermitian=self.hermitian)

    @functools.cached_property
    def orthogonalisation_metric_inv(self) -> Array:
        """Get the inverse of the orthogonalisation metric."""
        return util.matrix_power(self.moments[0], 0.5, hermitian=self.hermitian)

    @functools.lru_cache(maxsize=64)
    def orthogonalised_moment(self, order: int) -> Array:
        """Compute an orthogonalised moment.

        Args:
            order: The order of the moment.

        Returns:
            The orthogonalised moment.
        """
        return self.orthogonalisation_metric @ self.moments[order] @ self.orthogonalisation_metric

    @abstractmethod
    def reconstruct_moments(self, iteration: int) -> Array:
        """Reconstruct the moments.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed moments.
        """
        pass

    def moment_error(self, iteration: int | None = None):
        """Get the moment error at a given iteration.

        Args:
            iteration: The iteration to check.
        """
        if iteration is None:
            iteration = self.max_cycle

        # Construct the recovered moments
        moments = self.reconstruct_moments(iteration)

        # Get the error
        error = sum(
            util.scaled_error(predicted, actual)
            for predicted, actual in zip(moments, self.moments[: 2 * iteration + 2])
        )

        return error

    @abstractmethod
    def initialise_recurrence(self) -> tuple[float | None, float | None, float | None]:
        """Initialise the recurrence (zeroth iteration).

        Returns:
            If :attr:`calculate_errors`, the error metrics in the square root of the off-diagonal
            block, the inverse square root of the off-diagonal block, and the error in the
            recovered moments. If not, all three are `None`.
        """
        pass

    @abstractmethod
    def _recurrence_iteration_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a Hermitian self-energy."""
        pass

    @abstractmethod
    def _recurrence_iteration_non_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a non-Hermitian self-energy."""
        pass

    def recurrence_iteration(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence.

        Args:
            iteration: The iteration to perform.

        Returns:
            If :attr:`calculate_errors`, the error metrics in the square root of the off-diagonal
            block, the inverse square root of the off-diagonal block, and the error in the
            recovered moments. If not, all three are `None`.
        """
        if iteration == 0:
            return self.initialise_recurrence()
        if iteration > self.max_cycle:
            raise ValueError(f"Iteration {iteration} exceeds max_cycle {self.max_cycle}.")
        if self.hermitian:
            return self._recurrence_iteration_hermitian(iteration)
        return self._recurrence_iteration_non_hermitian(iteration)

    @property
    @abstractmethod
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        pass

    @property
    def moments(self) -> Array:
        """Get the moments of the self-energy."""
        return self._moments

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.moments.shape[-1]
