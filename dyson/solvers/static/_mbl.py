"""Common functionality for moment block Lanczos solvers."""

from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util, console, printing
from dyson.solvers.solver import StaticSolver

if TYPE_CHECKING:
    from dyson.spectral import Spectral
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
    hermitian: bool = True
    force_orthogonality: bool = True
    calculate_errors: bool = True
    _options: set[str] = {"max_cycle", "hermitian", "force_orthogonality", "calculate_errors"}

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )
        if self.calculate_errors:
            error = printing.format_float(
                self.moment_error(iteration=self.max_cycle),
                threshold=1e-10,
                scientific=True,
                precision=4,
            )
            console.print(f"Error in the moments: {error}")

    @abstractmethod
    def solve(self, iteration: int | None = None) -> Spectral:
        """Solve the eigenvalue problem at a given iteration.

        Args:
            iteration: The iteration to get the results for.

        Returns:
            The :cls:`Spectral` object.
        """
        pass

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Get the table
        table = printing.ConvergencePrinter(
            (), ("Error in moments", "Error in sqrt", "Error in inv. sqrt"), (1e-10, 1e-10, 1e-10)
        )
        progress = printing.IterationsPrinter(self.max_cycle)
        progress.start()

        # Run the solver
        for iteration in range(self.max_cycle + 1):  # TODO: check
            error_sqrt, error_inv_sqrt, error_moments = self.recurrence_iteration(iteration)
            if not self.calculate_errors:
                error_sqrt = error_inv_sqrt = error_moments = np.nan
            table.add_row(iteration, (), (error_moments, error_sqrt, error_inv_sqrt))
            progress.update(iteration)

        progress.stop()
        table.print()

        # Diagonalise the compressed self-energy
        self.result = self.solve(iteration=self.max_cycle)

        return self.result

    @functools.cached_property
    def orthogonalisation_metric(self) -> Array:
        """Get the orthogonalisation metric."""
        return util.matrix_power(self.moments[0], -0.5, hermitian=self.hermitian)[0]

    @functools.cached_property
    def orthogonalisation_metric_inv(self) -> Array:
        """Get the inverse of the orthogonalisation metric."""
        return util.matrix_power(self.moments[0], 0.5, hermitian=self.hermitian)[0]

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
    def moments(self) -> Array:
        """Get the moments of the self-energy."""
        return self._moments

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.moments.shape[-1]
