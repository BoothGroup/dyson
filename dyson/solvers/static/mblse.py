"""Moment block Lanczos for moments of the self-energy."""

from __future__ import annotations

from abc import abstractmethod
import functools
from typing import TYPE_CHECKING

from dyson import numpy as np, util
from dyson.solvers.solver import StaticSolver
from dyson.solvers.static._mbl import BaseRecursionCoefficients, BaseMBL

if TYPE_CHECKING:
    from typing import Any, TypeAlias, TypeVar

    from dyson.typing import Array
    from dyson.lehmann import Lehmann

    Couplings: TypeAlias = Array | tuple[Array, Array]

    T = TypeVar("T", bound="BaseMBL")

# TODO: Use solvers for diagonalisation?


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

    def __init__(
        self,
        static: Array,
        moments: Array,
        max_cycle: int | None = None,
        hermitian: bool = True,
        force_orthogonality: bool = True,
        calculate_errors: bool = True,
    ) -> None:
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            moments: Moments of the self-energy.
            max_cycle: Maximum number of cycles.
            hermitian: Whether the self-energy is hermitian.
            force_orthogonality: Whether to force orthogonality of the recursion coefficients.
            calculate_errors: Whether to calculate errors.
        """
        self._static = static
        self._moments = moments
        self.max_cycle = max_cycle if max_cycle is not None else _infer_max_cycle(moments)
        self.hermitian = hermitian
        self.force_orthogonality = force_orthogonality
        self.calculate_errors = calculate_errors
        self._coefficients = self.Coefficients(
            self.nphys,
            hermitian=self.hermitian,
            dtype=np.result_type(static.dtype, moments.dtype).name,
            force_orthogonality=self.force_orthogonality,
        )
        self._on_diagonal: dict[int, Array] = {}
        self._off_diagonal: dict[int, Array] = {}

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> MBLSE:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        max_cycle = kwargs.get("max_cycle", 0)
        moments = self_energy.moments(range(2 * max_cycle + 2))
        return cls(static, moments, hermitian=self_energy.hermitian, **kwargs)

    def reconstruct_moments(self, iteration: int) -> Array:
        """Reconstruct the moments.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed moments.
        """
        self_energy = self.get_self_energy(iteration=iteration)
        energies = self_energy.energies
        left, right = self_energy.unpack_couplings()

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
        i = iteration + 1
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
            residual -= off_diagonal[i - 1].T.conj(), coefficients[i - 1, i, n]
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
        i = iteration + 1
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
            residual -= off_diagonal[i - 1], coefficients[i - 1, i, n]
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
        on_diagonal = [self.on_diagonal[i] for i in range(iteration + 2)]
        off_diagonal = [self.off_diagonal[i] for i in range(iteration + 1)]
        hamiltonian = util.build_block_tridiagonal(
            on_diagonal,
            off_diagonal,
            off_diagonal if not self.hermitian else None,
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
            couplings = self.off_diagonal[0].T.conj() @ rotated[: self.nphys]
        else:
            couplings = (
                self.off_diagonal[0] @ rotated[: self.nphys],
                self.off_diagonal[0].T.conj() @ np.linalg.inv(rotated).T.conj()[: self.nphys],
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
        if iteration == self.max_cycle and self.eigvals is not None and self.eigvecs is not None:
            eigvals = self.eigvals
            eigvecs = self.eigvecs
        else:
            self_energy = self.get_self_energy(iteration=iteration)
            eigvals, eigvecs = self_energy.diagonalise_matrix(self.static)

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
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        return self._static

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


class BlockMBLSE(StaticSolver):
    """Moment block Lanczos for block-wise moments of the self-energy.

    Args:
        static: Static part of the self-energy.
        moments: Blocks of moments of the self-energy.
    """

    Solver = MBLSE

    def __init__(
        self,
        static: Array,
        *moments: Array,
        max_cycle: int | None = None,
        hermitian: bool = True,
        force_orthogonality: bool = True,
        calculate_errors: bool = True,
    ) -> None:
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            moments: Blocks of moments of the self-energy.
            max_cycle: Maximum number of cycles.
            hermitian: Whether the self-energy is hermitian.
            force_orthogonality: Whether to force orthogonality of the recursion coefficients.
            calculate_errors: Whether to calculate errors.
        """
        self._solvers = [
            self.Solver(
                static,
                block,
                max_cycle=max_cycle,
                hermitian=hermitian,
                force_orthogonality=force_orthogonality,
                calculate_errors=calculate_errors,
            )
            for block in moments
        ]
        self.hermitian = hermitian

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> BlockMBLSE:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            For the block-wise solver, this function separates the self-energy into occupied and
            virtual moments.
        """
        max_cycle = kwargs.get("max_cycle", 0)
        self_energy_parts = (self_energy.occupied(), self_energy.virtual())
        moments = [
            self_energy_part.moments(range(2 * max_cycle + 2))
            for self_energy_part in self_energy_parts
        ]
        hermitian = all(self_energy_part.hermitian for self_energy_part in self_energy_parts)
        return cls(static, *moments, hermitian=hermitian, **kwargs)

    def kernel(self) -> None:
        """Run the solver."""
        # Run the solvers
        for solver in self.solvers:
            solver.kernel()
        self.eigvals, self.eigvecs = self.get_eigenfunctions()

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
            iteration = min(solver.max_cycle for solver in self.solvers)
        if kwargs:
            raise TypeError(
                f"get_auxiliaries() got unexpected keyword argument {next(iter(kwargs))}"
            )

        # Combine the energies and couplings
        energies_list: list[Array] = []
        couplings_list: list[Couplings] = []
        for solver in self.solvers:
            energies_i, couplings_i = solver.get_auxiliaries(iteration=iteration)
            energies_list.append(energies_i)
            couplings_list.append(couplings_i)
        energies = np.concatenate(energies_list)
        couplings: Couplings
        if any(isinstance(coupling, tuple) for coupling in couplings_list):
            couplings_list = [
                coupling_i if isinstance(coupling_i, tuple) else (coupling_i, coupling_i)
                for coupling_i in couplings_list
            ]
            couplings = (
                np.concatenate([coupling_i[0] for coupling_i in couplings_list], axis=1),
                np.concatenate([coupling_i[1] for coupling_i in couplings_list], axis=1),
            )
        else:
            couplings = np.concatenate(couplings_list, axis=1)

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
        max_cycle = min(solver.max_cycle for solver in self.solvers)
        if iteration is None:
            iteration = max_cycle
        if kwargs:
            raise TypeError(
                f"get_eigenfunctions() got unexpected keyword argument {next(iter(kwargs))}"
            )

        # Get the eigenvalues and eigenvectors
        if iteration == max_cycle and self.eigvals is not None and self.eigvecs is not None:
            eigvals = self.eigvals
            eigvecs = self.eigvecs
        else:
            self_energy = self.get_self_energy(iteration=iteration)
            eigvals, eigvecs = self_energy.diagonalise_matrix(self.static)

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
    def solvers(self) -> list[MBLSE]:
        """Get the solvers."""
        return self._solvers

    @property
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        return self.get_static_self_energy()  # FIXME

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.solvers[0].nphys
