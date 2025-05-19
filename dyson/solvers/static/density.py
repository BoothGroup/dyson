"""Density matrix relaxing solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import lib

from dyson import numpy as np
from dyson.lehmann import Lehmann, shift_energies
from dyson.solvers.solver import StaticSolver
from dyson.solvers.static.exact import Exact
from dyson.solvers.static.chempot import AuxiliaryShift, AufbauPrinciple

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, TypeAlias

    from dyson.typing import Array
    from dyson.spectral import Spectral
    from dyson.expression.expression import Expression


class DensityRelaxation(StaticSolver):
    """Solve a self-energy and relax the density matrix in the presence of the auxiliaries.

    Args:
        get_static: Function to get the static self-energy (including Fock contributions) for a
            given density matrix.
        self_energy: Self-energy.
        nelec: Target number of electrons.
    """

    occupancy: float = 2.0
    solver_outer: type[AuxiliaryShift] = AuxiliaryShift
    solver_inner: type[AufbauPrinciple] = AufbauPrinciple
    diis_min_space: int = 2
    diis_max_space: int = 8
    max_cycle_outer: int = 20
    max_cycle_inner: int = 50
    conv_tol: float = 1e-8
    _options: set[str] = {
        "occupancy", "solver_outer", "solver_inner", "diis_min_space", "diis_max_space",
        "max_cycle_outer", "max_cycle_inner", "conv_tol"
    }

    converged: bool | None = None

    def __init__(
        self,
        get_static: Callable[[Array], Array],
        self_energy: Lehmann,
        nelec: int,
        **kwargs: Any,
    ):
        """Initialise the solver

        Args:
            get_static: Function to get the static self-energy (including Fock contributions) for a
                given density matrix.
            self_energy: Self-energy.
            nelec: Target number of electrons.
            occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
                otherwise.
            solver_outer: Solver to use for the self-energy and chemical potential search in the
                outer loop.
            solver_inner: Solver to use for the self-energy and chemical potential search in the
                inner loop.
            diis_min_space: Minimum size of the DIIS space.
            diis_max_space: Maximum size of the DIIS space.
            max_cycle_outer: Maximum number of outer iterations.
            max_cycle_inner: Maximum number of inner iterations.
            conv_tol: Convergence tolerance in the density matrix.
        """
        self._get_static = get_static
        self._self_energy = self_energy
        self._nelec = nelec
        for key, val in kwargs.items():
            if key not in self._options:
                raise ValueError(f"Unknown option for {self.__class__.__name__}: {key}")
            setattr(self, key, val)

    @classmethod
    def from_self_energy(
        cls, static: Array, self_energy: Lehmann, **kwargs: Any
    ) -> DensityRelaxation:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            To initialise this solver from a self-energy, the `nelec` and `get_static` keyword
            arguments must be provided.
        """
        if "nelec" not in kwargs:
            raise ValueError("Missing required argument nelec.")
        if "get_static" not in kwargs:
            raise ValueError("Missing required argument get_static.")
        kwargs = kwargs.copy()
        nelec = kwargs.pop("nelec")
        get_static = kwargs.pop("get_static")
        return cls(get_static, self_energy, nelec, **kwargs)

    @classmethod
    def from_expression(cls, expression: Expression, **kwargs: Any) -> DensityRelaxation:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            "Cannot instantiate DensityRelaxation from expression, use from_self_energy instead."
        )

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        self_energy = self.self_energy
        nocc = self.nelec // self.occupancy
        rdm1 = np.diag(np.arange(self.nphys) < nocc).astype(self_energy.dtype) * self.occupancy
        static = self.get_static(rdm1)

        converged = False
        eigvals: Array | None = None
        eigvecs: Array | None = None
        for cycle_outer in range(1, self.max_cycle_outer + 1):
            # Solve the self-energy
            solver_outer = self.solver_outer.from_self_energy(static, self_energy, nelec=self.nelec)
            result = solver_outer.kernel()

            # Initialise DIIS for the inner loop
            diis = lib.diis.DIIS()
            diis.space = self.diis_min_space
            diis.max_space = self.diis_max_space
            diis.incore = True
            diis.verbose = 0

            for cycle_inner in range(1, self.max_cycle_inner + 1):
                # Solve the self-energy
                solver_inner = self.solver_inner.from_self_energy(
                    static, self_energy, nelec=self.nelec
                )
                result = solver_inner.kernel()

                # Get the density matrix
                greens_function = result.get_greens_function()
                rdm1_prev = rdm1.copy()
                rdm1 = greens_function.occupied().moment(0) * self.occupancy

                # Update the static self-energy
                static = self.get_static(rdm1)
                try:
                    static = diis.update(static, xerr=None)
                except np.linalg.LinAlgError:
                    pass

                # Check for convergence
                error = np.linalg.norm(rdm1 - rdm1_prev, ord=np.inf)
                if error < self.conv_tol:
                    break

            # Check for convergence
            if error < self.conv_tol and solver_outer.converged:
                converged = True
                break

        # Set the results
        self.converged = converged
        self.result = result

        return result

    @property
    def get_static(self) -> Callable[[Array], Array]:
        """Get the static self-energy function."""
        return self._get_static

    @property
    def self_energy(self) -> Lehmann:
        """Get the self-energy."""
        return self._self_energy

    @property
    def nelec(self) -> int:
        """Get the target number of electrons."""
        return self._nelec

    @property
    def nphys(self) -> int:
        """Get the number of physical states."""
        return self.self_energy.nphys
