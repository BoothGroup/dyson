"""Chemical potential optimising solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.optimize

from dyson import numpy as np, util
from dyson.lehmann import Lehmann, shift_energies
from dyson.solvers.solver import StaticSolver
from dyson.solvers.static.exact import Exact

if TYPE_CHECKING:
    from typing import Any, Literal

    from dyson.typing import Array


def search_aufbau_direct(
    greens_function: Lehmann, nelec: int, occupancy: float = 2.0
) -> tuple[float, float]:
    """Search for a chemical potential in a Green's function using the Aufbau principle.

    Args:
        greens_function: Green's function.
        nelec: Target number of electrons.
        occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
            otherwise.

    Returns:
        The chemical potential and the error in the number of electrons.
    """
    energies = greens_function.energies
    left, right = greens_function.unpack_couplings()

    # Find the two states bounding the chemical potential
    sum_i = sum_j = 0.0
    for i in range(greens_function.naux):
        number = (left[:, i] @ right[:, i].conj()).real * occupancy
        sum_i, sum_j = sum_j, sum_i + number
        if i and sum_i < nelec <= sum_j:
            break

    # Find the best HOMO
    if abs(sum_i - nelec) < abs(sum_j - nelec):
        homo = i - 1
        error = nelec - sum_i
    else:
        homo = i
        error = nelec - sum_j

    # Find the chemical potential
    lumo = homo + 1
    if homo < 0 or lumo >= energies.size:
        raise ValueError("Failed to identify HOMO and LUMO")
    chempot = 0.5 * (energies[homo] + energies[lumo])

    return chempot, error


def search_aufbau_bisect(
    greens_function: Lehmann, nelec: int, occupancy: float = 2.0, max_cycle: int = 1000
) -> tuple[float, float]:
    """Search for a chemical potential in a Green's function using Aufbau principle and bisection.

    Args:
        greens_function: Green's function.
        nelec: Target number of electrons.
        occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
            otherwise.
        max_cycle: Maximum number of iterations.

    Returns:
        The chemical potential and the error in the number of electrons.
    """
    energies = greens_function.energies
    weights = greens_function.weights(occupancy=occupancy)
    cumweights = np.cumsum(weights)

    # Find the two states bounding the chemical potential
    low, mid, high = 0, greens_function.naux // 2, greens_function.naux
    for cycle in range(1, max_cycle + 1):
        number = cumweights[mid]
        if number < nelec:
            low, mid = mid, mid + (high - low) // 2
        elif number > nelec:
            high, mid = mid, mid - (high - low) // 2
        if low == mid or mid == high:
            break
    else:
        raise ValueError("Failed to converge bisection")
    sum_i = cumweights[low]
    sum_j = cumweights[high]

    # Find the best HOMO
    if abs(sum_i - nelec) < abs(sum_j - nelec):
        homo = low - 1
        error = nelec - sum_i
    else:
        homo = high - 1
        error = nelec - sum_j

    # Find the chemical potential
    lumo = homo + 1
    if homo < 0 or lumo >= energies.size:
        raise ValueError("Failed to identify HOMO and LUMO")
    chempot = 0.5 * (energies[homo] + energies[lumo])

    return chempot, error


class ChemicalPotentialSolver(StaticSolver):
    """Base class for a solver for a self-energy that optimises the chemical potential.

    Args:
        static: Static part of the self-energy.
        self_energy: Self-energy.
        nelec: Target number of electrons.
    """

    _static: Array
    _self_energy: Lehmann
    _nelec: int

    error: float | None = None
    chempot: float | None = None
    converged: bool | None = None

    @property
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        return self._static

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
        """Get the number of physical degrees of freedom."""
        return self._self_energy.nphys


class AufbauPrinciple(ChemicalPotentialSolver):
    """Solve a self-energy and assign a chemical potential based on the Aufbau principle.

    Args:
        static: Static part of the self-energy.
        self_energy: Self-energy.
        nelec: Target number of electrons.
    """

    def __init__(
        self,
        static: Array,
        self_energy: Lehmann,
        nelec: int,
        occupancy: float = 2.0,
        solver: type[Exact] = Exact,
        method: Literal["direct", "bisect"] = "direct",
    ):
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            nelec: Target number of electrons.
            occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
                otherwise.
            solver: Solver to use for the self-energy.
            method: Method to use for the chemical potential search.
        """
        self._static = static
        self._self_energy = self_energy
        self._nelec = nelec
        self.occupancy = occupancy
        self.solver = solver
        self.method = method

    @classmethod
    def from_self_energy(
        cls, static: Array, self_energy: Lehmann, **kwargs: Any
    ) -> AufbauPrinciple:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            To initialise this solver from a self-energy, the `nelec` keyword argument must be
            provided.
        """
        if "nelec" not in kwargs:
            raise ValueError("Missing required argument nelec.")
        kwargs = kwargs.copy()
        nelec = kwargs.pop("nelec")
        return cls(static, self_energy, nelec, **kwargs)

    def kernel(self) -> None:
        """Run the solver."""
        # Solve the self-energy
        solver = self.solver.from_self_energy(self.static, self.self_energy)
        result = solver.kernel()
        greens_function = result.get_green_function()

        # Get the chemical potential and error
        if self.method == "direct":
            chempot, error = search_aufbau_direct(greens_function, self.nelec, self.occupancy)
        elif self.method == "bisect":
            chempot, error = search_aufbau_bisect(greens_function, self.nelec, self.occupancy)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        result.chempot = chempot

        # Set the results
        self.result = result
        self.chempot = chempot
        self.error = error
        self.converged = True


class AuxiliaryShift(ChemicalPotentialSolver):
    """Shift the self-energy auxiliaries to best assign a chemical potential.

    Args:
        static: Static part of the self-energy.
        self_energy: Self-energy.
        nelec: Target number of electrons.
    """

    shift: float | None = None

    def __init__(
        self,
        static: Array,
        self_energy: Lehmann,
        nelec: int,
        occupancy: float = 2.0,
        solver: type[AufbauPrinciple] = AufbauPrinciple,
        max_cycle: int = 200,
        conv_tol: float = 1e-8,
        guess: float = 0.0,
    ):
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            nelec: Target number of electrons.
            occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
                otherwise.
            solver: Solver to use for the self-energy and chemical potential search.
            max_cycle: Maximum number of iterations.
            conv_tol: Convergence tolerance for the number of electrons.
            guess: Initial guess for the chemical potential.
        """
        self._static = static
        self._self_energy = self_energy
        self._nelec = nelec
        self.occupancy = occupancy
        self.solver = solver
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.guess = guess

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> AuxiliaryShift:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            To initialise this solver from a self-energy, the `nelec` keyword argument must be
            provided.
        """
        if "nelec" not in kwargs:
            raise ValueError("Missing required argument nelec.")
        nelec = kwargs.pop("nelec")
        return cls(static, self_energy, nelec, **kwargs)

    def objective(self, shift: float) -> float:
        """Objective function for the chemical potential search.

        Args:
            shift: Shift to apply to the self-energy.

        Returns:
            The error in the number of electrons.
        """
        with shift_energies(self.self_energy, np.ravel(shift)[0]):
            solver = self.solver.from_self_energy(self.static, self.self_energy, nelec=self.nelec)
            solver.kernel()
        assert solver.error is not None
        return solver.error**2

    def gradient(self, shift: float) -> tuple[float, Array]:
        """Gradient of the objective function.

        Args:
            shift: Shift to apply to the self-energy.

        Returns:
            The error in the number of electrons, and the gradient of the error.
        """
        with shift_energies(self.self_energy, np.ravel(shift)[0]):
            solver = self.solver.from_self_energy(self.static, self.self_energy, nelec=self.nelec)
            solver.kernel()
        assert solver.error is not None
        eigvals, eigvecs = solver.get_eigenfunctions()
        left, right = util.unpack_vectors(eigvecs)
        nphys = self.nphys
        nocc = np.count_nonzero(eigvals < solver.chempot)

        h1 = -left[nphys:, nocc:].T.conj() @ right[nphys:, :nocc]
        z = h1 / (eigvals[nocc:, None] - eigvals[None, :nocc])
        pert_coeff_occ_left = left[:nphys, nocc:] @ z
        pert_coeff_occ_right = right[:nphys, nocc:] @ z
        pert_rdm1 = pert_coeff_occ_left @ pert_coeff_occ_right.T.conj() * 4.0  # occupancy?
        grad = np.trace(pert_rdm1).real * solver.error * self.occupancy

        return solver.error**2, grad

    def _callback(self, shift: float) -> None:
        """Callback function for the minimizer.

        Args:
            shift: Shift to apply to the self-energy.
        """
        pass

    def _minimize(self) -> scipy.optimize.OptimizeResult:
        """Minimise the objective function.

        Returns:
            The :class:`OptimizeResult` object from the minimizer.
        """
        return scipy.optimize.minimize(
            self.objective,
            x0=self.guess,
            method="TNC",
            jac=True,
            options=dict(
                maxfun=self.max_cycle,
                ftol=self.conv_tol**2,
                xtol=0.0,
                gtol=0.0,
            ),
            callback=self._callback,
        )

    def kernel(self) -> None:
        """Run the solver."""
        # Minimize the objective function
        opt = self._minimize()

        # Get the shifted self-energy
        self_energy = Lehmann(
            self.self_energy.energies + opt.x,
            self.self_energy.couplings,
            chempot=self.self_energy.chempot,
            sort=False,
        )

        # Solve the self-energy
        solver = self.solver.from_self_energy(self.static, self_energy, nelec=self.nelec)
        result = solver.kernel()

        # Set the results
        self.result = result
        self.chempot = solver.chempot
        self.error = solver.error
        self.converged = opt.success
        self.shift = opt.x
