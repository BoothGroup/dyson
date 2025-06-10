"""Chemical potential optimising solvers."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING

import scipy.optimize

from dyson import console, printing, util
from dyson import numpy as np
from dyson.lehmann import Lehmann, shift_energies
from dyson.solvers.solver import StaticSolver
from dyson.solvers.static.exact import Exact

if TYPE_CHECKING:
    from typing import Any, Literal

    from dyson.expressions.expression import BaseExpression
    from dyson.spectral import Spectral
    from dyson.typing import Array


def _warn_or_raise_if_negative_weight(
    weight: float | Array, hermitian: bool, tol: float = 1e-6
) -> None:
    """Warn or raise an error for negative weights.

    Args:
        weight: Weight to check.
        hermitian: Whether the Green's function is hermitian.
        tol: Tolerance for the weight to be considered negative.

    Raises:
        ValueError: If the weight is negative and the Green's function is hermitian.
        UserWarning: If the weight is negative and the Green's function is not hermitian.
    """
    if not isinstance(weight, np.ndarray):
        weight = np.array(weight)
    if np.any(weight < -tol):
        if hermitian:
            raise ValueError(
                f"Negative number of electrons in state: {weight:.6f}. This should be "
                "impossible for a hermitian Green's function."
            )
        else:
            warnings.warn(
                f"Negative number of electrons in state: {weight:.6f}. This is possible for "
                "a non-hermitian Green's function, but may be problematic for finding the "
                "chemical potential. Consider using the global method.",
                UserWarning,
            )


def search_aufbau_global(
    greens_function: Lehmann, nelec: int, occupancy: float = 2.0
) -> tuple[float, float]:
    """Search for a chemical potential in a Green's function using a global minimisation.

    Args:
        greens_function: Green's function.
        nelec: Target number of electrons.
        occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
            otherwise.

    Returns:
        The chemical potential and the error in the number of electrons.
    """
    energies = greens_function.energies
    weights = greens_function.weights(occupancy=occupancy)
    cumweights = np.cumsum(weights)

    # Find the global minimum
    i = np.argmin(np.abs(cumweights - nelec))
    error = cumweights[i] - nelec
    homo = i
    lumo = i + 1

    # Find the chemical potential
    if homo == -1:
        chempot = energies[lumo].real - 1e-4
    elif lumo == energies.size:
        chempot = energies[homo].real + 1e-4
    else:
        chempot = 0.5 * (energies[homo] + energies[lumo]).real

    return chempot, error


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
        number = np.vdot(left[:, i], right[:, i]).real * occupancy
        _warn_or_raise_if_negative_weight(number, greens_function.hermitian)
        sum_i, sum_j = sum_j, sum_j + number
        if sum_i < nelec <= sum_j:
            break

    # Find the best HOMO
    if abs(sum_i - nelec) < abs(sum_j - nelec):
        homo = i - 1
        error = nelec - sum_i
    else:
        homo = i
        error = nelec - sum_j
    lumo = homo + 1

    # Find the chemical potential
    if homo == -1:
        chempot = energies[lumo].real - 1e-4
    elif lumo == energies.size:
        chempot = energies[homo].real + 1e-4
    else:
        chempot = 0.5 * (energies[homo] + energies[lumo]).real

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
            low = mid
            mid = mid + (high - low) // 2
        else:
            high = mid
            mid = mid - (high - low) // 2
        if mid in {low, high}:
            break
    else:
        raise ValueError("Failed to converge bisection")
    sum_i = cumweights[low]
    sum_j = cumweights[high]

    # Find the best HOMO
    if abs(sum_i - nelec) < abs(sum_j - nelec):
        homo = low
        error = nelec - sum_i
    else:
        homo = high
        error = nelec - sum_j
    lumo = homo + 1

    # Find the chemical potential
    if homo == -1:
        chempot = energies[lumo].real - 1e-4
    elif lumo == energies.size:
        chempot = energies[homo].real + 1e-4
    else:
        chempot = 0.5 * (energies[homo] + energies[lumo]).real

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
    _overlap: Array | None

    error: float | None = None
    chempot: float | None = None
    converged: bool | None = None

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.static.ndim != 2 or self.static.shape[0] != self.static.shape[1]:
            raise ValueError("static must be a square matrix.")
        if self.self_energy.nphys != self.static.shape[0]:
            raise ValueError(
                "self_energy must have the same number of physical degrees of freedom as static."
            )
        if self.overlap is not None and (
            self.overlap.ndim != 2 or self.overlap.shape[0] != self.overlap.shape[1]
        ):
            raise ValueError("overlap must be a square matrix or None.")
        if self.overlap is not None and self.overlap.shape != self.static.shape:
            raise ValueError("overlap must have the same shape as static.")

        # Print the input information
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        console.print(f"Number of auxiliary states: [input]{self.self_energy.naux}[/input]")
        console.print(f"Target number of electrons: [input]{self.nelec}[/input]")
        if self.overlap is not None:
            cond = printing.format_float(
                np.linalg.cond(self.overlap), threshold=1e10, scientific=True, precision=4
            )
            console.print(f"Overlap condition number: {cond}")

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
    def overlap(self) -> Array | None:
        """Get the overlap matrix, if available."""
        return self._overlap

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

    occupancy: float = 2.0
    solver: type[Exact] = Exact
    method: Literal["direct", "bisect", "global"] = "global"
    _options: set[str] = {"occupancy", "solver", "method"}

    def __init__(  # noqa: D417
        self,
        static: Array,
        self_energy: Lehmann,
        nelec: int,
        overlap: Array | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            nelec: Target number of electrons.
            overlap: Overlap matrix for the physical space.
            occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
                otherwise.
            solver: Solver to use for the self-energy.
            method: Method to use for the chemical potential search.
        """
        self._static = static
        self._self_energy = self_energy
        self._nelec = nelec
        self._overlap = overlap
        self.set_options(**kwargs)

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        assert self.result is not None
        assert self.chempot is not None
        assert self.error is not None
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print("")
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )
        cpt = printing.format_float(self.chempot)
        err = printing.format_float(self.error, threshold=1e-3, precision=4, scientific=True)
        console.print(f"Chemical potential: [output]{cpt}[/output]")
        console.print(f"Error in number of electrons: [output]{err}[/output]")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> AufbauPrinciple:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
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
        return cls(static, self_energy, nelec, overlap=overlap, **kwargs)

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> AufbauPrinciple:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            "Cannot instantiate AufbauPrinciple from expression, use from_self_energy instead."
        )

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Solve the self-energy
        with printing.quiet:
            solver = self.solver.from_self_energy(
                self.static, self.self_energy, overlap=self.overlap
            )
            result = solver.kernel()
        greens_function = result.get_greens_function()

        # Get the chemical potential and error
        if self.method == "direct":
            chempot, error = search_aufbau_direct(greens_function, self.nelec, self.occupancy)
        elif self.method == "bisect":
            chempot, error = search_aufbau_bisect(greens_function, self.nelec, self.occupancy)
        elif self.method == "global":
            chempot, error = search_aufbau_global(greens_function, self.nelec, self.occupancy)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        result.chempot = chempot

        # Set the results
        self.result = result
        self.chempot = chempot
        self.error = error
        self.converged = True

        return result


class AuxiliaryShift(ChemicalPotentialSolver):
    """Shift the self-energy auxiliaries to best assign a chemical potential.

    Args:
        static: Static part of the self-energy.
        self_energy: Self-energy.
        nelec: Target number of electrons.

    Notes:
        Convergence is met when either of the thresholds `conv_tol` or `conv_tol_grad` are met,
        rather than both, due to constraints of the :meth:`scipy.optimize.minimize` method.
    """

    occupancy: float = 2.0
    solver: type[AufbauPrinciple] = AufbauPrinciple
    max_cycle: int = 200
    conv_tol: float = 1e-8
    conv_tol_grad: float = 0.0
    guess: float = 0.0
    _options: set[str] = {"occupancy", "solver", "max_cycle", "conv_tol", "conv_tol_grad", "guess"}

    shift: float | None = None

    def __init__(  # noqa: D417
        self,
        static: Array,
        self_energy: Lehmann,
        nelec: int,
        overlap: Array | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            nelec: Target number of electrons.
            overlap: Overlap matrix for the physical space.
            occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
                otherwise.
            solver: Solver to use for the self-energy and chemical potential search.
            max_cycle: Maximum number of iterations.
            conv_tol: Convergence tolerance for the number of electrons.
            conv_tol_grad: Convergence tolerance for the gradient of the objective function.
            guess: Initial guess for the chemical potential.
        """
        self._static = static
        self._self_energy = self_energy
        self._nelec = nelec
        self._overlap = overlap
        self.set_options(**kwargs)

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        assert self.result is not None
        assert self.chempot is not None
        assert self.error is not None
        assert self.shift is not None
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )
        cpt = printing.format_float(self.chempot)
        err = printing.format_float(self.error, threshold=1e-3, precision=4, scientific=True)
        shift = printing.format_float(self.shift, precision=4, scientific=True)
        console.print(f"Chemical potential: [output]{cpt}[/output]")
        console.print(f"Auxiliary shift: [output]{shift}[/output]")
        console.print(f"Error in number of electrons: [output]{err}[/output]")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> AuxiliaryShift:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
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
        return cls(static, self_energy, nelec, overlap=overlap, **kwargs)

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> AuxiliaryShift:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            "Cannot instantiate AuxiliaryShift from expression, use from_self_energy instead."
        )

    def objective(self, shift: float) -> float:
        """Objective function for the chemical potential search.

        Args:
            shift: Shift to apply to the self-energy.

        Returns:
            The error in the number of electrons.
        """
        with printing.quiet:
            with shift_energies(self.self_energy, np.ravel(shift)[0]):
                solver = self.solver.from_self_energy(
                    self.static, self.self_energy, nelec=self.nelec, overlap=self.overlap
                )
                solver.kernel()
        assert solver.error is not None
        return solver.error**2

    @functools.lru_cache(maxsize=16)
    def gradient(self, shift: float) -> tuple[float, Array]:
        """Gradient of the objective function.

        Args:
            shift: Shift to apply to the self-energy.

        Returns:
            The error in the number of electrons, and the gradient of the error.
        """
        with printing.quiet:
            with shift_energies(self.self_energy, np.ravel(shift)[0]):
                solver = self.solver.from_self_energy(
                    self.static, self.self_energy, nelec=self.nelec, overlap=self.overlap
                )
                solver.kernel()
        assert solver.error is not None
        assert solver.result is not None
        eigvals = solver.result.eigvals
        left, right = util.unpack_vectors(solver.result.eigvecs)
        nphys = self.nphys
        nocc = np.count_nonzero(eigvals < solver.chempot)

        h1 = -left[nphys:, nocc:].T.conj() @ right[nphys:, :nocc]
        z = h1 / (eigvals[nocc:, None] - eigvals[None, :nocc])
        pert_coeff_occ_left = left[:nphys, nocc:] @ z
        pert_coeff_occ_right = right[:nphys, nocc:] @ z
        pert_rdm1 = pert_coeff_occ_left @ pert_coeff_occ_right.T.conj() * 4.0  # occupancy?
        grad = np.trace(pert_rdm1).real * solver.error * self.occupancy

        return solver.error**2, grad

    def _minimize(self) -> scipy.optimize.OptimizeResult:
        """Minimise the objective function.

        Returns:
            The :class:`OptimizeResult` object from the minimizer.
        """
        # Get the table and callback function
        table = printing.ConvergencePrinter(
            ("Shift",), ("Error", "Gradient"), (self.conv_tol, self.conv_tol_grad)
        )
        progress = printing.IterationsPrinter(self.max_cycle)
        progress.start()
        cycle = 1

        def _callback(xk: Array) -> None:
            """Callback function for the minimizer."""
            nonlocal cycle
            error, grad = self.gradient(np.ravel(xk)[0])
            error = np.sqrt(error)
            table.add_row(cycle, (np.ravel(xk)[0],), (error, np.ravel(grad)[0]))
            progress.update(cycle)
            cycle += 1

        with util.catch_warnings(np.exceptions.ComplexWarning):
            opt = scipy.optimize.minimize(
                lambda x: self.gradient(np.ravel(x)[0]),
                x0=self.guess,
                method="TNC",
                jac=True,
                options=dict(
                    maxfun=self.max_cycle,
                    xtol=0.0,
                    ftol=self.conv_tol**2,
                    gtol=self.conv_tol_grad,
                ),
                callback=_callback,
            )

        progress.stop()
        table.print()

        return opt

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
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
        with printing.quiet:
            solver = self.solver.from_self_energy(
                self.static, self_energy, nelec=self.nelec, overlap=self.overlap
            )
            result = solver.kernel()

        # Set the results
        self.result = result
        self.chempot = solver.chempot
        self.error = solver.error
        self.converged = opt.success
        self.shift = np.ravel(opt.x)[0]

        return result
