"""Downfolded frequency-space diagonalisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import Progress
import scipy.linalg

from dyson import numpy as np
from dyson import util, console, printing
from dyson.grids.frequency import RealFrequencyGrid
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from typing import Any, Callable

    from dyson.expressions.expression import BaseExpression
    from dyson.typing import Array

# TODO: Use Newton solver as C* Σ(ω) C - ω = 0
# TODO: Diagonal version


class Downfolded(StaticSolver):
    r"""Downfolded frequency-space diagonalisation.

    Self-consistently satisfies the eigenvalue problem

    .. math::
        \Sigma(\omega) C = \omega C

    where :math:`\Sigma(\omega)` is the downfolded self-energy.

    Args:
        static: The static part of the self-energy.
        function: The function to return the downfolded self-energy at a given frequency, the only
            argument.
    """

    guess: float = 0.0
    max_cycle: int = 100
    conv_tol: float = 1e-8
    hermitian: bool = True
    _options: set[str] = {"guess", "max_cycle", "conv_tol", "hermitian"}

    converged: bool | None = None

    def __init__(  # noqa: D417
        self,
        static: Array,
        function: Callable[[float], Array],
        overlap: Array | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            static: The static part of the self-energy.
            function: The function to return the downfolded self-energy at a given frequency, the
                only argument.
            overlap: Overlap matrix for the physical space.
            guess: Initial guess for the eigenvalue.
            max_cycle: Maximum number of iterations.
            conv_tol: Convergence tolerance for the eigenvalue.
            hermitian: Whether the matrix is hermitian.
        """
        self._static = static
        self._function = function
        self._overlap = overlap
        self.set_options(**kwargs)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.static.ndim != 2 or self.static.shape[0] != self.static.shape[1]:
            raise ValueError("static must be a square matrix.")
        if not callable(self.function):
            raise ValueError("function must be a callable that takes a single float argument.")
        if self.overlap is not None and (
            self.overlap.ndim != 2 or self.overlap.shape[0] != self.overlap.shape[1]
        ):
            raise ValueError("overlap must be a square matrix or None.")
        if self.overlap is not None and self.overlap.shape != self.static.shape:
            raise ValueError("overlap must have the same shape as static.")

        # Print the input information
        console.print(f"Matrix shape: [input]{self.static.shape}[/input]")
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        if self.overlap is not None:
            cond = printing.format_float(np.linalg.cond(self.overlap), threshold=1e10, scientific=True, precision=4)
            console.print(f"Overlap condition number: {cond}")

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> Downfolded:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        kwargs = kwargs.copy()
        eta = kwargs.pop("eta", 1e-3)

        def _function(freq: float) -> Array:
            """Evaluate the self-energy at the frequency."""
            grid = RealFrequencyGrid(1, buffer=np.array([freq]))
            grid.eta = eta
            return grid.evaluate_lehmann(self_energy, ordering="time-ordered")[0]

        return cls(
            static,
            _function,
            overlap=overlap,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> Downfolded:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            "Cannot instantiate Downfolded from expression, use from_self_energy instead."
        )

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Get the table
        table = printing.ConvergencePrinter(("Best root",), ("Change",), (self.conv_tol,))
        progress = printing.IterationsPrinter(self.max_cycle)
        progress.start()

        # Initialise the guess
        root = self.guess
        root_prev = 0.0

        converged = False
        for cycle in range(1, self.max_cycle + 1):
            # Update the root
            matrix = self.static + self.function(root)
            roots = scipy.linalg.eigvals(matrix, b=self.overlap)
            root_prev = root
            root = roots[np.argmin(np.abs(roots - self.guess))]

            # Check for convergence
            converged = np.abs(root - root_prev) < self.conv_tol
            table.add_row(cycle, (root,), (root - root_prev,))
            progress.update(cycle)
            if converged:
                break

        progress.stop()
        table.print()

        # Get final eigenvalues and eigenvectors
        matrix = self.static + self.function(root)
        if self.hermitian:
            eigvals, eigvecs = util.eig(matrix, hermitian=self.hermitian, overlap=self.overlap)
        else:
            eigvals, eigvecs_tuple = util.eig_lr(matrix, hermitian=self.hermitian, overlap=self.overlap)
            eigvecs = np.array(eigvecs_tuple)

        # Store the results
        self.result = Spectral(eigvals, eigvecs, self.nphys)
        self.converged = converged

        return self.result

    @property
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        return self._static

    @property
    def function(self) -> Callable[[float], Array]:
        """Get the function to return the downfolded self-energy at a given frequency."""
        return self._function

    @property
    def overlap(self) -> Array | None:
        """Get the overlap matrix for the physical space."""
        return self._overlap

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._static.shape[0]
