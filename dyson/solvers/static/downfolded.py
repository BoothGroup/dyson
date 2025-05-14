"""Downfolded frequency-space diagonalisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np, util
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver
from dyson.grids.frequency import RealFrequencyGrid
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from typing import Any, Callable

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

    converged: bool | None = None

    def __init__(
        self,
        static: Array,
        function: Callable[[float], Array],
        guess: float = 0.0,
        max_cycle: int = 100,
        conv_tol: float = 1e-8,
        hermitian: bool = True,
    ):
        """Initialise the solver.

        Args:
            static: The static part of the self-energy.
            function: The function to return the downfolded self-energy at a given frequency, the
                only argument.
            guess: Initial guess for the eigenvalue.
            max_cycle: Maximum number of iterations.
            conv_tol: Convergence tolerance for the eigenvalue.
            hermitian: Whether the matrix is hermitian.
        """
        self._static = static
        self._function = function
        self.guess = guess
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.hermitian = hermitian

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> Downfolded:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        kwargs = kwargs.copy()
        eta = kwargs.pop("eta", 1e-3)

        def _function(freq: float) -> Array:
            """Evaluate the self-energy at the frequency."""
            grid = RealFrequencyGrid(freq)
            grid.eta = eta
            return grid.evaluate_lehmann(
                self_energy,
                ordering="time-ordered",
            )

        return cls(
            static,
            _function,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Initialise the guess
        root = self.guess
        root_prev = 0.0

        converged = False
        for cycle in range(1, self.max_cycle + 1):
            # Update the root
            matrix = self.static + self.function(root)
            roots = np.linalg.eigvals(matrix)
            root_prev = root
            root = roots[np.argmin(np.abs(roots - self.guess))]

            # Check for convergence
            if np.abs(root - root_prev) < self.conv_tol:
                converged = True
                break

        # Get final eigenvalues and eigenvectors
        matrix = self.static + self.function(root)
        if self.hermitian:
            eigvals, eigvecs = util.eig(matrix, hermitian=self.hermitian)
        else:
            eigvals, eigvecs_tuple = util.eig_lr(matrix, hermitian=self.hermitian)
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
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._static.shape[0]
