"""Downfolded frequency-space diagonalisation.""":w

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver

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
        function: Callable[[Array], Array],
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
    def from_self_energy(self, static: Array, self_energy: Lehmann, **kwargs: Any) -> Exact:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        eta = kwargs.pop("eta", 1e-3)
        function = lambda freq: self_energy.on_grid(
            np.asarray([freq]),
            eta=eta,
            ordering="time-ordered",
            axis="real",
        )[0]
        return Downfolded(
            static,
            function,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    def kernel(self) -> None:
        """Run the solver."""
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
            eigvals, eigvecs = np.linalg.eigh(matrix)
        else:
            eigvals, eigvecs = np.linalg.eig(matrix)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigvals)
        self.eigenvalues = eigvals[idx]
        self.eigenvectors = eigvecs[:, idx]
        self.converged = converged

    @property
    def static(self) -> Array:
        """Get the static part of the self-energy."""
        return self._static

    @property
    def function(self) -> Callable[[Array], Array]:
        """Get the function to return the downfolded self-energy at a given frequency."""
        return self._function

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._static.shape[0]
