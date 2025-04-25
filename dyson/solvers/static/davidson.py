"""Davidson algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from pyscf import lib

from dyson import numpy as np
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver

if TYPE_CHECKING:
    from typing import Any, Callable

    from dyson.typing import Array


def _pick_real_eigenvalues(
    eigvals: Array,
    eigvecs: Array,
    nroots: int,
    env: dict[str, Any],
    threshold=1e-3,
) -> tuple[Array, Array, int]:
    """Pick real eigenvalues."""
    iabs = np.abs(eigvals.imag)
    tol = max(threshold, np.sort(iabs)[min(eigvals.size, nroots) - 1])
    idx = np.where(iabs <= tol)[0]

    # Check we have enough real eigenvalues
    num = np.count_nonzero(iabs[idx] < threshold)
    if num < nroots and eigvals.size >= nroots:
        warnings.warn(
            f"Only {num} of requested {nroots} real eigenvalues found with threshold {tol:.2e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Make the eigenvalues real
    real_system = issubclass(env.get("dtype", np.float64), (complex, np.complexfloating))
    eigvals, eigvecs, _ = lib.linalg_helper._eigs_cmplx2real(
        eigvals,
        eigvecs,
        idx,
        real_eigenvectors=real_system,
    )

    # Sort the eigenvalues
    idx = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs, 0


class Davidson(StaticSolver):
    """Davidson algorithm for diagonalisation of the supermatrix form of the self-energy.

    Args:
        matvec: The matrix-vector operation for the self-energy supermatrix.
        diagonal: The diagonal of the self-energy supermatrix.
        nphys: Number of physical degrees of freedom.
    """

    converged: Array | None = None

    def __init__(
        self,
        matvec: Callable[[Array], Array],
        diagonal: Array,
        nphys: int,
        hermitian: bool = True,
        nroots: int = 1,
        max_cycle: int = 100,
        max_space: int = 16,
        conv_tol: float = 1e-8,
        conv_tol_residual: float = 1e-5,
    ):
        """Initialise the solver.

        Args:
            matvec: The matrix-vector operation for the self-energy supermatrix.
            diagonal: The diagonal of the self-energy supermatrix.
            nphys: Number of physical degrees of freedom.
            hermitian: Whether the matrix is hermitian.
            nroots: Number of roots to find.
            max_cycle: Maximum number of iterations.
            max_space: Maximum size of the subspace.
            conv_tol: Convergence tolerance for the eigenvalues.
            conv_tol_residual: Convergence tolerance for the residual.
        """
        self._matvec = matvec
        self._diagonal = diagonal
        self._nphys = nphys
        self.hermitian = hermitian
        self.nroots = nroots
        self.max_cycle = max_cycle
        self.max_space = max_space
        self.conv_tol = conv_tol
        self.conv_tol_residual = conv_tol_residual

    @classmethod
    def from_self_energy(self, static: Array, self_energy: Lehmann, **kwargs: Any) -> Davidson:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        return Davidson(
            lambda vector: self_energy.matvec(static, vector),
            self_energy.diagonal(static),
            self_energy.nphys,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    def get_guesses(self) -> list[Array]:
        """Get the initial guesses for the eigenvectors.

        Returns:
            Initial guesses for the eigenvectors.
        """
        args = np.argsort(np.abs(self.diagonal))
        return [np.eye(self.diagonal.size, 1, k=i).ravel() for i in args[: self.nroots]]

    def kernel(self) -> None:
        """Run the solver."""
        # Get the Davidson function
        function = (
            lib.linalg_helper.davidson1 if self.hermitian else lib.linalg_helper.davidson_nosym1
        )

        # Call the Davidson function
        converged, eigvals, eigvecs = function(
            lambda vectors: [self.matvec(vector) for vector in vectors],
            self.get_guesses(),
            self.diagonal,
            pick=_pick_real_eigenvalues,
            tol=self.conv_tol,
            tol_residual=self.conv_tol_residual,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=self.nroots,
            verbose=0,
        )
        eigvals = np.array(eigvals)
        eigvecs = np.array(eigvecs).T
        converged = np.array(converged)

        # Sort the eigenvalues
        mask = np.argsort(eigvals)
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
        converged = converged[mask]

        # Store the results
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.converged = converged

    @property
    def matvec(self) -> Callable[[Array], Array]:
        """Get the matrix-vector operation for the self-energy supermatrix."""
        return self._matvec

    @property
    def diagonal(self) -> Array:
        """Get the diagonal of the self-energy supermatrix."""
        return self._diagonal

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys
