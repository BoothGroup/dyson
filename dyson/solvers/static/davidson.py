"""Davidson algorithm."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pyscf import lib

from dyson import console, printing, util
from dyson import numpy as np
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver
from dyson.spectral import Spectral

if TYPE_CHECKING:
    from typing import Any, Callable

    from dyson.expressions.expression import BaseExpression
    from dyson.typing import Array


def _pick_real_eigenvalues(
    eigvals: Array,
    eigvecs: Array,
    nroots: int,
    env: dict[str, Any],
    threshold: float = 1e-3,
) -> tuple[Array, Array, Array]:
    """Pick real eigenvalues."""
    iabs = np.abs(eigvals.imag)
    tol = max(threshold, np.sort(iabs)[min(eigvals.size, nroots) - 1])
    real_idx = np.where(iabs <= tol)[0]

    # Check we have enough real eigenvalues
    num = np.count_nonzero(iabs[real_idx] < threshold)
    if num < nroots and eigvals.size >= nroots:
        warnings.warn(
            f"Only {num} of requested {nroots} real eigenvalues found with threshold {tol:.2e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Sort the eigenvalues
    idx = real_idx[np.argsort(np.abs(eigvals[real_idx]))]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Make the eigenvalues real
    real_system = issubclass(env.get("dtype", np.float64).type, (complex, np.complexfloating))
    if real_system:
        degen_idx = np.where(eigvals.imag != 0)[0]
        if degen_idx.size > 0:
            eigvecs[:, degen_idx[1::2]] = eigvecs[:, degen_idx[1::2]].imag
        eigvecs = eigvecs.real

    return eigvals, eigvecs, idx


class Davidson(StaticSolver):
    """Davidson algorithm for diagonalisation of the supermatrix form of the self-energy.

    Args:
        matvec: The matrix-vector operation for the self-energy supermatrix.
        diagonal: The diagonal of the self-energy supermatrix.
        bra: The bra state vector mapping the supermatrix to the physical space.
        ket: The ket state vector mapping the supermatrix to the physical space.
    """

    hermitian: bool = True
    nroots: int = 1
    max_cycle: int = 100
    max_space: int = 16
    conv_tol: float = 1e-8
    conv_tol_residual: float = 1e-5
    _options: set[str] = {
        "hermitian",
        "nroots",
        "max_cycle",
        "max_space",
        "conv_tol",
        "conv_tol_residual",
    }

    converged: Array | None = None

    def __init__(  # noqa: D417
        self,
        matvec: Callable[[Array], Array],
        diagonal: Array,
        bra: Array,
        ket: Array | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            matvec: The matrix-vector operation for the self-energy supermatrix.
            diagonal: The diagonal of the self-energy supermatrix.
            bra: The bra state vector mapping the supermatrix to the physical space.
            ket: The ket state vector mapping the supermatrix to the physical space. If `None`, use
                the same vectors as `bra`.
            hermitian: Whether the matrix is hermitian.
            nroots: Number of roots to find.
            max_cycle: Maximum number of iterations.
            max_space: Maximum size of the subspace.
            conv_tol: Convergence tolerance for the eigenvalues.
            conv_tol_residual: Convergence tolerance for the residual.
        """
        self._matvec = matvec
        self._diagonal = diagonal
        self._bra = bra
        self._ket = ket if ket is not None else bra
        self.set_options(**kwargs)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.diagonal.ndim != 1:
            raise ValueError("diagonal must be a 1D array.")
        if self.bra.ndim != 2 or self.bra.shape[1] != self.diagonal.size:
            raise ValueError("bra must be a 2D array with the same number of columns as diagonal.")
        if self.ket is not None and (self.ket.ndim != 2 or self.ket.shape[1] != self.diagonal.size):
            raise ValueError("ket must be a 2D array with the same number of columns as diagonal.")
        if self.ket is not None and self.ket.shape[0] != self.bra.shape[0]:
            raise ValueError("ket must have the same number of rows as bra.")
        if not callable(self.matvec):
            raise ValueError("matvec must be a callable function.")

        # Print the input information
        console.print(f"Matrix shape: [input]{(self.diagonal.size, self.diagonal.size)}[/input]")
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        assert self.result is not None
        assert self.converged is not None
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )
        rating = "good" if np.all(self.converged) else "okay" if np.any(self.converged) else "bad"
        console.print(
            f"Converged [{rating}]{np.sum(self.converged)} of {self.nroots}[/{rating}] roots."
        )

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> Davidson:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        size = self_energy.nphys + self_energy.naux
        bra = ket = np.array([util.unit_vector(size, i) for i in range(self_energy.nphys)])
        if overlap is not None:
            hermitian = self_energy.hermitian
            orth = util.matrix_power(overlap, 0.5, hermitian=hermitian)[0]
            unorth = util.matrix_power(overlap, -0.5, hermitian=hermitian)[0]
            bra = util.rotate_subspace(bra, orth.T.conj())
            ket = util.rotate_subspace(ket, orth) if not hermitian else bra
            static = unorth @ static @ unorth
            self_energy = self_energy.rotate_couplings(
                unorth if hermitian else (unorth, unorth.T.conj())
            )
        return cls(
            lambda vector: self_energy.matvec(static, vector),
            self_energy.diagonal(static),
            bra,
            ket,
            hermitian=self_energy.hermitian,
            **kwargs,
        )

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> Davidson:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        diagonal = expression.diagonal()
        matvec = expression.apply_hamiltonian
        bra = np.array([expression.get_state_bra(i) for i in range(expression.nphys)])
        ket = (
            np.array([expression.get_state_ket(i) for i in range(expression.nphys)])
            if not expression.hermitian
            else None
        )
        return cls(
            matvec,
            diagonal,
            bra,
            ket,
            hermitian=expression.hermitian,
            **kwargs,
        )

    def get_guesses(self) -> list[Array]:
        """Get the initial guesses for the eigenvectors.

        Returns:
            Initial guesses for the eigenvectors.
        """
        args = np.argsort(np.abs(self.diagonal))
        dtype = "<f8" if self.hermitian else "<c16"
        return [util.unit_vector(self.diagonal.size, i, dtype=dtype) for i in args[: self.nroots]]

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Get the table and callback function
        table = printing.ConvergencePrinter(
            ("Smallest root",), ("Change", "Residual"), (self.conv_tol, self.conv_tol_residual)
        )
        progress = printing.IterationsPrinter(self.max_cycle)
        progress.start()

        def _callback(env: dict[str, Any]) -> None:
            """Callback function for the Davidson algorithm."""
            root = env["e"][np.argmin(np.abs(env["e"]))]
            table.add_row(
                env["icyc"] + 1, (root,), (np.max(np.abs(env["de"])), np.max(env["dx_norm"]))
            )
            progress.update(env["icyc"] + 1)
            del env

        # Call the Davidson function
        if self.hermitian:
            converged, eigvals, eigvecs = lib.linalg_helper.davidson1(
                lambda vectors: [self.matvec(vector) for vector in vectors],
                self.get_guesses(),
                self.diagonal,
                pick=_pick_real_eigenvalues,
                tol=self.conv_tol,
                tol_residual=self.conv_tol_residual,
                max_cycle=self.max_cycle,
                max_space=self.max_space,
                nroots=self.nroots,
                callback=_callback,
                verbose=0,
            )

            eigvals = np.array(eigvals)
            eigvecs = np.array(eigvecs).T

        else:
            with util.catch_warnings(UserWarning):
                converged, eigvals, left, right = lib.linalg_helper.davidson_nosym1(
                    lambda vectors: [self.matvec(vector) for vector in vectors],
                    self.get_guesses(),
                    self.diagonal,
                    pick=_pick_real_eigenvalues,
                    tol=self.conv_tol,
                    tol_residual=self.conv_tol_residual,
                    max_cycle=self.max_cycle,
                    max_space=self.max_space,
                    nroots=self.nroots,
                    left=True,
                    callback=_callback,
                    verbose=0,
                )

            eigvals = np.array(eigvals)
            left = np.array(left).T
            right = np.array(right).T
            eigvecs = np.array([left, right])

        # TODO: How to print the final iteration?
        progress.stop()
        table.print()

        # Sort the eigenvalues
        mask = np.argsort(eigvals)
        eigvals = eigvals[mask]
        eigvecs = eigvecs[..., mask]
        converged = converged[mask]

        # Get the full map onto physical + auxiliary and rotate the eigenvectors
        vectors = util.null_space_basis(self.bra, ket=self.ket if not self.hermitian else None)
        if self.ket is None or self.hermitian:
            rotation = np.concatenate([self.bra, vectors[0]], axis=0)
            eigvecs = rotation @ eigvecs
        else:
            # Ensure biorthonormality of auxiliary vectors
            overlap = vectors[1].T.conj() @ vectors[0]
            overlap -= self.ket.T.conj() @ self.bra
            vectors = (
                vectors[0],
                vectors[1] @ util.matrix_power(overlap, -1, hermitian=False)[0],
            )
            rotation = (
                np.concatenate([self.bra, vectors[1]], axis=0),
                np.concatenate([self.ket, vectors[0]], axis=0),
            )
            eigvecs = np.array([rotation[0] @ eigvecs[0], rotation[1] @ eigvecs[1]])

        # Store the results
        self.result = Spectral(eigvals, eigvecs, self.nphys)
        self.converged = converged

        return self.result

    @property
    def matvec(self) -> Callable[[Array], Array]:
        """Get the matrix-vector operation for the self-energy supermatrix."""
        return self._matvec

    @property
    def diagonal(self) -> Array:
        """Get the diagonal of the self-energy supermatrix."""
        return self._diagonal

    @property
    def bra(self) -> Array:
        """Get the bra state vector mapping the supermatrix to the physical space."""
        return self._bra

    @property
    def ket(self) -> Array:
        """Get the ket state vector mapping the supermatrix to the physical space."""
        if self._ket is None:
            return self._bra
        return self._ket

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.bra.shape[0]
