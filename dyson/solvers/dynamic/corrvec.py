"""Correction vector Green's function solver."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from scipy.sparse.linalg import LinearOperator, lgmres

from dyson import numpy as np
from dyson import util, console, printing
from dyson.grids.frequency import RealFrequencyGrid
from dyson.solvers.solver import DynamicSolver

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    from dyson.expressions.expression import BaseExpression
    from dyson.lehmann import Lehmann
    from dyson.typing import Array

#TODO: Can we use DIIS?


class CorrectionVector(DynamicSolver):
    """Correction vector Green's function solver.

    Args:
        matvec: The matrix-vector operation for the self-energy supermatrix.
        diagonal: The diagonal of the self-energy supermatrix.
        nphys: The number of physical degrees of freedom.
        grid: Real frequency grid upon which to evaluate the Green's function.
    """

    trace: bool = False
    include_real: bool = True
    conv_tol: float = 1e-8
    ordering: Literal["time-ordered", "advanced", "retarded"] = "time-ordered"
    _options: set[str] = {"trace", "include_real", "conv_tol", "ordering"}

    def __init__(  # noqa: D417
        self,
        matvec: Callable[[Array], Array],
        diagonal: Array,
        nphys: int,
        grid: RealFrequencyGrid,
        get_state_bra: Callable[[int], Array] | None = None,
        get_state_ket: Callable[[int], Array] | None = None,
        **kwargs: Any,
    ):
        r"""Initialise the solver.

        Args:
            matvec: The matrix-vector operation for the self-energy supermatrix.
            diagonal: The diagonal of the self-energy supermatrix.
            nphys: The number of physical degrees of freedom.
            grid: Real frequency grid upon which to evaluate the Green's function.
            get_state_bra: Function to get the bra vector corresponding to a fermion operator acting
                on the ground state. If `None`, the state vector is :math:`v_{i} = \delta_{ij}` for
                orbital :math:`j`.
            get_state_ket: Function to get the ket vector corresponding to a fermion operator acting
                on the ground state. If `None`, the :arg:`get_state_bra` function is used.
            trace: Whether to return only the trace.
            include_real: Whether to include the real part of the Green's function.
            conv_tol: Convergence tolerance for the solver.
            ordering: Time ordering of the resolvent.
        """
        self._matvec = matvec
        self._diagonal = diagonal
        self._nphys = nphys
        self._grid = grid
        self._get_state_bra = get_state_bra
        self._get_state_ket = get_state_ket
        self.set_options(**kwargs)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.diagonal.ndim != 1:
            raise ValueError("diagonal must be a 1D array.")
        if not callable(self.matvec):
            raise ValueError("matvec must be a callable function.")

        # Print the input information
        console.print(f"Matrix shape: [input]{(self.diagonal.size, self.diagonal.size)}[/input]")
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> CorrectionVector:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        if "grid" not in kwargs:
            raise ValueError("Missing required argument grid.")
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
            self_energy.nphys,
            kwargs.pop("grid"),
            bra.__getitem__,
            ket.__getitem__,
            **kwargs,
        )

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> CorrectionVector:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        if "grid" not in kwargs:
            raise ValueError("Missing required argument grid.")
        diagonal = expression.diagonal()
        matvec = expression.apply_hamiltonian
        return cls(
            matvec,
            diagonal,
            expression.nphys,
            kwargs.pop("grid"),
            expression.get_state_bra,
            expression.get_state_ket,
            **kwargs,
        )

    def matvec_dynamic(self, vector: Array, grid: RealFrequencyGrid) -> Array:
        r"""Perform the matrix-vector operation for the dynamic self-energy supermatrix.

        .. math::
            \mathbf{x}_\omega = \left(\omega - \mathbf{H} - i\eta\right) \mathbf{r}

        Args:
            vector: The vector to operate on.
            grid: The real frequency grid.

        Returns:
            The result of the matrix-vector operation.
        """
        resolvent = grid.resolvent(
            np.array(0.0), -self.diagonal, ordering=self.ordering, invert=False
        )
        result: Array = vector[None] * resolvent
        result -= self.matvec(vector.real)[None]
        if np.any(np.abs(vector.imag) > 1e-14):
            result -= self.matvec(vector.imag)[None] * 1.0j
        return result

    def matdiv_dynamic(self, vector: Array, grid: RealFrequencyGrid) -> Array:
        r"""Approximately perform a matrix-vector division for the dynamic self-energy supermatrix.

        .. math::
            \mathbf{x}_\omega = \frac{\mathbf{r}}{\omega - \mathbf{H} - i\eta}

        Args:
            vector: The vector to operate on.
            grid: The real frequency grid.

        Returns:
            The result of the matrix-vector division.

        Notes:
            The inversion is approximated using the diagonal of the matrix.
        """
        resolvent = grid.resolvent(self.diagonal, 0.0, ordering=self.ordering)
        result = vector[None] * resolvent[:, None]
        result[np.isinf(result)] = np.nan  # or 0?
        return result

    def get_state_bra(self, orbital: int) -> Array:
        """Get the bra vector corresponding to a fermion operator acting on the ground state.

        Args:
            orbital: Orbital index.

        Returns:
            Bra vector.
        """
        if self._get_state_bra is None:
            return np.eye(self.nphys, 1, k=orbital).ravel()
        return self._get_state_bra(orbital)

    def get_state_ket(self, orbital: int) -> Array:
        """Get the ket vector corresponding to a fermion operator acting on the ground state.

        Args:
            orbital: Orbital index.

        Returns:
            Ket vector.
        """
        if self._get_state_ket is None:
            return self.get_state_bra(orbital)
        return self._get_state_ket(orbital)

    def kernel(self) -> Array:
        """Run the solver.

        Returns:
            The Green's function on the real frequency grid.
        """
        # Get the printing helpers
        progress = printing.IterationsPrinter(self.nphys * self.grid.size, description="Frequency")
        progress.start()

        # Precompute bra vectors  # TODO: Optional
        bras = list(map(self.get_state_bra, range(self.nphys)))

        # Loop over ket vectors
        shape = (self.grid.size,) if self.trace else (self.grid.size, self.nphys, self.nphys)
        greens_function = np.zeros(shape, dtype=complex)
        failed: set[int] = set()
        for i in range(self.nphys):
            ket = self.get_state_ket(i)

            # Loop over frequencies
            x: Array | None = None
            outer_v: list[tuple[Array, Array]] = []
            for w in range(self.grid.size):
                progress.update(i * self.grid.size + w + 1)
                if w in failed:
                    continue

                shape = (self.diagonal.size, self.diagonal.size)
                matvec = LinearOperator(
                    shape, lambda v: self.matvec_dynamic(v, self.grid[[w]]), dtype=complex
                )
                matdiv = LinearOperator(
                    shape, lambda v: self.matdiv_dynamic(v, self.grid[[w]]), dtype=complex
                )

                # Solve the linear system
                x, info = lgmres(
                    matvec,
                    ket,
                    x0=x,
                    M=matdiv,
                    rtol=0.0,
                    atol=self.conv_tol,
                    outer_v=outer_v,
                )

                # Contract the Green's function
                if info != 0:
                    greens_function[w] = np.nan
                    failed.add(w)
                elif not self.trace:
                    for j in range(self.nphys):
                        greens_function[w, i, j] = bras[j] @ x
                else:
                    greens_function[w] += bras[i] @ x

        progress.stop()
        rating = printing.rate_error(len(failed) / self.grid.size, 1e-100, 1e-2)
        console.print("")
        console.print(
            f"Converged [output]{self.grid.size - len(failed)} of {self.grid.size}[/output] "
            f"frequencies ([{rating}]{len(failed) / self.grid.size:.2%}[/{rating}])."
        )

        return greens_function if self.include_real else greens_function.imag

    @property
    def matvec(self) -> Callable[[Array], Array]:
        """Get the matrix-vector operation for the self-energy supermatrix."""
        return self._matvec

    @property
    def diagonal(self) -> Array:
        """Get the diagonal of the self-energy supermatrix."""
        return self._diagonal

    @property
    def grid(self) -> RealFrequencyGrid:
        """Get the real frequency grid."""
        return self._grid

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys
