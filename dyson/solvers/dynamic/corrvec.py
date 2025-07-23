"""Correction vector Green's function solver [nocera2016]_.

.. [nocera2016] Nocera, A., & Alvarez, G. (2016). Spectral functions with the density matrix
   renormalization group: Krylov-space approach for correction vectors. Physical Review. E, 94(5).
   https://doi.org/10.1103/physreve.94.053308
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from scipy.sparse.linalg import LinearOperator, lgmres

from dyson import console, printing, util
from dyson import numpy as np
from dyson.grids.frequency import RealFrequencyGrid
from dyson.representations.dynamic import Dynamic
from dyson.representations.enums import Component, Ordering, Reduction
from dyson.solvers.solver import DynamicSolver

if TYPE_CHECKING:
    from typing import Any, Callable

    from dyson.expressions.expression import BaseExpression
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array

# TODO: Can we use DIIS?


class CorrectionVector(DynamicSolver):
    """Correction vector Green's function solver.

    Args:
        matvec: The matrix-vector operation for the self-energy supermatrix.
        diagonal: The diagonal of the self-energy supermatrix.
        nphys: The number of physical degrees of freedom.
        grid: Real frequency grid upon which to evaluate the Green's function.
    """

    reduction: Reduction = Reduction.NONE
    component: Component = Component.FULL
    ordering: Ordering = Ordering.ORDERED
    conv_tol: float = 1e-8
    _options: set[str] = {"reduction", "component", "ordering", "conv_tol"}

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
            component: The component of the dynamic representation to solve for.
            reduction: The reduction of the dynamic representation to solve for.
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
            expression.get_excitation_bra,
            expression.get_excitation_ket,
            **kwargs,
        )

    def matvec_dynamic(self, vector: Array, grid_index: int) -> Array:
        r"""Perform the matrix-vector operation for the dynamic self-energy supermatrix.

        .. math::
            \mathbf{x}_\omega = \left(\omega - \mathbf{H} - i\eta\right) \mathbf{r}

        Args:
            vector: The vector to operate on.
            grid_index: Index of the real frequency grid.

        Returns:
            The result of the matrix-vector operation.
        """
        grid = cast(RealFrequencyGrid, self.grid[[grid_index]])
        resolvent = grid.resolvent(
            np.array(0.0), -self.diagonal, ordering=self.ordering, invert=False
        )
        result: Array = vector[None] * resolvent
        result -= self.matvec(vector.real)[None]
        if np.any(np.abs(vector.imag) > 1e-14):
            result -= self.matvec(vector.imag)[None] * 1.0j
        return result

    def matdiv_dynamic(self, vector: Array, grid_index: int) -> Array:
        r"""Approximately perform a matrix-vector division for the dynamic self-energy supermatrix.

        .. math::
            \mathbf{x}_\omega = \frac{\mathbf{r}}{\omega - \mathbf{H} - i\eta}

        Args:
            vector: The vector to operate on.
            grid_index: Index of the real frequency grid.

        Returns:
            The result of the matrix-vector division.

        Notes:
            The inversion is approximated using the diagonal of the matrix.
        """
        grid = cast(RealFrequencyGrid, self.grid[[grid_index]])
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

    def kernel(self) -> Dynamic[RealFrequencyGrid]:
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
        shape = (self.grid.size,) + (self.nphys,) * self.reduction.ndim
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

                linop_shape = (self.diagonal.size, self.diagonal.size)
                matvec = LinearOperator(
                    linop_shape, lambda v: self.matvec_dynamic(v, w), dtype=complex
                )
                matdiv = LinearOperator(
                    linop_shape, lambda v: self.matdiv_dynamic(v, w), dtype=complex
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
                elif self.reduction == Reduction.NONE:
                    for j in range(self.nphys):
                        greens_function[w, i, j] = bras[j] @ x
                elif self.reduction == Reduction.DIAG:
                    greens_function[w, i] = bras[i] @ x
                elif self.reduction == Reduction.TRACE:
                    greens_function[w] += bras[i] @ x
                else:
                    self.reduction.raise_invalid_representation()

        # Post-process the Green's function component
        # TODO: Can we do this earlier to avoid computing unnecessary components?
        if self.component == Component.REAL:
            greens_function = greens_function.real
        elif self.component == Component.IMAG:
            greens_function = greens_function.imag

        progress.stop()
        rating = printing.rate_error(len(failed) / self.grid.size, 1e-100, 1e-2)
        console.print("")
        console.print(
            f"Converged [output]{self.grid.size - len(failed)} of {self.grid.size}[/output] "
            f"frequencies ([{rating}]{1 - len(failed) / self.grid.size:.2%}[/{rating}])."
        )

        return Dynamic(
            self.grid,
            greens_function,
            reduction=self.reduction,
            component=self.component,
            hermitian=self.get_state_ket is None,
        )

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
