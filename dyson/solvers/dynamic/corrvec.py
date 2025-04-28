"""Correction vector Green's function solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.sparse.linalg import LinearOperator, gcrotmk

from dyson import numpy as np
from dyson.solvers.solver import DynamicSolver

if TYPE_CHECKING:
    from typing import Any, Callable

    from dyson.typing import Array

# TODO: (m,k) for GCROTMK, more solvers, DIIS


class CorrectionVector(DynamicSolver):
    """Correction vector Green's function solver.

    Args:
        matvec: The matrix-vector operation for the self-energy supermatrix.
        diagonal: The diagonal of the self-energy supermatrix.
        nphys: The number of physical degrees of freedom.
        grid: Real frequency grid upon which to evaluate the Green's function.
    """

    def __init__(
        self,
        matvec: Callable[[Array], Array],
        diagonal: Array,
        nphys: int,
        grid: Array,
        get_state_bra: Callable[[int], Array] | None = None,
        get_state_ket: Callable[[int], Array] | None = None,
        eta: float = 1e-2,
        trace: bool = False,
        include_real: bool = True,
        conv_tol: float = 1e-8,
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
            eta: The broadening parameter.
            trace: Whether to return only the trace.
            include_real: Whether to include the real part of the Green's function.
            conv_tol: Convergence tolerance for the solver.
        """
        self._matvec = matvec
        self._diagonal = diagonal
        self._nphys = nphys
        self._grid = grid
        self._get_state_bra = get_state_bra
        self._get_state_ket = get_state_ket
        self.eta = eta
        self.trace = trace
        self.include_real = include_real
        self.conv_tol = conv_tol

    def matvec_dynamic(self, vector: Array, grid: Array) -> Array:
        r"""Perform the matrix-vector operation for the dynamic self-energy supermatrix.

        .. math::
            \mathbf{x}_\omega = \left(\omega - \mathbf{H} - i\eta\right) \mathbf{r}

        Args:
            vector: The vector to operate on.
            grid: The real frequency grid.

        Returns:
            The result of the matrix-vector operation.
        """
        result = (grid[:, None] - 1.0j * self.eta) * vector[None]
        result -= self.matvec(vector.real)[None]
        if np.any(np.abs(vector.imag) > 1e-14):
            result -= self.matvec(vector.imag)[None] * 1.0j
        return result

    def matdiv_dynamic(self, vector: Array, grid: Array) -> Array:
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
        result = vector[None] / (grid[:, None] - self.diagonal[None] - 1.0j * self.eta)
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
        # Precompute bra vectors  # TODO: Optional
        bras = list(map(self.get_state_bra, range(self.nphys)))

        # Loop over ket vectors
        greens_function = np.zeros(
            (self.grid.size,) if self.trace else (self.grid.size, self.nphys, self.nphys),
            dtype=complex,
        )
        for i in range(self.nphys):
            ket = self.get_state_ket(i)

            # Loop over frequencies
            x: Array | None = None
            for w in range(self.grid.size):
                shape = (self.diagonal.size, self.diagonal.size)
                matvec = LinearOperator(shape, lambda ω: self.matvec_dynamic(ket, ω), dtype=complex)
                matdiv = LinearOperator(shape, lambda ω: self.matdiv_dynamic(ket, ω), dtype=complex)
                if x is None:
                    x = matdiv @ ket
                x, info = gcrotmk(
                    matvec,
                    ket,
                    x0=x,
                    M=matdiv,
                    atol=0.0,
                    rtol=self.conv_tol,
                )

                if info != 0:
                    greens_function[w] = np.nan
                elif not self.trace:
                    for j in range(self.nphys):
                        greens_function[w, i, j] = bras[j] @ x
                else:
                    greens_function[w] += bras[i] @ x

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
    def grid(self) -> Array:
        """Get the real frequency grid."""
        return self._grid

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys
