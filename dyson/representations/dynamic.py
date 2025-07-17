"""Container for a dynamic representation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import util

if TYPE_CHECKING:
    from dyson.grids.grid import BaseGrid
    from dyson.typing import Array
    from dyson.representations.lehmann import Lehmann


class Dynamic:
    r"""Dynamic representation.

    The dynamic representation is a set of arrays in some physical space at each point in a time or
    frequency grid. This class contains the arrays and the grid information.
    """

    def __init__(self, grid: BaseGrid, array: Array, hermitian: bool = False):
        """Initialise the object.

        Args:
            grid: The grid on which the dynamic representation is defined.
            array: The array of values at each point in the grid.
            hermitian: Whether the array is Hermitian.
        """
        self._grid = grid
        self._array = array
        self._hermitian = hermitian
        if array.shape[0] != grid.size:
            raise ValueError(
                f"Array must have the same size as the grid in the first dimension, but got "
                f"{array.shape[0]} for grid size {grid.size}."
            )
        if array.ndim not in {1, 2, 3}:
            raise ValueError(f"Array must be 1D, 2D, or 3D, but got {array.ndim}D.")

    def from_lehmann(cls, lehmann: Lehmann, grid: BaseGrid, trace: bool = False) -> Dynamic:
        """Construct a dynamic representation from a Lehmann representation.

        Args:
            lehmann: The Lehmann representation to convert.
            grid: The grid on which the dynamic representation is defined.
            trace: If True, return the trace of the dynamic representation.

        Returns:
            A dynamic representation.
        """
        return grid.evaluate_lehmann(lehmann, trace=trace)

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.array.shape[-1]

    @property
    def grid(self) -> BaseGrid:
        """Get the grid on which the dynamic representation is defined."""
        return self._grid

    @property
    def array(self) -> Array:
        """Get the array of values at each point in the grid."""
        return self._array

    @property
    def hermitian(self) -> bool:
        """Get a flag indicating whether the array is Hermitian."""
        return self._hermitian or not self.full

    @property
    def full(self) -> bool:
        """Get a flag indicating whether the dynamic representation is full."""
        return self.array.ndim == 3

    @property
    def diagonal(self) -> bool:
        """Get a flag indicating whether the dynamic representation is diagonal."""
        return self.array.ndim == 2

    @property
    def traced(self) -> bool:
        """Get a flag indicating whether the dynamic representation is traced."""
        return self.array.ndim == 1

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the array."""
        return self._array.dtype

    def __repr__(self) -> str:
        """Get a string representation of the dynamic representation."""
        return (
            f"Dynamic(grid={self.grid}, shape={self.array.shape}, hermitian={self.hermitian})"
        )

    def copy(self, deep: bool = True) -> Dynamic:
        """Return a copy of the dynamic representation.

        Args:
            deep: Whether to return a deep copy of the energies and couplings.

        Returns:
            A new dynamic representation.
        """
        grid = self.grid
        array = self.array

        # Copy the array if requested
        if deep:
            array = array.copy()

        return self.__class__(grid, array, hermitian=self.hermitian)

    def as_full(self) -> Dynamic:
        """Return the dynamic representation as a full representation.

        Returns:
            A new dynamic representation with the full array.
        """
        if self.full:
            array = self.array
        elif self.diagonal:
            array = np.zeros((self.grid.size, self.nphys, self.nphys), dtype=self.array.dtype)
            np.fill_diagonal(array, self.array)
        elif self.traced:
            raise ValueError(
                "Cannot convert a traced dynamic representation to a full representation."
            )
        return self.__class__(self.grid, array, hermitian=self.hermitian)

    def as_diagonal(self) -> Dynamic:
        """Return the dynamic representation as a diagonal representation.

        Returns:
            A new dynamic representation with the diagonal of the array.
        """
        if self.full:
            array = np.diagonal(self.array, axis1=1, axis2=2)
        elif self.diagonal:
            array = self.array
        else:
            raise ValueError(
                "Cannot convert a traced dynamic representation to a diagonal representation."
            )
        return self.__class__(self.grid, array, hermitian=self.hermitian)

    def as_trace(self) -> Dynamic:
        """Return the trace of the dynamic representation.

        Returns:
            A new dynamic representation with the trace of the array.
        """
        if self.full:
            array = np.trace(self.array, axis1=1, axis2=2)
        elif self.diagonal:
            array = np.sum(self.array, axis=1)
        else:
            array = self.array
        return self.__class__(self.grid, array, hermitian=self.hermitian)

    def rotate(self, rotation: Array | tuple[Array, Array]) -> Dynamic:
        """Rotate the dynamic representation.

        Args:
            rotation: The rotation matrix to apply to the array. If the matrix has three dimensions,
                the first dimension is used to rotate on the left, and the second dimension is used
                to rotate on the right.

        Returns:
            A new dynamic representation with the rotated array.
        """
        left, right = rotation if isinstance(rotation, tuple) else (rotation, rotation)
        if self.traced:
            array = util.einsum("wp,pi,pj->wij", self.array, left.conj(), right)
        else:
            array = util.einsum("wpq,pi,qj->wij", self.array, left.conj(), right)
        return self.__class__(self.grid, array, hermitian=self.hermitian)

    def __eq__(self, other: object) -> bool:
        """Check if two dynamic representations are equal."""
        if not isinstance(other, Dynamic):
            return NotImplemented
        if other.nphys != self.nphys:
            return False
        if other.grid.size != self.grid.size:
            return False
        if other.hermitian != self.hermitian:
            return False
        return np.allclose(other.grid, self.grid) and (
            np.allclose(other.grid.weights, self.grid.weights)
            and np.allclose(other.array, self.array)
        )

    def __hash__(self) -> int:
        """Return a hash of the dynamic representation."""
        return hash(
            (tuple(self.grid), tuple(self.grid.weights), tuple(self.array.ravel()), self.hermitian)
        )
