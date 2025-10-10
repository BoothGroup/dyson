"""Container for a dynamic representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from dyson import numpy as np
from dyson import util
from dyson.grids.grid import BaseGrid
from dyson.representations.enums import Component, Reduction
from dyson.representations.representation import BaseRepresentation

if TYPE_CHECKING:
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array

_TGrid = TypeVar("_TGrid", bound=BaseGrid)


def _cast_reduction(first: Reduction, second: Reduction) -> Reduction:
    """Find the reduction that is compatible with both reductions."""
    values = {Reduction.NONE: 0, Reduction.DIAG: 1, Reduction.TRACE: 2}
    if values[first] <= values[second]:
        return first
    return second


def _cast_component(first: Component, second: Component) -> Component:
    """Find the component that is compatible with both components."""
    if first == second:
        return first
    return Component.FULL


def _cast_arrays(first: Dynamic[_TGrid], second: Dynamic[_TGrid]) -> tuple[Array, Array]:
    """Cast the arrays of two dynamic representations to the same component and reduction."""
    component = _cast_component(first.component, second.component)
    reduction = _cast_reduction(first.reduction, second.reduction)
    array_first = first.as_dynamic(component=component, reduction=reduction).array
    array_second = second.as_dynamic(component=component, reduction=reduction).array
    return array_first, array_second


def _same_grid(first: Dynamic[_TGrid], second: Dynamic[_TGrid]) -> bool:
    """Check if two dynamic representations have the same grid."""
    # TODO: Move to BaseGrid
    if not isinstance(second.grid, type(first.grid)):
        return False
    if len(first.grid) != len(second.grid):
        return False
    if not all(
        getattr(first.grid, attr) == getattr(second.grid, attr) for attr in first.grid._options
    ):
        return False
    if not np.allclose(first.grid.weights, second.grid.weights):
        return False
    return np.allclose(first.grid.points, second.grid.points)


class Dynamic(BaseRepresentation, Generic[_TGrid]):
    r"""Dynamic representation.

    The dynamic representation is a set of arrays in some physical space at each point in a time or
    frequency grid. This class contains the arrays and the grid information.
    """

    def __init__(
        self,
        grid: _TGrid,
        array: Array,
        reduction: Reduction = Reduction.NONE,
        component: Component = Component.FULL,
        hermitian: bool = False,
    ):
        """Initialise the object.

        Args:
            grid: The grid on which the dynamic representation is defined.
            array: The array of values at each point in the grid.
            reduction: The reduction of the dynamic representation.
            component: The component of the dynamic representation.
            hermitian: Whether the array is Hermitian.
        """
        self._grid = grid
        self._array = array
        self._hermitian = hermitian
        self._reduction = Reduction(reduction)
        self._component = Component(component)
        if array.shape[0] != len(grid):
            raise ValueError(
                f"Array must have the same size as the grid in the first dimension, but got "
                f"{array.shape[0]} for grid size {len(grid)}."
            )
        if (array.ndim - 1) != self.reduction.ndim:
            raise ValueError(
                f"Array must be {self.reduction.ndim}D for reduction {self.reduction}, but got "
                f"{array.ndim}D."
            )
        if int(np.iscomplexobj(array)) + 1 != self.component.ncomp:
            raise ValueError(
                f"Array must only be complex valued for component {Component.FULL}, but got "
                f"{array.dtype} for {self.component}."
            )

    @classmethod
    def from_lehmann(
        cls,
        lehmann: Lehmann,
        grid: _TGrid,
        reduction: Reduction = Reduction.NONE,
        component: Component = Component.FULL,
    ) -> Dynamic[_TGrid]:
        """Construct a dynamic representation from a Lehmann representation.

        Args:
            lehmann: The Lehmann representation to convert.
            grid: The grid on which the dynamic representation is defined.
            reduction: The reduction of the dynamic representation.
            component: The component of the dynamic representation.

        Returns:
            A dynamic representation.
        """
        return grid.evaluate_lehmann(lehmann, reduction=reduction, component=component)

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.array.shape[-1]

    @property
    def grid(self) -> _TGrid:
        """Get the grid on which the dynamic representation is defined."""
        return self._grid

    @property
    def array(self) -> Array:
        """Get the array of values at each point in the grid."""
        return self._array

    @property
    def reduction(self) -> Reduction:
        """Get the reduction of the dynamic representation."""
        return self._reduction

    @property
    def component(self) -> Component:
        """Get the component of the dynamic representation."""
        return self._component

    @property
    def hermitian(self) -> bool:
        """Get a boolean indicating if the system is Hermitian."""
        return self._hermitian or self.reduction != Reduction.NONE

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the array."""
        return self._array.dtype

    def __repr__(self) -> str:
        """Get a string representation of the dynamic representation."""
        return f"Dynamic(grid={self.grid}, shape={self.array.shape}, hermitian={self.hermitian})"

    def copy(
        self,
        deep: bool = True,
        reduction: Reduction | None = None,
        component: Component | None = None,
    ) -> Dynamic[_TGrid]:
        """Return a copy of the dynamic representation.

        Args:
            deep: Whether to return a deep copy of the energies and couplings.
            component: The component of the dynamic representation.
            reduction: The reduction of the dynamic representation.

        Returns:
            A new dynamic representation.
        """
        grid = self.grid
        array = self.array
        if reduction is None:
            reduction = self.reduction
        reduction = Reduction(reduction)
        if component is None:
            component = self.component
        component = Component(component)

        # Copy the array if requested
        if deep:
            array = array.copy()

        # Adjust the reduction if necessary
        if reduction != self.reduction:
            if (self.reduction, reduction) == (Reduction.NONE, Reduction.DIAG):
                array = np.diagonal(array, axis1=1, axis2=2)
            elif (self.reduction, reduction) == (Reduction.NONE, Reduction.TRACE):
                array = np.trace(array, axis1=1, axis2=2)
            elif (self.reduction, reduction) == (Reduction.DIAG, Reduction.TRACE):
                array = np.sum(array, axis=1)
            elif (self.reduction, reduction) == (Reduction.DIAG, Reduction.NONE):
                array_new = np.zeros((len(grid), self.nphys, self.nphys), dtype=array.dtype)
                np.fill_diagonal(array_new, array)
                array = array_new
            else:
                raise ValueError(
                    f"Cannot convert from {self.reduction} to {reduction} for dynamic "
                    "representation."
                )

        # Adjust the component if necessary
        if component != self.component:
            if (self.component, component) == (Component.FULL, Component.REAL):
                array = np.real(array)
            elif (self.component, component) == (Component.FULL, Component.IMAG):
                array = np.imag(array)
            elif (self.component, component) == (Component.REAL, Component.FULL):
                array = array + 1.0j * np.zeros_like(array)
            elif (self.component, component) == (Component.IMAG, Component.FULL):
                array = np.zeros_like(array) + 1.0j * array
            else:
                raise ValueError(
                    f"Cannot convert from {self.component} to {component} for dynamic "
                    "representation."
                )

        return self.__class__(
            grid, array, hermitian=self.hermitian, reduction=reduction, component=component
        )

    def as_dynamic(
        self, component: Component | None = None, reduction: Reduction | None = None
    ) -> Dynamic[_TGrid]:
        """Return the dynamic representation with the specified component and reduction.

        Args:
            component: The component of the dynamic representation.
            reduction: The reduction of the dynamic representation.

        Returns:
            A new dynamic representation with the specified component and reduction.
        """
        return self.copy(deep=False, component=component, reduction=reduction)

    def rotate(self, rotation: Array | tuple[Array, Array]) -> Dynamic[_TGrid]:
        """Rotate the dynamic representation.

        Args:
            rotation: The rotation matrix to apply to the array. If the matrix has three dimensions,
                the first dimension is used to rotate on the left, and the second dimension is used
                to rotate on the right.

        Returns:
            A new dynamic representation with the rotated array.
        """
        left, right = rotation if isinstance(rotation, tuple) else (rotation, rotation)

        if np.iscomplexobj(left) or np.iscomplexobj(right):
            array = self.as_dynamic(component=Component.FULL).array
            component = Component.FULL
        else:
            array = self.array
            component = self.component

        if self.reduction == Reduction.NONE:
            array = util.einsum("wpq,pi,qj->wij", array, left.conj(), right)
        elif self.reduction == Reduction.DIAG:
            array = util.einsum("wp,pi,pj->wij", array, left.conj(), right)
        elif self.reduction == Reduction.TRACE:
            raise ValueError("Cannot rotate a dynamic representation with trace reduction.")

        return self.__class__(
            self.grid,
            array,
            component=component,
            reduction=Reduction.NONE,
            hermitian=self.hermitian,
        )

    def __add__(self, other: Dynamic[_TGrid]) -> Dynamic[_TGrid]:
        """Add two dynamic representations."""
        if not isinstance(other, Dynamic):
            return NotImplemented
        if not _same_grid(self, other):
            raise ValueError("Cannot add dynamic representations with different grids.")
        return self.__class__(
            self.grid,
            np.add(*_cast_arrays(self, other)),
            component=_cast_component(self.component, other.component),
            reduction=_cast_reduction(self.reduction, other.reduction),
            hermitian=self.hermitian or other.hermitian,
        )

    def __sub__(self, other: Dynamic[_TGrid]) -> Dynamic[_TGrid]:
        """Subtract two dynamic representations."""
        if not isinstance(other, Dynamic):
            return NotImplemented
        if not _same_grid(self, other):
            raise ValueError("Cannot subtract dynamic representations with different grids.")
        return self.__class__(
            self.grid,
            np.subtract(*_cast_arrays(self, other)),
            component=_cast_component(self.component, other.component),
            reduction=_cast_reduction(self.reduction, other.reduction),
            hermitian=self.hermitian or other.hermitian,
        )

    def __mul__(self, other: float | int) -> Dynamic[_TGrid]:
        """Multiply the dynamic representation by a scalar."""
        if not isinstance(other, (float, int)):
            return NotImplemented
        return self.__class__(
            self.grid,
            self.array * other,
            component=self.component,
            reduction=self.reduction,
            hermitian=self.hermitian,
        )

    __rmul__ = __mul__

    def __neg__(self) -> Dynamic[_TGrid]:
        """Negate the dynamic representation."""
        return -1 * self

    def __array__(self) -> Array:
        """Return the dynamic representation as a NumPy array."""
        return self.array

    def __eq__(self, other: object) -> bool:
        """Check if two dynamic representations are equal."""
        if not isinstance(other, Dynamic):
            return NotImplemented
        if other.nphys != self.nphys:
            return False
        if len(other.grid) != len(self.grid):
            return False
        if other.hermitian != self.hermitian:
            return False
        if not _same_grid(self, other):
            return False
        return np.allclose(other.array, self.array)

    def __hash__(self) -> int:
        """Return a hash of the dynamic representation."""
        return hash(
            (
                tuple(self.grid.points),
                tuple(self.grid.weights),
                tuple(self.array.ravel()),
                self.hermitian,
            )
        )
