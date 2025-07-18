"""Base class for grids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson.representations.enums import Component, Reduction
from dyson.typing import Array

if TYPE_CHECKING:
    from typing import Any

    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann


class BaseGrid(Array, ABC):
    """Base class for grids."""

    _weights: Array | None = None

    def __new__(cls, *args: Any, weights: Array | None = None, **kwargs: Any) -> BaseGrid:
        """Create a new instance of the grid.

        Args:
            args: Positional arguments for :class:`numpy.ndarray`.
            weights: Weights of the grid.
            kwargs: Keyword arguments for :class:`numpy.ndarray`.

        Returns:
            New instance of the grid.
        """
        obj = super().__new__(cls, *args, **kwargs).view(cls)
        obj._weights = weights
        return obj

    @abstractmethod
    def evaluate_lehmann(
        self,
        lehmann: Lehmann,
        reduction: Reduction = Reduction.NONE,
        component: Component = Component.FULL,
    ) -> Dynamic[Any]:
        """Evaluate a Lehmann representation on the grid.

        Args:
            lehmann: Lehmann representation to evaluate.
            reduction: The reduction of the dynamic representation.
            component: The component of the dynamic representation.

        Returns:
            Lehmann representation, realised on the grid.
        """
        pass

    @property
    def weights(self) -> Array:
        """Get the weights of the grid.

        Returns:
            Weights of the grid.
        """
        if self._weights is None:
            return np.ones_like(self) / self.size
        return self._weights

    @weights.setter
    def weights(self, value: Array) -> None:
        """Set the weights of the grid.

        Args:
            value: Weights of the grid.
        """
        self._weights = value

    @property
    def uniformly_spaced(self) -> bool:
        """Check if the grid is uniformly spaced.

        Returns:
            True if the grid is uniformly spaced, False otherwise.
        """
        if self.size < 2:
            raise ValueError("Grid is too small to compute separation.")
        return np.allclose(np.diff(self), self[1] - self[0])

    @property
    def uniformly_weighted(self) -> bool:
        """Check if the grid is uniformly weighted.

        Returns:
            True if the grid is uniformly weighted, False otherwise.
        """
        return np.allclose(self.weights, self.weights[0])

    @property
    def separation(self) -> float:
        """Get the separation of the grid.

        Returns:
            Separation of the grid.
        """
        if not self.uniformly_spaced:
            raise ValueError("Grid is not uniformly spaced.")
        return np.abs(self[1] - self[0])

    @property
    @abstractmethod
    def domain(self) -> str:
        """Get the domain of the grid."""
        pass

    @property
    @abstractmethod
    def reality(self) -> bool:
        """Get the reality of the grid."""
        pass

    def __array_finalize__(self, obj: Array | None, *args: Any, **kwargs: Any) -> None:
        """Finalize the array.

        Args:
            obj: Array to finalize.
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """
        if obj is None:
            return
        super().__array_finalize__(obj, *args, **kwargs)
        self._weights = getattr(obj, "_weights", None)

    @property
    def __array_priority__(self) -> float:
        """Get the array priority.

        Returns:
            Array priority.

        Notes:
            Grids have a lower priority than the default :class:`numpy.ndarray` priority. This is
            because most algebraic operations of a grid are to compute the Green's function or
            self-energy, which should not be of type :class:`BaseGrid`.
        """
        return -1
