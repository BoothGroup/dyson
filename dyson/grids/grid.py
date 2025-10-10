"""Base class for grids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson.representations.enums import Component, Reduction, RepresentationEnum, Ordering

if TYPE_CHECKING:
    from typing import Any, Iterable

    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array


class BaseGrid(ABC):
    """Base class for grids."""

    _options: set[str] = set()

    _points: Array
    _weights: Array | None = None

    def __init__(  # noqa: D417
        self, points: Array, weights: Array | None = None, **kwargs: Any
    ) -> None:
        """Initialise the grid.

        Args:
            points: Points of the grid.
            weights: Weights of the grid.
        """
        self._points = np.asarray(points)
        self._weights = np.asarray(weights) if weights is not None else None
        self.set_options(**kwargs)

    def set_options(self, **kwargs: Any) -> None:
        """Set options for the solver.

        Args:
            kwargs: Keyword arguments to set as options.
        """
        for key, val in kwargs.items():
            if key not in self._options:
                raise ValueError(f"Unknown option for {self.__class__.__name__}: {key}")
            if isinstance(getattr(self, key), RepresentationEnum):
                # Casts string to the appropriate enum type if the default value is an enum
                val = getattr(self, key).__class__(val)
            setattr(self, key, val)

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

    @abstractmethod
    def evaluate_tail(
        self,
        moments: Iterable[Array],
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Evaluate the tail on the grid, via a moment expansion.

        Args:
            moments: Moments of the tail expansion.

        Returns:
            Values of the tail expansion on the grid.
        """
        pass

    @property
    def points(self) -> Array:
        """Get the points of the grid.

        Returns:
            Points of the grid.
        """
        return self._points

    @property
    def weights(self) -> Array:
        """Get the weights of the grid.

        Returns:
            Weights of the grid.
        """
        if self._weights is None:
            return np.ones_like(self.points) / len(self)
        return self._weights

    def __getitem__(self, key: int | slice | list[int] | Array) -> BaseGrid:
        """Get a subset of the grid.

        Args:
            key: Index or slice to get.

        Returns:
            Subset of the grid.
        """
        points = self.points[key]
        weights = self.weights[key] if self._weights is not None else None
        kwargs = {opt: getattr(self, opt) for opt in self._options}
        return self.__class__(points, weights=weights, **kwargs)

    def __len__(self) -> int:
        """Get the size of the grid.

        Returns:
            Size of the grid.
        """
        return self.points.shape[0]

    @property
    def uniformly_spaced(self) -> bool:
        """Check if the grid is uniformly spaced.

        Returns:
            True if the grid is uniformly spaced, False otherwise.
        """
        if len(self) < 2:
            raise ValueError("Grid is too small to compute separation.")
        return np.allclose(np.diff(self.points), self.points[1] - self.points[0])

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
        return np.abs(self.points[1] - self.points[0])

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
