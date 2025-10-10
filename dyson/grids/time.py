"""Time grids."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from math import factorial

from dyson import numpy as np
from dyson import util
from dyson.grids.grid import BaseGrid
from dyson.representations.enums import Component, Ordering, Reduction

if TYPE_CHECKING:
    from typing import Any, Iterable

    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array


class BaseTimeGrid(BaseGrid):
    """Base class for time grids."""

    def evaluate_lehmann(
        self,
        lehmann: Lehmann,
        reduction: Reduction = Reduction.NONE,
        component: Component = Component.FULL,
        **kwargs: Any,
    ) -> Dynamic[BaseTimeGrid]:
        r"""Evaluate a Lehmann representation on the grid.

        Args:
            lehmann: Lehmann representation to evaluate.
            reduction: The reduction of the dynamic representation.
            component: The component of the dynamic representation.
            kwargs: Additional keyword arguments for the resolvent.

        Returns:
            Lehmann representation, realised on the grid.
        """
        raise NotImplementedError

    @property
    def domain(self) -> str:
        """Return the domain of the grid."""
        return "time"


class RealTimeGrid(BaseTimeGrid):
    """Real time grid."""

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

    def evaluate_tail(
        self,
        moments: Iterable[Array],
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Evaluate the tail (short time) on the grid, via a moment expansion.

        Args:
            moments: Moments of the tail expansion.

        Returns:
            Values of the tail expansion on the grid.
        """
        raise NotImplementedError

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return True

    @classmethod
    def from_uniform(cls, start: float, stop: float, num: int) -> RealTimeGrid:
        """Create a uniform real time grid.

        Args:
            start: Start of the grid.
            stop: End of the grid.
            num: Number of points in the grid.

        Returns:
            Uniform real time grid.
        """
        points = np.linspace(start, stop, num, endpoint=True)
        return cls(points)


GridRT = RealTimeGrid


class ImaginaryTimeGrid(BaseTimeGrid):
    """Imaginary time grid."""

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

    def evaluate_tail(
        self,
        moments: Iterable[Array],
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Evaluate the tail (short time) on the grid, via a moment expansion.

        Args:
            moments: Moments of the tail expansion.

        Returns:
            Values of the tail expansion on the grid.
        """
        tail: Array = 0.0
        for i, moment in enumerate(moments):
            coefficient = (-1) ** i / factorial(i + 1)
            x = self.points ** (i + 1) - self.beta ** i * self.points
            tail -= util.einsum("...,w->w...", moment, coefficient * x)
        return tail

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return False

    @property
    def beta(self) -> float:
        """Get the inverse temperature of the grid.

        Returns:
            Inverse temperature of the grid.
        """
        return self.points[-1] - self.points[0]

    @classmethod
    def from_uniform(cls, num: int, beta: float) -> RealTimeGrid:
        """Create a uniform real time grid.

        Args:
            num: Number of points in the grid.
            beta: Inverse temperature.

        Returns:
            Uniform real time grid.
        """
        points = np.linspace(0, beta, num, endpoint=True)
        return cls(points)


GridIT = ImaginaryTimeGrid
