"""Enumerations for representations."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dyson.typing import Array


class Reduction(Enum):
    """Enumeration for the reduction of the dynamic representation."""

    NONE = auto()
    DIAG = auto()
    TRACE = auto()

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the array for this reduction."""
        return {Reduction.NONE: 2, Reduction.DIAG: 1, Reduction.TRACE: 0}[self]

    def raise_invalid_reduction(self) -> None:
        """Raise an error for invalid reduction."""
        raise ValueError(
            f"Invalid reduction: {self.name}. Valid reductions are: "
            f"{', '.join(r.name for r in Reduction)}"
        )


class Component(Enum):
    """Enumeration for the component of the dynamic representation."""

    FULL = auto()
    REAL = auto()
    IMAG = auto()

    @property
    def ncomp(self) -> int:
        """Get the number of components for this component type."""
        return 2 if self == Component.FULL else 1

    def raise_invalid_component(self) -> None:
        """Raise an error for invalid component."""
        raise ValueError(
            f"Invalid component: {self.name}. Valid components are: "
            f"{', '.join(c.name for c in Component)}"
        )
