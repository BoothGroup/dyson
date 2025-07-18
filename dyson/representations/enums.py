"""Enumerations for representations."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dyson.typing import Array


class RepresentationEnum(Enum):
    """Base enumeration for representations."""

    def raise_invalid_representation(self) -> None:
        """Raise an error for invalid representation."""
        name = self.__class__.__name__.lower()
        valid = [r.name for r in self.__class__]
        raise ValueError(f"Invalid {name}: {self.name}. Valid {name}s are: {', '.join(valid)}")


class Reduction(RepresentationEnum):
    """Enumeration for the reduction of the dynamic representation."""

    NONE = "none"
    DIAG = "diag"
    TRACE = "trace"

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the array for this reduction."""
        return {Reduction.NONE: 2, Reduction.DIAG: 1, Reduction.TRACE: 0}[self]


class Component(RepresentationEnum):
    """Enumeration for the component of the dynamic representation."""

    FULL = "full"
    REAL = "real"
    IMAG = "imag"

    @property
    def ncomp(self) -> int:
        """Get the number of components for this component type."""
        return 2 if self == Component.FULL else 1


class Ordering(RepresentationEnum):
    """Enumeration for the time ordering of the dynamic representation."""

    ORDERED = "ordered"
    ADVANCED = "advanced"
    RETARDED = "retarded"
