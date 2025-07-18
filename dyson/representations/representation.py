"""Base class for representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class BaseRepresentation(ABC):
    """Base class for representations."""

    @property
    @abstractmethod
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        pass

    @property
    @abstractmethod
    def hermitian(self) -> bool:
        """Get a boolean indicating if the system is Hermitian."""
        pass
