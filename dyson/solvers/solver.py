"""Base class for Dyson equation solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import TYPE_CHECKING, cast

from dyson import numpy as np, util
from dyson.lehmann import Lehmann
from dyson.typing import Array

if TYPE_CHECKING:
    from typing import Any, Callable, TypeAlias

    from dyson.spectral import Spectral
    from dyson.expression.expression import Expression


class BaseSolver(ABC):
    """Base class for Dyson equation solvers."""

    @abstractmethod
    def kernel(self) -> Any:
        """Run the solver."""
        pass

    @classmethod
    @abstractmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> BaseSolver:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            This method will extract the appropriate quantities or functions from the self-energy
            to instantiate the solver. In some cases, additional keyword arguments may required.
        """
        pass

    @classmethod
    @abstractmethod
    def from_expression(cls, expression: Expression, **kwargs: Any) -> BaseSolver:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            This method will extract the appropriate quantities or functions from the expression
            to instantiate the solver. In some cases, additional keyword arguments may required.
        """
        pass

    @property
    @abstractmethod
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        pass


class StaticSolver(BaseSolver):
    """Base class for static Dyson equation solvers."""

    hermitian: bool

    result: Spectral | None = None

    @abstractmethod
    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        pass


class DynamicSolver(BaseSolver):
    """Base class for dynamic Dyson equation solvers."""

    @abstractmethod
    def kernel(self) -> Array:
        """Run the solver.

        Returns:
            Dynamic Green's function resulting from the Dyson equation.
        """
        pass
