"""Base class for Dyson equation solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rich import box
from rich.table import Table

from dyson import console, printing
from dyson.lehmann import Lehmann
from dyson.typing import Array

if TYPE_CHECKING:
    from typing import Any

    from dyson.expressions.expression import BaseExpression
    from dyson.spectral import Spectral


class BaseSolver(ABC):
    """Base class for Dyson equation solvers."""

    _options: set[str] = set()

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        """Initialise a subclass of :class:`BaseSolver`."""

        def wrap_init(init: Any) -> Any:
            """Wrapper to call __post_init__ after __init__."""

            def wrapped_init(self: BaseSolver, *args: Any, **kwargs: Any) -> None:
                init(self, *args, **kwargs)
                if init.__name__ == "__init__":
                    self.__log_init__()
                    self.__post_init__()

            return wrapped_init

        def wrap_kernel(kernel: Any) -> Any:
            """Wrapper to call __post_kernel__ after kernel."""

            def wrapped_kernel(self: BaseSolver, *args: Any, **kwargs: Any) -> Any:
                result = kernel(self, *args, **kwargs)
                if kernel.__name__ == "kernel":
                    self.__post_kernel__()
                return result

            return wrapped_kernel

        cls.__init__ = wrap_init(cls.__init__)
        cls.kernel = wrap_kernel(cls.kernel)

    def __log_init__(self) -> None:
        """Hook called after :meth:`__init__` for logging purposes."""
        printing.init_console()
        console.print("")

        # Print the solver name
        console.print(f"[method]{self.__class__.__name__}[/method]")

        # Print the options table
        table = Table(box=box.SIMPLE)
        table.add_column("Option")
        table.add_column("Value", style="input")
        for key in sorted(self._options):
            if not hasattr(self, key):
                raise ValueError(f"Option {key} not set in {self.__class__.__name__}")
            value = getattr(self, key)
            if hasattr(value, "__name__"):
                name = value.__name__
            else:
                name = str(value)
            table.add_row(key, name)
        console.print(table)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        pass

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        pass

    def set_options(self, **kwargs: Any) -> None:
        """Set options for the solver.

        Args:
            kwargs: Keyword arguments to set as options.
        """
        for key, val in kwargs.items():
            if key not in self._options:
                raise ValueError(f"Unknown option for {self.__class__.__name__}: {key}")
            setattr(self, key, val)

    @abstractmethod
    def kernel(self) -> Any:
        """Run the solver."""
        pass

    @classmethod
    @abstractmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> BaseSolver:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
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
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> BaseSolver:
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

    _options: set[str] = set()

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
