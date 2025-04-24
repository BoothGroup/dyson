"""Base class for Dyson equation solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson.lehmann import Lehmann

if TYPE_CHECKING:
    from typing import Any, Callable, TypeAlias

    from dyson.typing import Array

    Couplings: TypeAlias = Array | tuple[Array, Array]


class BaseSolver(ABC):
    """Base class for Dyson equation solvers."""

    @abstractmethod
    def kernel(self) -> Any:
        """Run the solver."""
        pass


class StaticSolver(BaseSolver):
    """Base class for static Dyson equation solvers."""

    hermitian: bool

    eigvals: Array
    eigvecs: Couplings

    @abstractmethod
    def kernel(self) -> tuple[Lehmann, Lehmann]:
        """Run the solver.

        Returns:
            Lehmann representations for the self-energy and Green's function, connected by the Dyson
            equation.
        """
        pass

    @abstractmethod
    def get_auxiliaries(self, **kwargs: Any) -> tuple[Array, Couplings]:
        """Get the auxiliary energies and couplings contributing to the self-energy.

        Returns:
            Auxiliary energies and couplings.
        """
        pass

    def get_eigenfunctions(self, **kwargs: Any) -> tuple[Array, Couplings]:
        """Get the eigenfunctions of the self-energy.

        Returns:
            Eigenvalues and eigenvectors.
        """
        return self.eigvals, self.eigvecs

    def get_dyson_orbitals(self, **kwargs: Any) -> tuple[Array, Couplings]:
        """Get the Dyson orbitals contributing to the Green's function.

        Returns:
            Dyson orbital energies and couplings.
        """
        eigvals, eigvecs = self.get_eigenfunctions(**kwargs)
        if self.hermitian:
            if isinstance(eigvecs, tuple):
                raise ValueError("Hermitian solver should not get a tuple of eigenvectors.")
            eigvecs = eigvecs[: self.nphys]
        elif isinstance(eigvecs, tuple):
            eigvecs = (eigvecs[0][: self.nphys], eigvecs[1][: self.nphys])
        else:
            eigvecs = (eigvecs[: self.nphys], np.linalg.inv(eigvecs).T.conj()[: self.nphys])
        return eigvals, eigvecs

    def get_self_energy(self, chempot: float = 0.0, **kwargs: Any) -> Lehmann:
        """Get the Lehmann representation of the self-energy.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the self-energy.
        """
        return Lehmann(*self.get_auxiliaries(**kwargs), chempot=chempot)

    def get_green_function(self, chempot: float = 0.0, **kwargs: Any) -> Lehmann:
        """Get the Lehmann representation of the Green's function.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the Green's function.
        """
        return Lehmann(*self.get_dyson_orbitals(**kwargs), chempot=chempot)

    @abstractmethod
    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
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
