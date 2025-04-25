"""Base class for Dyson equation solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import TYPE_CHECKING, cast

from dyson import numpy as np
from dyson.lehmann import Lehmann
from dyson.typing import Array

if TYPE_CHECKING:
    from typing import Any, Callable, TypeAlias

    Couplings: TypeAlias = Array | tuple[Array, Array]

einsum = functools.partial(np.einsum, optimize=True)  # TODO: Move


class BaseSolver(ABC):
    """Base class for Dyson equation solvers."""

    @abstractmethod
    def kernel(self) -> Any:
        """Run the solver."""
        pass

    @abstractmethod
    @classmethod
    def from_self_energy(self, static: Array, self_energy: Lehmann, **kwargs: Any) -> BaseSolver:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            This method will extract the appropriate quantities or functions from the self-energy
            to instantiate the solver. In some cases, additional keyword arguments are required.
        """
        pass


class StaticSolver(BaseSolver):
    """Base class for static Dyson equation solvers."""

    hermitian: bool

    eigvals: Array | None = None
    eigvecs: Couplings | None = None

    @abstractmethod
    def kernel(self) -> None:
        """Run the solver."""
        pass

    def get_static_self_energy(self, **kwargs: Any) -> Array:
        """Get the static part of the self-energy.

        Returns:
            Static self-energy.
        """
        # FIXME: Is this generally true? Even if so, some solvers can do this more cheaply and
        # should implement this method.
        nphys = self.nphys
        eigvals, (left, right) = self.get_eigenfunctions(unpack=True, **kwargs)

        # Project back to the static part
        static = einsum("pk,qk,k->pq", left[: nphys], right[: nphys].conj(), eigvals)

        return static

    def get_auxiliaries(self, **kwargs: Any) -> tuple[Array, Couplings]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Returns:
            Auxiliary energies and couplings.
        """
        # FIXME: Is this generally true? Even if so, some solvers can do this more cheaply and
        # should implement this method.
        nphys = self.nphys
        eigvals, (left, right) = self.get_eigenfunctions(unpack=True, **kwargs)

        # Project back to the auxiliary subspace
        energies = einsum("pk,qk,k->pq", left[nphys :], right[nphys :].conj(), eigvals)

        # Diagonalise the subspace to get the energies and basis for the couplings
        if self.hermitian:
            energies, rotation = np.linalg.eigh(energies)
        else:
            energies, rotation = np.linalg.eig(energies)

        # Project back to the couplings
        couplings_left = einsum("pk,qk,k->pq", left[: nphys], right[nphys :].conj(), eigvals)
        if self.hermitian:
            couplings = couplings_left
        else:
            couplings_right = einsum("pk,qk,k->pq", left[nphys :], right[: nphys].conj(), eigvals)
            couplings_right = couplings_right.T.conj()
            couplings = (couplings_left, couplings_right)

        # Rotate the couplings to the auxiliary basis
        if self.hermitian:
            couplings = rotation.T.conj() @ couplings
        else:
            couplings = (
                rotation.T.conj() @ couplings_left,
                rotation.T.conj() @ couplings_right,
            )

        return energies, couplings

    def get_eigenfunctions(self, unpack: bool = False, **kwargs: Any) -> tuple[Array, Couplings]:
        """Get the eigenfunctions of the self-energy.

        Args:
            unpack: Whether to unpack the eigenvectors into left and right components, regardless
                of the hermitian property.

        Returns:
            Eigenvalues and eigenvectors.
        """
        if self.eigvals is None or self.eigvecs is None:
            raise ValueError("Must call kernel() to compute eigenvalues and eigenvectors.")
        if unpack:
            if self.hermitian:
                if isinstance(self.eigvecs, tuple):
                    raise ValueError("Hermitian solver should not get a tuple of eigenvectors.")
                return self.eigvals, (self.eigvecs, self.eigvecs)
            elif isinstance(self.eigvecs, tuple):
                return self.eigvals, self.eigvecs
            else:
                return self.eigvals, (self.eigvecs, np.linalg.inv(self.eigvecs).T.conj())
        return self.eigvals, self.eigvecs

    def get_dyson_orbitals(self, **kwargs: Any) -> tuple[Array, Couplings]:
        """Get the Dyson orbitals contributing to the Green's function.

        Returns:
            Dyson orbital energies and couplings.
        """
        eigvals, eigvecs = self.get_eigenfunctions(unpack=False, **kwargs)
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
