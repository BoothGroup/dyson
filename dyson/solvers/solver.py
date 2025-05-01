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

    #Couplings: TypeAlias = Array | tuple[Array, Array]

einsum = functools.partial(np.einsum, optimize=True)  # TODO: Move


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
            to instantiate the solver. In some cases, additional keyword arguments are required.
        """
        pass


class StaticSolver(BaseSolver):
    """Base class for static Dyson equation solvers."""

    hermitian: bool

    eigvals: Array | None = None
    eigvecs: Array | None = None

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
        eigvals, eigvecs = self.get_eigenfunctions(**kwargs)
        left, right = util.unpack_vectors(eigvecs)

        # Project back to the static part
        static = einsum("pk,qk,k->pq", right[:nphys], left[:nphys].conj(), eigvals)

        return static

    def get_auxiliaries(self, **kwargs: Any) -> tuple[Array, Array]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Returns:
            Auxiliary energies and couplings.
        """
        # FIXME: Is this generally true? Even if so, some solvers can do this more cheaply and
        # should implement this method.
        nphys = self.nphys
        eigvals, eigvecs = self.get_eigenfunctions(**kwargs)
        left, right = util.unpack_vectors(eigvecs)

        # Project back to the auxiliary subspace
        subspace = einsum("pk,qk,k->pq", right[nphys:], left[nphys:].conj(), eigvals)

        # Diagonalise the subspace to get the energies and basis for the couplings
        energies, rotation = util.eig_biorth(subspace, hermitian=self.hermitian)

        # Project back to the couplings
        couplings_right = einsum("pk,qk,k->pq", right[:nphys], left[nphys:].conj(), eigvals)
        if self.hermitian:
            couplings = couplings_right
        else:
            couplings_left = einsum("pk,qk,k->pq", right[nphys:], left[:nphys].conj(), eigvals)
            couplings_left = couplings_left.T.conj()
            couplings = np.array([couplings_left, couplings_right])

        # Rotate the couplings to the auxiliary basis
        if self.hermitian:
            couplings = couplings @ rotation[0]
        else:
            couplings = np.array([couplings_left @ rotation[0], couplings_right @ rotation[1]])

        return energies, couplings

    def get_eigenfunctions(self, **kwargs: Any) -> tuple[Array, Array]:
        """Get the eigenfunctions of the self-energy.

        Returns:
            Eigenvalues and eigenvectors.
        """
        if kwargs:
            raise TypeError(
                f"get_auxiliaries() got unexpected keyword argument {next(iter(kwargs))}"
            )
        if self.eigvals is None or self.eigvecs is None:
            raise ValueError("Must call kernel() to compute eigenvalues and eigenvectors.")
        return self.eigvals, self.eigvecs

    def get_dyson_orbitals(self, **kwargs: Any) -> tuple[Array, Array]:
        """Get the Dyson orbitals contributing to the Green's function.

        Returns:
            Dyson orbital energies and couplings.
        """
        eigvals, eigvecs = self.get_eigenfunctions(**kwargs)
        orbitals = eigvecs[..., : self.nphys, :]
        return eigvals, orbitals

    def get_self_energy(self, chempot: float | None = None, **kwargs: Any) -> Lehmann:
        """Get the Lehmann representation of the self-energy.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the self-energy.
        """
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_auxiliaries(**kwargs), chempot=chempot)

    def get_greens_function(self, chempot: float | None = None, **kwargs: Any) -> Lehmann:
        """Get the Lehmann representation of the Green's function.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the Green's function.
        """
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_dyson_orbitals(**kwargs), chempot=chempot)

    @property
    @abstractmethod
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
