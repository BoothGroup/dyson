"""Componentwise solver."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from dyson import numpy as np, util
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver
from dyson.solvers.static.exact import Exact

if TYPE_CHECKING:
    from typing import Any

    from dyson.typing import Array


class Componentwise(StaticSolver):
    """Wrapper for solvers of multiple components of the self-energy.

    Args:
        solvers: Solver for each component of the self-energy.

    Note:
        The resulting Green's function is orthonormalised such that the zeroth moment is identity.
        This may not be the desired behaviour in cases where your components do not span the full
        space.
    """

    def __init__(self, *solvers: StaticSolver, shared_static: bool = False):
        """Initialise the solver.

        Args:
            solvers: List of solvers for each component of the self-energy.
            shared_static: Whether the solvers share the same static part of the self-energy.
        """
        self._solvers = list(solvers)
        self._shared_static = shared_static
        self.hermitian = all(solver.hermitian for solver in solvers)

    @classmethod
    def from_self_energy(cls, static: Array, self_energy: Lehmann, **kwargs: Any) -> Componentwise:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            For the component-wise solver, this function separates the self-energy into occupied and
            virtual parts.
        """
        raise NotImplementedError(
            "Componentwise solver does not support self-energy decomposition. Intialise each "
            "solver from the self-energy directly and pass them to the constructor."
        )

    def get_static_self_energy(self, **kwargs: Any) -> Array:
        """get the static part of the self-energy.

        returns:
            static self-energy.
        """
        # Combine the static part of the self-energy
        static_parts = [solver.get_static_self_energy(**kwargs) for solver in self.solvers]
        static_equal = all(
            util.scaled_error(static, static_parts[0]) < 1e-10 for static in static_parts
        )
        if self.shared_static:
            if not static_equal:
                warnings.warn(
                    "shared_static is True, but the static parts of the self-energy do not appear "
                    "to be the same for each solver. This may lead to unexpected behaviour.",
                    UserWarning,
                    stacklevel=2,
                )
            static = static_parts[0]
        else:
            if static_equal:
                warnings.warn(
                    "shared_static is False, but the static parts of the self-energy appear to be "
                    "the same for each solver. Please ensure this is not double counting.",
                    UserWarning,
                    stacklevel=2,
                )
            static = sum(static_parts)
        return static

    def get_auxiliaries(self, **kwargs: Any) -> tuple[Array, Array]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Returns:
            Auxiliary energies and couplings.
        """
        # Combine the auxiliaries
        energies: Array = np.zeros((0))
        left: Array = np.zeros((self.nphys, 0))
        right: Array = np.zeros((self.nphys, 0))
        for solver in self.solvers:
            energies_i, couplings_i = solver.get_auxiliaries(**kwargs)
            energies = np.concatenate([energies, energies_i])
            if self.hermitian:
                left = np.concatenate([left, couplings_i], axis=1)
            else:
                left_i, right_i = util.unpack_vectors(couplings_i)
                left = np.concatenate([left, left_i], axis=1)
                right = np.concatenate([right, right_i], axis=1)
        couplings = np.array([left, right]) if not self.hermitian else left
        return energies, couplings

    #def get_eigenfunctions(self, **kwargs: Any) -> tuple[Array, Array]:
    #    """Get the eigenfunctions of the self-energy.

    #    Returns:
    #        Eigenvalues and eigenvectors.
    #    """
    #    # Get the eigenfunctions
    #    eigvals: Array = np.zeros((0))
    #    left: list[Array] = []
    #    right: list[Array] = []
    #    for solver in self.solvers:
    #        eigvals_i, eigvecs_i = solver.get_eigenfunctions(**kwargs)
    #        eigvals = np.concatenate([eigvals, eigvals_i])
    #        if self.hermitian:
    #            left.append(eigvecs_i)
    #        else:
    #            left_i, right_i = util.unpack_vectors(eigvecs_i)
    #            left.append(left_i)
    #            right.append(right_i)

    #    # Combine the eigenfunctions
    #    if self.hermitian:
    #        eigvecs = util.concatenate_paired_vectors(left, self.nphys)
    #    else:
    #        eigvecs = np.array(
    #            [
    #                util.concatenate_paired_vectors(left, self.nphys),
    #                util.concatenate_paired_vectors(right, self.nphys),
    #            ]
    #        )

    #    # Sort the eigenvalues and eigenvectors
    #    idx = np.argsort(eigvals)
    #    eigvals = eigvals[idx]
    #    eigvecs = eigvecs[..., idx]

    #    return eigvals, eigvecs

    def kernel(self) -> None:
        """Run the solver."""
        # TODO: We can combine the eigenvalues but can we project out the double counting that way?
        # Run the solvers
        for solver in self.solvers:
            solver.kernel()

        # Combine the self-energies
        static = self.get_static_self_energy()
        energies, couplings = self.get_auxiliaries()

        # Solve the self-energy
        exact = Exact.from_self_energy(static, Lehmann(energies, couplings))
        exact.kernel()
        self.eigvals, self.eigvecs = exact.get_eigenfunctions()

    @property
    def solvers(self) -> list[StaticSolver]:
        """Get the list of solvers."""
        return self._solvers

    @property
    def shared_static(self) -> bool:
        """Get the shared static flag."""
        return self._shared_static

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        if not len(set(solver.nphys for solver in self.solvers)) == 1:
            raise ValueError(
                "All solvers must have the same number of physical degrees of freedom."
            )
        return self.solvers[0].nphys
