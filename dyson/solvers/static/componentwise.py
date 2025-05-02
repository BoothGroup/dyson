"""Componentwise solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np, util
from dyson.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver

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

    def __init__(self, *solvers: StaticSolver):
        """Initialise the solver.

        Args:
            solvers: List of solvers for each component of the self-energy.
        """
        self._solvers = list(solvers)
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

    def kernel(self) -> None:
        """Run the solver."""
        # Run the solvers
        for solver in self.solvers:
            solver.kernel()

        # Get the eigenvalues and eigenvectors
        eigvals_list = []
        left_list = []
        right_list = []
        for solver in self.solvers:
            eigvals, eigvecs = solver.get_eigenfunctions()
            eigvals_list.append(eigvals)
            left, right = util.unpack_vectors(eigvecs)
            left_list.append(left)
            right_list.append(right)

        # Combine the eigenvalues and eigenvectors
        eigvals = np.concatenate(eigvals_list)
        left = util.concatenate_paired_vectors(left_list, self.nphys)
        if not self.hermitian:
            right = util.concatenate_paired_vectors(right_list, self.nphys)

        # Biorthogonalise the eigenvectors
        # FIXME: Can we make this work to properly recover non-Hermitian?
        #if self.hermitian:
        #    #left = np.concatenate(
        #    #    [
        #    #        util.orthonormalise(left[:self.nphys], transpose=True),
        #    #        util.orthonormalise(left[self.nphys:], transpose=True),
        #    #    ],
        #    #    axis=0,
        #    #)
        #    left_p, left_a = left[:self.nphys], left[self.nphys:]
        #    left_p = util.orthonormalise(left_p)
        #    left = np.concatenate([left_p, left_a], axis=0)
        #else:
        #    #left_p, right_p = left[:self.nphys], right[:self.nphys]
        #    left_a, right_a = left[self.nphys:], right[self.nphys:]
        #    left_p, right_p = util.biorthonormalise(left[:self.nphys], right[:self.nphys])
        #    #left_a, right_a = util.biorthonormalise(left[self.nphys:], right[self.nphys:])
        #    left = np.concatenate([left_p, left_a], axis=0)
        #    right = np.concatenate([right_p, right_a], axis=0)

        # Store the eigenvalues and eigenvectors
        self.eigvals = eigvals
        self.eigvecs = left if self.hermitian else np.array([left, right])

    @property
    def solvers(self) -> list[StaticSolver]:
        """Get the list of solvers."""
        return self._solvers

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        if not len(set(solver.nphys for solver in self.solvers)) == 1:
            raise ValueError(
                "All solvers must have the same number of physical degrees of freedom."
            )
        return self.solvers[0].nphys
