"""Density matrix relaxing solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.lib.diis import DIIS

from dyson import console, printing
from dyson import numpy as np
from dyson._backend import _BACKEND
from dyson.representations.lehmann import Lehmann
from dyson.solvers.solver import StaticSolver
from dyson.solvers.static.chempot import AufbauPrinciple, AuxiliaryShift

if TYPE_CHECKING:
    from typing import Any, Protocol

    from pyscf import scf

    from dyson.expressions.expression import BaseExpression
    from dyson.representations.spectral import Spectral
    from dyson.typing import Array

    class StaticFunction(Protocol):
        """Protocol for a function that computes the static self-energy."""

        def __call__(
            self,
            rdm1: Array,
            rdm1_prev: Array | None = None,
            static_prev: Array | None = None,
        ) -> Array:
            """Compute the static self-energy for a given density matrix.

            Args:
                rdm1: Density matrix.
                rdm1_prev: Previous density matrix. Used for direct build.
                static_prev: Previous Fock matrix. Used for direct build.

            Returns:
                Static self-energy.
            """
            ...


if _BACKEND == "jax" and not TYPE_CHECKING:
    # Try to get the JAX version of DIIS
    try:
        from pyscfad.lib.diis import DIIS
    except ImportError:
        pass


def get_fock_matrix_function(mf: scf.hf.RHF) -> StaticFunction:
    """Get a function to compute the Fock matrix for a given density matrix.

    Args:
        mf: Mean-field object.

    Returns:
        Function to compute the Fock matrix.
    """
    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()

    def get_fock(
        rdm1: Array, rdm1_prev: Array | None = None, static_prev: Array | None = None
    ) -> Array:
        """Compute the Fock matrix for a given density matrix.

        Args:
            rdm1: Density matrix.
            rdm1_prev: Previous density matrix. Used for direct build.
            static_prev: Previous Fock matrix. Used for direct build.

        Returns:
            Fock matrix.
        """
        # Transform to AO basis
        rdm1 = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T.conj()
        if (rdm1_prev is None) != (static_prev is None):
            raise ValueError(
                "Both rdm1_prev and static_prev must be None or both must be provided."
            )
        if rdm1_prev is not None and static_prev is not None:
            rdm1_prev = mf.mo_coeff @ rdm1_prev @ mf.mo_coeff.T.conj()
            static_prev = mf.mo_coeff @ static_prev @ mf.mo_coeff.T.conj()

        # Compute the new Fock matrix
        veff_last = static_prev - h1e if static_prev is not None else None
        veff = mf.get_veff(dm=rdm1, dm_last=rdm1_prev, vhf_last=veff_last)
        fock = mf.get_fock(h1e=h1e, s1e=s1e, vhf=veff, dm=rdm1)

        # Transform back to MO basis
        fock = mf.mo_coeff.T.conj() @ fock @ mf.mo_coeff

        return fock

    return get_fock


class DensityRelaxation(StaticSolver):
    """Solve a self-energy and relax the density matrix in the presence of the auxiliaries.

    Args:
        get_static: Function to get the static self-energy (including Fock contributions) for a
            given density matrix.
        self_energy: Self-energy.
        nelec: Target number of electrons.
    """

    occupancy: float = 2.0
    solver_outer: type[AuxiliaryShift] = AuxiliaryShift
    solver_inner: type[AufbauPrinciple] = AufbauPrinciple
    diis_min_space: int = 2
    diis_max_space: int = 8
    max_cycle_outer: int = 20
    max_cycle_inner: int = 50
    conv_tol: float = 1e-8
    favour_rdm: bool = True
    _options: set[str] = {
        "occupancy",
        "solver_outer",
        "solver_inner",
        "diis_min_space",
        "diis_max_space",
        "max_cycle_outer",
        "max_cycle_inner",
        "conv_tol",
        "favour_rdm",
    }

    converged: bool | None = None

    def __init__(  # noqa: D417
        self,
        get_static: StaticFunction,
        self_energy: Lehmann,
        nelec: int,
        overlap: Array | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            get_static: Function to get the static self-energy (including Fock contributions) for a
                given density matrix.
            self_energy: Self-energy.
            nelec: Target number of electrons.
            overlap: Overlap matrix for the physical space.
            occupancy: Occupancy of each state, typically 2 for a restricted reference and 1
                otherwise.
            solver_outer: Solver to use for the self-energy and chemical potential search in the
                outer loop.
            solver_inner: Solver to use for the self-energy and chemical potential search in the
                inner loop.
            diis_min_space: Minimum size of the DIIS space.
            diis_max_space: Maximum size of the DIIS space.
            max_cycle_outer: Maximum number of outer iterations.
            max_cycle_inner: Maximum number of inner iterations.
            conv_tol: Convergence tolerance in the density matrix.
            favour_rdm: Whether to favour the density matrix over the number of electrons in the
                non-commuting solutions.
        """
        self._get_static = get_static
        self._self_energy = self_energy
        self._nelec = nelec
        self._overlap = overlap
        self.set_options(**kwargs)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if not callable(self.get_static):
            raise TypeError("get_static must be a callable function.")
        if self.overlap is not None and (
            self.overlap.ndim != 2 or self.overlap.shape[0] != self.overlap.shape[1]
        ):
            raise ValueError("overlap must be a square matrix or None.")
        if self.overlap is not None and self.overlap.shape[0] != self.self_energy.nphys:
            raise ValueError(
                "overlap must have the same number of physical states as the self-energy."
            )

        # Print the input information
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        console.print(f"Number of auxiliary states: [input]{self.self_energy.naux}[/input]")
        console.print(f"Target number of electrons: [input]{self.nelec}[/input]")
        if self.overlap is not None:
            cond = printing.format_float(
                np.linalg.cond(self.overlap), threshold=1e10, scientific=True, precision=4
            )
            console.print(f"Overlap condition number: {cond}")

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        assert self.result is not None
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )
        nelec = np.trace(self.result.get_greens_function().occupied().moment(0)) * self.occupancy
        if self.result.chempot is not None:
            cpt = printing.format_float(self.result.chempot)
            console.print(f"Chemical potential: [output]{cpt}[/output]")
        err = printing.format_float(
            self.nelec - nelec, threshold=1e-3, precision=4, scientific=True
        )
        console.print(f"Error in number of electrons: [output]{err}[/output]")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> DensityRelaxation:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.

        Notes:
            To initialise this solver from a self-energy, the ``nelec`` and ``get_static`` keyword
            arguments must be provided.
        """
        if "nelec" not in kwargs:
            raise ValueError("Missing required argument nelec.")
        if "get_static" not in kwargs:
            raise ValueError("Missing required argument get_static.")
        kwargs = kwargs.copy()
        nelec = kwargs.pop("nelec")
        get_static = kwargs.pop("get_static")
        return cls(get_static, self_energy, nelec, overlap=overlap, **kwargs)

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> DensityRelaxation:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            "Cannot instantiate DensityRelaxation from expression, use from_self_energy instead."
        )

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Get the table
        table = printing.ConvergencePrinter(
            ("Shift",),
            (
                "Error",
                "Gradient",
                "Change in RDM",
            ),
            (
                self.solver_outer.conv_tol,
                self.solver_outer.conv_tol_grad,
                self.conv_tol,
            ),
        )
        progress = printing.IterationsPrinter(self.max_cycle_outer)
        progress.start()

        # Get the initial parameters
        self_energy = self.self_energy
        nocc = self.nelec // self.occupancy
        rdm1 = np.diag(np.arange(self.nphys) < nocc).astype(self_energy.dtype) * self.occupancy
        static = self.get_static(rdm1)

        converged = False
        for cycle_outer in range(1, self.max_cycle_outer + 1):
            if self.favour_rdm:
                # Solve the self-energy
                with printing.quiet:
                    solver_outer = self.solver_outer.from_self_energy(
                        static, self_energy, nelec=self.nelec, overlap=self.overlap
                    )
                    result = solver_outer.kernel()
                    self_energy = result.get_self_energy()

            # Initialise DIIS for the inner loop
            diis = DIIS()
            diis.space = self.diis_min_space
            diis.max_space = self.diis_max_space
            diis.incore = True
            diis.verbose = 0

            for cycle_inner in range(1, self.max_cycle_inner + 1):
                # Solve the self-energy
                with printing.quiet:
                    solver_inner = self.solver_inner.from_self_energy(
                        static, self_energy, nelec=self.nelec, overlap=self.overlap
                    )
                    result = solver_inner.kernel()
                    self_energy = result.get_self_energy()

                # Get the density matrix
                greens_function = result.get_greens_function()
                rdm1_prev = rdm1.copy()
                rdm1 = greens_function.occupied().moment(0) * self.occupancy

                # Update the static self-energy
                static = self.get_static(rdm1, rdm1_prev=rdm1_prev, static_prev=static)
                try:
                    if not self_energy.hermitian and not np.iscomplexobj(static):
                        # Avoid casting errors if non-Hermitian self-energy starts as real and
                        # becomes complex during the iterations... probably more efficient to
                        # subclass DIIS to handle this.
                        static = static.astype(np.complex128)
                    static = diis.update(static, xerr=None)
                except np.linalg.LinAlgError:
                    pass

                # Check for convergence
                error = np.max(np.abs(rdm1 - rdm1_prev))
                if error < self.conv_tol:
                    break

            if not self.favour_rdm:
                # Solve the self-energy
                with printing.quiet:
                    solver_outer = self.solver_outer.from_self_energy(
                        static, self_energy, nelec=self.nelec, overlap=self.overlap
                    )
                    result = solver_outer.kernel()
                    self_energy = result.get_self_energy()

            # Check for convergence
            converged = bool(error < self.conv_tol and solver_outer.converged)
            grad = np.ravel(solver_outer.gradient(solver_outer.shift)[1])[0]
            table.add_row(cycle_outer, (solver_outer.shift,), (solver_outer.error, grad, error))
            progress.update(cycle_outer)
            if converged:
                break

        progress.stop()
        table.print()

        # Set the results
        self.converged = converged
        self.result = result

        return result

    @property
    def get_static(self) -> StaticFunction:
        """Get the static self-energy function."""
        return self._get_static

    @property
    def self_energy(self) -> Lehmann:
        """Get the self-energy."""
        return self._self_energy

    @property
    def nelec(self) -> int:
        """Get the target number of electrons."""
        return self._nelec

    @property
    def overlap(self) -> Array | None:
        """Get the overlap matrix for the physical space."""
        return self._overlap

    @property
    def nphys(self) -> int:
        """Get the number of physical states."""
        return self.self_energy.nphys
