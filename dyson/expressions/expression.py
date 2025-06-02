"""Base class for expressions."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from dyson import numpy as np
from dyson import util

if TYPE_CHECKING:
    from typing import Callable, ItemsView, KeysView, ValuesView

    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array


class BaseExpression(ABC):
    """Base class for expressions."""

    hermitian: bool = True

    @classmethod
    @abstractmethod
    def from_mf(cls, mf: RHF) -> BaseExpression:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object.

        Returns:
            Expression object.
        """
        pass

    @abstractmethod
    def apply_hamiltonian(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        pass

    def apply_hamiltonian_left(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the left.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return self.apply_hamiltonian(vector)

    def apply_hamiltonian_right(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the right.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return self.apply_hamiltonian(vector)

    @abstractmethod
    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        pass

    def build_matrix(self) -> Array:
        """Build the Hamiltonian matrix.

        Returns:
            Hamiltonian matrix.

        Notes:
            This method uses :func:`apply_hamiltonian` to build the matrix by applying unit vectors,
            it is not designed to be efficient.
        """
        size = self.diagonal().size
        if size > 2048:
            warnings.warn(
                "The Hamiltonian matrix is large. This may take a while to compute.",
                UserWarning,
                2,
            )
        return np.array([self.apply_hamiltonian(util.unit_vector(size, i)) for i in range(size)])

    @abstractmethod
    def get_state(self, orbital: int) -> Array:
        r"""Obtain the state vector corresponding to a fermion operator acting on the ground state.

        This state vector is a generalisation of

        .. math::
            a_i^{\pm} \left| \Psi_0 \right>

        where :math:`a_i^{\pm}` is the fermionic creation or annihilation operator, depending on the
        particular expression.

        The state vector can be used to find the action of the singles and higher-order
        configurations in the Hamiltonian on the physical space, required to compute Green's
        functions.

        Args:
            orbital: Orbital index.

        Returns:
            State vector.
        """
        pass

    def get_state_bra(self, orbital: int) -> Array:
        r"""Obtain the bra vector corresponding to a fermion operator acting on the ground state.

        The bra vector is the state vector corresponding to the bra state, which may or may not be
        the same as the ket state vector.

        Args:
            orbital: Orbital index.

        Returns:
            Bra vector.

        See Also:
            :func:`get_state`: Function to get the state vector when the bra and ket are the same.
        """
        return self.get_state(orbital)

    def get_state_ket(self, orbital: int) -> Array:
        r"""Obtain the ket vector corresponding to a fermion operator acting on the ground state.

        The ket vector is the state vector corresponding to the ket state, which may or may not be
        the same as the bra state vector.

        Args:
            orbital: Orbital index.

        Returns:
            Ket vector.

        See Also:
            :func:`get_state`: Function to get the state vector when the bra and ket are the same.
        """
        return self.get_state(orbital)

    def _build_gf_moments(
        self,
        get_bra: Callable[[int], Array],
        get_ket: Callable[[int], Array],
        apply_hamiltonian_poly: Callable[[Array, Array, int], Array],
        nmom: int,
        store_vectors: bool = True,
        left: bool = False,
    ) -> Array:
        """Build the moments of the Green's function."""
        # Precompute bra vectors if needed
        if store_vectors:
            bras = list(map(get_bra, range(self.nphys)))

        # Loop over ket vectors
        moments: dict[tuple[int, int, int], Array] = {}
        for i in range(self.nphys):
            ket = ket_prev = get_ket(i)

            # Loop over moment orders
            for n in range(nmom):
                # Loop over bra vectors
                for j in range(i if self.hermitian else 0, self.nphys):
                    bra = bras[j] if store_vectors else get_bra(j)

                    # Contract the bra and ket vectors
                    moments[n, i, j] = bra.conj() @ ket
                    if self.hermitian:
                        moments[n, j, i] = moments[n, i, j].conj()

                # Apply the Hamiltonian to the ket vector
                if n != nmom - 1:
                    ket, ket_prev = apply_hamiltonian_poly(ket, ket_prev, n), ket

        # Convert the moments to a numpy array
        moments_array = np.array(
            [
                moments[n, i, j]
                for n in range(nmom)
                for i in range(self.nphys)
                for j in range(self.nphys)
            ]
        )
        moments_array = moments_array.reshape(nmom, self.nphys, self.nphys)

        # If left-handed, transpose the moments
        if left:
            moments_array = moments_array.transpose(0, 2, 1).conj()

        return moments_array

    def build_gf_moments(self, nmom: int, store_vectors: bool = True, left: bool = False) -> Array:
        """Build the moments of the Green's function.

        Args:
            nmom: Number of moments to compute.
            store_vectors: Whether to store the vectors on disk. Storing the vectors makes the
                memory overhead scale worse, but the CPU overhead scales better.
            left: Whether to use the left-handed Hamiltonian application.

        Returns:
            Moments of the Green's function.

        Notes:
            Unlike :func:`dyson.lehmann.Lehmann.moments`, this function takes the number of moments
            to compute as an argument, rather than a single order or list of orders. This is because
            in this case, the moments are computed recursively.
        """
        # Get the appropriate functions
        if left:
            get_bra = self.get_state_ket
            get_ket = self.get_state_bra
            apply_hamiltonian = self.apply_hamiltonian_left
        else:
            get_bra = self.get_state_bra
            get_ket = self.get_state_ket
            apply_hamiltonian = self.apply_hamiltonian_right

        return self._build_gf_moments(
            get_bra,
            get_ket,
            lambda vector, vector_prev, n: apply_hamiltonian(vector),
            nmom,
            store_vectors=store_vectors,
            left=left,
        )

    def build_gf_chebyshev_moments(
        self,
        nmom: int,
        store_vectors: bool = True,
        left: bool = False,
        scaling: tuple[float, float] | None = None,
    ) -> Array:
        """Build the moments of the Green's function using Chebyshev polynomials.

        Args:
            nmom: Number of moments to compute.
            store_vectors: Whether to store the vectors on disk. Storing the vectors makes the
                memory overhead scale worse, but the CPU overhead scales better.
            left: Whether to use the left-handed Hamiltonian application.
            scaling: Scaling factors to ensure the energy scale of the Lehmann representation is
                in `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`. If
                `None`, the default scaling is computed as
                `(max(energies) - min(energies)) / (2.0 - 1e-3)` and
                `(max(energies) + min(energies)) / 2.0`, respectively.

        Returns:
            Chebyshev polynomial moments of the Green's function.

        Notes:
            Unlike :func:`dyson.lehmann.Lehmann.chebyshev_moments`, this function takes the number
            of moments to compute as an argument, rather than a single order or list of orders. This
            is because in this case, the moments are computed recursively.
        """
        if scaling is None:
            # Approximate the energy scale of the spectrum using the diagonal -- can also use an
            # iterative eigensolver to better approximate this
            diag = self.diagonal()
            emin = diag.min()
            emax = diag.max()
            scaling = (
                (emax - emin) / (2.0 - 1e-3),
                (emax + emin) / 2.0,
            )

        # Get the appropriate functions
        if left:
            get_bra = self.get_state_ket
            get_ket = self.get_state_bra
            apply_hamiltonian = self.apply_hamiltonian_left
        else:
            get_bra = self.get_state_bra
            get_ket = self.get_state_ket
            apply_hamiltonian = self.apply_hamiltonian_right

        def _apply_hamiltonian_poly(vector: Array, vector_prev: Array, n: int) -> Array:
            """Apply the scaled Hamiltonian polynomial to a vector."""
            # [(H - b) / a] v = H (v / a) - b (v / a)
            vector_scaled = vector / scaling[0]
            result = apply_hamiltonian(vector_scaled) - scaling[1] * vector_scaled
            if n == 0:
                return result  # u_{1} = H u_{0}
            return 2.0 * result - vector_prev  # u_{n} = 2 H u_{n-1} - u_{n-2}

        return self._build_gf_moments(
            get_bra,
            get_ket,
            _apply_hamiltonian_poly,
            nmom,
            store_vectors=store_vectors,
            left=left,
        )

    @abstractmethod
    def build_se_moments(self, nmom: int) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.

        Returns:
            Moments of the self-energy.
        """
        pass

    @property
    @abstractmethod
    def mol(self) -> Mole:
        """Molecule object."""
        pass

    @property
    @abstractmethod
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        pass

    @property
    def nphys(self) -> int:
        """Number of physical orbitals."""
        return self.mol.nao

    @property
    @abstractmethod
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        pass

    @property
    def nconfig(self) -> int:
        """Number of configurations in the non-singles sectors."""
        return self.diagonal().size - self.nsingle

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the Hamiltonian matrix."""
        return (self.nconfig + self.nsingle, self.nconfig + self.nsingle)

    @property
    def nocc(self) -> int:
        """Number of occupied orbitals."""
        if self.mol.nelectron % 2:
            raise NotImplementedError("Open-shell systems are not supported.")
        return self.mol.nelectron // 2

    @property
    def nvir(self) -> int:
        """Number of virtual orbitals."""
        return self.nphys - self.nocc


class ExpressionCollection:
    """Collection of expressions for different parts of the Green's function."""

    def __init__(
        self,
        hole: type[BaseExpression] | None = None,
        particle: type[BaseExpression] | None = None,
        central: type[BaseExpression] | None = None,
        neutral: type[BaseExpression] | None = None,
        name: str | None = None,
    ):
        """Initialise the collection.

        Args:
            hole: Hole expression.
            particle: Particle expression.
            central: Central expression.
            neutral: Neutral expression.
            name: Name of the collection.
        """
        self._hole = hole
        self._particle = particle
        self._central = central
        self._neutral = neutral
        self.name = name

    @property
    def hole(self) -> type[BaseExpression]:
        """Hole expression."""
        if self._hole is None:
            raise ValueError("Hole expression is not set.")
        return self._hole

    ip = o = h = cast(type[BaseExpression], hole)

    @property
    def particle(self) -> type[BaseExpression]:
        """Particle expression."""
        if self._particle is None:
            raise ValueError("Particle expression is not set.")
        return self._particle

    ea = v = p = cast(type[BaseExpression], particle)

    @property
    def central(self) -> type[BaseExpression]:
        """Central expression."""
        if self._central is None:
            raise ValueError("Central expression is not set.")
        return self._central

    dyson = cast(type[BaseExpression], central)

    @property
    def neutral(self) -> type[BaseExpression]:
        """Neutral expression."""
        if self._neutral is None:
            raise ValueError("Neutral expression is not set.")
        return self._neutral

    ee = ph = cast(type[BaseExpression], neutral)

    def __dict__(self) -> dict[str, type[BaseExpression]]:  # type: ignore[override]
        """Get a dictionary representation of the collection."""
        exps: dict[str, type[BaseExpression]] = {}
        for key in ("hole", "particle", "central", "neutral"):
            if key in self:
                exps[key] = getattr(self, key)
        return exps

    def keys(self) -> KeysView[str]:
        """Get the keys of the collection."""
        return self.__dict__().keys()

    def values(self) -> ValuesView[type[BaseExpression]]:
        """Get the values of the collection."""
        return self.__dict__().values()

    def items(self) -> ItemsView[str, type[BaseExpression]]:
        """Get an item view of the collection."""
        return self.__dict__().items()

    def __getitem__(self, key: str) -> type[BaseExpression]:
        """Get an expression by its name."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if an expression exists by its name."""
        try:
            self[key]
            return True
        except ValueError:
            return False

    def __repr__(self) -> str:
        """String representation of the collection."""
        return f"ExpressionCollection({self.name})" if self.name else "ExpressionCollection"
