"""Base class for expressions."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.representations.enums import Reduction

if TYPE_CHECKING:
    from typing import Callable

    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array


class BaseExpression(ABC):
    """Base class for expressions."""

    hermitian_downfolded: bool = True
    hermitian_upfolded: bool = True

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
    def get_excitation_vector(self, orbital: int) -> Array:
        r"""Obtain the vector corresponding to a fermionic operator acting on the ground state.

        This vector is a generalisation of

        .. math::
            f_i^{\pm} \left| \Psi_0 \right>

        where :math:`f_i^{\pm}` is the fermionic creation or annihilation operator, or a product
        thereof, depending on the particular expression and what Green's function it corresponds to.

        The vector defines the excitaiton manifold probed by the Green's function corresponding to
        the expression.

        Args:
            orbital: Orbital index.

        Returns:
            Excitation vector.
        """
        pass

    def get_excitation_bra(self, orbital: int) -> Array:
        r"""Obtain the bra vector corresponding to a fermionic operator acting on the ground state.

        The bra vector is the excitation vector corresponding to the bra state, which may or may not
        be the same as the ket state vector.

        Args:
            orbital: Orbital index.

        Returns:
            Bra excitation vector.

        See Also:
            :func:`get_excitation_vector`: Function to get the excitation vector when the bra and
            ket are the same.
        """
        return self.get_excitation_vector(orbital)

    def get_excitation_ket(self, orbital: int) -> Array:
        r"""Obtain the ket vector corresponding to a fermionic operator acting on the ground state.

        The ket vector is the excitation vector corresponding to the ket state, which may or may not
        be the same as the bra state vector.

        Args:
            orbital: Orbital index.

        Returns:
            Ket excitation vector.

        See Also:
            :func:`get_excitation_vector`: Function to get the excitation vector when the bra and
            ket are the same.
        """
        return self.get_excitation_vector(orbital)

    def get_excitation_vectors(self) -> list[Array]:
        """Get the excitation vectors for all orbitals.

        Returns:
            List of excitation vectors for all orbitals.

        See Also:
            :func:`get_excitation_vector`: Function to get the excitation vector for a single
            orbital.
        """
        return [self.get_excitation_vector(i) for i in range(self.nphys)]

    def get_excitation_bras(self) -> list[Array]:
        """Get the bra excitation vectors for all orbitals.

        Returns:
            List of bra excitation vectors for all orbitals.

        See Also:
            :func:`get_excitation_bra`: Function to get the bra excitation vector for a single
            orbital.
        """
        return [self.get_excitation_bra(i) for i in range(self.nphys)]

    def get_excitation_kets(self) -> list[Array]:
        """Get the ket excitation vectors for all orbitals.

        Returns:
            List of ket excitation vectors for all orbitals.

        See Also:
            :func:`get_excitation_ket`: Function to get the ket excitation vector for a single
            orbital.
        """
        return [self.get_excitation_ket(i) for i in range(self.nphys)]

    def _build_gf_moments(
        self,
        get_bra: Callable[[int], Array],
        get_ket: Callable[[int], Array],
        apply_hamiltonian_poly: Callable[[Array, Array, int], Array],
        nmom: int,
        store_vectors: bool = True,
        left: bool = False,
        reduction: Reduction = Reduction.NONE,
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
                if Reduction(reduction) == Reduction.NONE:
                    # Loop over bra vectors
                    for j in range(i if self.hermitian_downfolded else 0, self.nphys):
                        bra = bras[j] if store_vectors else get_bra(j)

                        # Contract the bra and ket vectors
                        moments[n, i, j] = bra.conj() @ ket
                        if self.hermitian_downfolded:
                            moments[n, j, i] = moments[n, i, j].conj()

                else:
                    # Contract the bra and ket vectors
                    bra = bras[i] if store_vectors else get_bra(i)
                    moments[n, i, i] = bra.conj() @ ket

                # Apply the Hamiltonian to the ket vector
                if n != nmom - 1:
                    ket, ket_prev = apply_hamiltonian_poly(ket, ket_prev, n), ket

        if Reduction(reduction) == Reduction.NONE:
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

        elif Reduction(reduction) == Reduction.DIAG:
            # Convert the moments to a numpy array, only keeping the diagonal elements
            moments_array = np.array(
                [moments[n, i, i] for n in range(nmom) for i in range(self.nphys)]
            )
            moments_array = moments_array.reshape(nmom, self.nphys)

        elif Reduction(reduction) == Reduction.TRACE:
            # Convert the moments to a numpy array, only keeping the trace
            moments_array = np.array(
                [sum([moments[n, i, i] for i in range(self.nphys)]) for n in range(nmom)]
            )

        else:
            Reduction(reduction).raise_invalid_representation()

        return moments_array

    def build_gf_moments(
        self,
        nmom: int,
        store_vectors: bool = True,
        left: bool = False,
        reduction: Reduction = Reduction.NONE,
    ) -> Array:
        """Build the moments of the Green's function.

        Args:
            nmom: Number of moments to compute.
            store_vectors: Whether to store the vectors on disk. Storing the vectors makes the
                memory overhead scale worse, but the CPU overhead scales better.
            left: Whether to use the left-handed Hamiltonian application.
            reduction: Reduction to apply to the moments.

        Returns:
            Moments of the Green's function.

        Notes:
            Unlike :func:`dyson.representations.lehmann.Lehmann.moments`, this function takes the
            number of moments to compute as an argument, rather than a single order or list of
            orders. This is because in this case, the moments are computed recursively.
        """
        # Get the appropriate functions
        if left:
            get_bra = self.get_excitation_ket
            get_ket = self.get_excitation_bra
            apply_hamiltonian = self.apply_hamiltonian_left
        else:
            get_bra = self.get_excitation_bra
            get_ket = self.get_excitation_ket
            apply_hamiltonian = self.apply_hamiltonian_right

        return self._build_gf_moments(
            get_bra,
            get_ket,
            lambda vector, vector_prev, n: apply_hamiltonian(vector),
            nmom,
            store_vectors=store_vectors,
            left=left,
            reduction=reduction,
        )

    def build_gf_chebyshev_moments(
        self,
        nmom: int,
        store_vectors: bool = True,
        left: bool = False,
        scaling: tuple[float, float] | None = None,
        reduction: Reduction = Reduction.NONE,
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
            reduction: Reduction to apply to the moments.

        Returns:
            Chebyshev polynomial moments of the Green's function.

        Notes:
            Unlike :func:`dyson.representations.lehmann.Lehmann.chebyshev_moments`, this function
            takes the number of moments to compute as an argument, rather than a single order or
            list of orders. This is because in this case, the moments are computed recursively.
        """
        if scaling is None:
            # Approximate the energy scale of the spectrum using the diagonal -- can also use an
            # iterative eigensolver to better approximate this
            diag = self.diagonal()
            scaling = util.get_chebyshev_scaling_parameters(diag.min(), diag.max())

        # Get the appropriate functions
        if left:
            get_bra = self.get_excitation_ket
            get_ket = self.get_excitation_bra
            apply_hamiltonian = self.apply_hamiltonian_left
        else:
            get_bra = self.get_excitation_bra
            get_ket = self.get_excitation_ket
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
            reduction=reduction,
        )

    @abstractmethod
    def build_se_moments(self, nmom: int, reduction: Reduction = Reduction.NONE) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.
            reduction: Reduction to apply to the moments.

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
    def hermitian(self) -> bool:
        """Whether the expression is Hermitian."""
        return self.hermitian_downfolded and self.hermitian_upfolded

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


class _ExpressionCollectionMeta(type):
    """Metaclass for the ExpressionCollection class."""

    __wrapped__: bool = False

    def __getattr__(cls, key: str) -> type[BaseExpression]:
        """Get an expression by its name."""
        if key in {"hole", "ip", "o", "h"}:
            if cls._hole is None:
                raise ValueError("Hole expression is not set.")
            return cls._hole
        elif key in {"particle", "ea", "v", "p"}:
            if cls._particle is None:
                raise ValueError("Particle expression is not set.")
            return cls._particle
        elif key in {"central", "dyson"}:
            if cls._dyson is None:
                raise ValueError("Central (Dyson) expression is not set.")
            return cls._dyson
        elif key in {"neutral", "ee", "ph"}:
            if cls._neutral is None:
                raise ValueError("Neutral expression is not set.")
            return cls._neutral
        else:
            raise ValueError(f"Expression '{key}' is not defined in the collection.")

    __getitem__ = __getattr__

    @property
    def _classes(cls) -> set[type[BaseExpression]]:
        """Get all classes in the collection."""
        return {
            cls for cls in [cls._hole, cls._particle, cls._dyson, cls._neutral] if cls is not None
        }

    def __contains__(cls, key: str) -> bool:
        """Check if an expression exists by its name."""
        try:
            cls[key]  # type: ignore[index]
            return True
        except ValueError:
            return False


class ExpressionCollection(metaclass=_ExpressionCollectionMeta):
    """Collection of expressions for different parts of the Green's function."""

    _hole: type[BaseExpression] | None = None
    _particle: type[BaseExpression] | None = None
    _dyson: type[BaseExpression] | None = None
    _neutral: type[BaseExpression] | None = None
    _name: str | None = None

    @classmethod
    def __getattr__(cls, key: str) -> type[BaseExpression]:
        """Get an expression by its name."""
        return getattr(type(cls), key)

    __getitem__ = __getattr__

    def __contains__(cls, key: str) -> bool:
        """Check if an expression exists by its name."""
        return getattr(type(cls), key, None) is not None

    @classmethod
    def __repr__(cls) -> str:
        """String representation of the collection."""
        return f"ExpressionCollection({cls._name})" if cls._name else "ExpressionCollection"
