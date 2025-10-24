"""Container for an spectral representation (eigenvalues and eigenvectors) of a matrix."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.representations.lehmann import Lehmann
from dyson.representations.representation import BaseRepresentation

if TYPE_CHECKING:
    from dyson.typing import Array


class Spectral(BaseRepresentation):
    r"""Spectral representation matrix with a known number of physical degrees of freedom.

    The eigendecomposition (spectral decomposition) of a matrix consists of the eigenvalues
    :math:`\lambda_k` and eigenvectors :math:`v_{pk}` that represent the matrix as

    .. math::
        \sum_{k} \lambda_k v_{pk} u_{qk}^*,

    where the eigenvectors have right-handed components :math:`v` and left-handed components
    :math:`u`.

    Note that the order of eigenvectors is ``(left, right)``, whilst they act in the order
    ``(right, left)`` in the above equation. The naming convention is chosen to be consistent with
    the eigenvalue decomposition, where :math:`v` may be an eigenvector acting on the right of a
    matrix, and :math:`u` is an eigenvector acting on the left of a matrix.
    """

    def __init__(
        self,
        eigvals: Array,
        eigvecs: Array,
        nphys: int,
        sort: bool = False,
        chempot: float | None = None,
    ):
        """Initialise the object.

        Args:
            eigvals: Eigenvalues of the matrix.
            eigvecs: Eigenvectors of the matrix.
            nphys: Number of physical degrees of freedom.
            sort: Sort the eigenfunctions by eigenvalue.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.
        """
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._nphys = nphys
        self.chempot = chempot
        if sort:
            self.sort_()
        if not self.hermitian:
            if eigvecs.ndim != 3:
                raise ValueError(
                    f"Couplings must be 3D for a non-Hermitian system, but got {eigvecs.ndim}D."
                )
            if eigvecs.shape[0] != 2:
                raise ValueError(
                    f"Couplings must have shape (2, nphys, naux) for a non-Hermitian system, "
                    f"but got {eigvecs.shape}."
                )

    @classmethod
    def from_matrix(
        cls, matrix: Array, nphys: int, hermitian: bool = True, chempot: float | None = None
    ) -> Spectral:
        """Create a spectrum from a matrix by diagonalising it.

        Args:
            matrix: Matrix to diagonalise.
            nphys: Number of physical degrees of freedom.
            hermitian: Whether the matrix is Hermitian.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.

        Returns:
            Spectrum object.
        """
        if hermitian:
            eigvals, eigvecs = util.eig(matrix, hermitian=True)
        else:
            eigvals, (left, right) = util.eig_lr(matrix, hermitian=False)
            eigvecs = np.array([left, right])
        return cls(eigvals, eigvecs, nphys, chempot=chempot)

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
    ) -> Spectral:
        """Create a spectrum from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.

        Returns:
            Spectrum object.
        """
        return cls(
            *self_energy.diagonalise_matrix(static, overlap=overlap),
            self_energy.nphys,
            chempot=self_energy.chempot,
        )
    
    @classmethod
    def from_poles(cls, 
                   energies: Array, 
                   residues: Array,
                   chempot: float | None = None,
                   tol: float = 1e-12,
                   assume_non_degenerate: bool = False,
                   use_svd: bool = True) -> Spectral:
        """ Build a spectral represntation from a set of pole energies and residues.

            Args:
                energies: Pole energies.
                residues: Pole residues.
                tol: Tolerance for considering eigenvalues as positive.
                assume_non_degenerate: Whether to assume non-degenerate residues.

            Returns:
                Spectrum object
        """

        energies = np.array(energies)
        residues = np.array(residues)
        naux, nphys = residues.shape[0], residues.shape[1]
        new_energies, new_couplings = [], []

        for a in range(naux):
            val, vec = np.linalg.eigh(residues[a])
            if assume_non_degenerate:
                # Assume at most one positive eigenvalue
                if val[-1] > tol:
                    coup = vec[:,-1:] @ np.diag([np.sqrt(val[-1])])
                else:
                    coup = np.array([[]])
            else:    
                # Keep all positive eigenvalues
                idx = val > tol 
                coup = vec[:,idx] @ np.diag(np.sqrt(val[idx]))
            if coup.shape[1] > 0:
                new_energies += [energies[a]] * coup.shape[1]
                new_couplings.append(coup)

        new_energies, new_couplings = np.array(new_energies), np.hstack(new_couplings)
        u = np.zeros((new_energies.shape[0], new_energies.shape[0]), dtype=residues.dtype)

        # Apply orthogonalisation metric to obtain orthogonal couplings
        orthogonalisation_metric = util.linalg.matrix_power(residues.sum(axis=0), -0.5)[0]
        new_couplings = orthogonalisation_metric @ new_couplings

        # Use QR factorisation to perform Gram-Schmidt to complete rest of space
        if use_svd:
            # needs more testing
            U, s, Vh = np.linalg.svd(new_couplings, full_matrices=True)
            idx = np.sum(s > 1e-1)
            null =Vh[idx:, :]
            eigvecs = np.concatenate([new_couplings, null])
        else:
            u[:nphys, :] = new_couplings
            eigvecs, R = np.linalg.qr(u.T, mode='complete')

        # Transform back to non-orthogonal couplings
        eigvecs[:nphys, :] = np.linalg.inv(orthogonalisation_metric) @ eigvecs[:nphys, :]

        return Spectral(new_energies, eigvecs, nphys, chempot=chempot, sort=True)

    def sort_(self) -> None:
        """Sort the eigenfunctions by eigenvalue.

        Note:
            The object is sorted in place.
        """
        idx = np.argsort(self.eigvals)
        self._eigvals = self.eigvals[idx]
        self._eigvecs = self.eigvecs[..., idx]

    def _get_matrix_block(self, slices: tuple[slice, slice]) -> Array:
        """Get a block of the matrix.

        Args:
            slices: Slices to select.

        Returns:
            Block of the matrix.
        """
        left, right = util.unpack_vectors(self.eigvecs)
        return util.einsum("pk,qk,k->pq", right[slices[0]], left[slices[1]].conj(), self.eigvals)

    def get_static_self_energy(self) -> Array:
        """Get the static part of the self-energy.

        Returns:
            Static self-energy.

        Note:
            The static part of the self-energy is defined as the physical space part of the matrix
            from which the spectrum is derived.
        """
        return self._get_matrix_block((slice(self.nphys), slice(self.nphys)))

    def get_auxiliaries(self) -> tuple[Array, Array]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Returns:
            Auxiliary energies and couplings.

        Note:
            The auxiliary energies are the eigenvalues of the auxiliary subspace, and the couplings
            are the eigenvectors projected back to the auxiliary subspace using the
            physical-auxiliary block of the matrix from which the spectrum is derived.
        """
        phys = slice(None, self.nphys)
        aux = slice(self.nphys, None)

        # Project back to the auxiliary subspace
        subspace = self._get_matrix_block((aux, aux))

        # If there are no auxiliaries, return here
        if subspace.size == 0:
            energies = np.empty((0))
            couplings = np.empty((self.nphys, 0) if self.hermitian else (2, self.nphys, 0))
            return energies, couplings

        # Diagonalise the subspace to get the energies and basis for the couplings
        energies, rotation = util.eig_lr(subspace, hermitian=self.hermitian)

        # Project back to the couplings
        couplings_right = self._get_matrix_block((phys, aux))
        if not self.hermitian:
            couplings_left = self._get_matrix_block((aux, phys)).T.conj()

        # Rotate the couplings to the auxiliary basis
        if self.hermitian:
            couplings = couplings_right @ rotation[0]
        else:
            couplings = np.array([couplings_left @ rotation[0], couplings_right @ rotation[1]])

        return energies, couplings

    def get_dyson_orbitals(self) -> tuple[Array, Array]:
        """Get the Dyson orbitals.

        Returns:
            Dyson orbitals.
        """
        return self.eigvals, self.eigvecs[..., : self.nphys, :]

    def get_overlap(self) -> Array:
        """Get the overlap matrix in the physical space.

        Returns:
            Overlap matrix.

        Note:
            The overlap matrix is defined as the zeroth moment of the Green's function, and is given
            by the inner product of the Dyson orbitals.
        """
        _, orbitals = self.get_dyson_orbitals()
        left, right = util.unpack_vectors(orbitals)
        return util.einsum("pk,qk->pq", right, left.conj())

    def get_self_energy(self, chempot: float | None = None) -> Lehmann:
        """Get the Lehmann representation of the self-energy.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the self-energy.
        """
        if chempot is None:
            chempot = self.chempot
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_auxiliaries(), chempot=chempot)

    def get_greens_function(self, chempot: float | None = None) -> Lehmann:
        """Get the Lehmann representation of the Green's function.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the Green's function.
        """
        if chempot is None:
            chempot = self.chempot
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_dyson_orbitals(), chempot=chempot)

    def combine(self, *args: Spectral, chempot: float | None = None) -> Spectral:
        """Combine multiple spectral representations by concatenating self-energies.

        Args:
            args: Spectral representations to combine.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.

        Returns:
            Combined spectral representation.
        """
        # TODO: just concatenate the eigenvectors...?
        args = (self, *args)
        if len(set(arg.nphys for arg in args)) != 1:
            raise ValueError(
                "All Spectral objects must have the same number of physical degrees of freedom."
            )
        nphys = args[0].nphys

        # Sum the overlap and static self-energy matrices -- double counting is not an issue
        # with shared static parts because the overlap matrix accounts for the separation
        static = sum([arg.get_static_self_energy() for arg in args], np.zeros((nphys, nphys)))
        overlap = sum([arg.get_overlap() for arg in args], np.zeros((nphys, nphys)))

        # Check the chemical potentials
        if chempot is None:
            if any(arg.chempot is not None for arg in args):
                chempots = [arg.chempot for arg in args if arg.chempot is not None]
                if not all(np.isclose(chempots[0], part) for part in chempots[1:]):
                    raise ValueError(
                        "If not chempot is passed to combine, all chemical potentials must be "
                        "equal in the inputs."
                    )
                chempot = chempots[0]

        # Get the auxiliaries
        energies = np.zeros((0))
        left = np.zeros((nphys, 0))
        right = np.zeros((nphys, 0))
        for arg in args:
            energies_i, couplings_i = arg.get_auxiliaries()
            # print(couplings_i.shape)
            # overlap = arg.get_overlap()
            # orth = util.matrix_power(overlap, -0.5, hermitian=False)[0]
            # unorth = util.matrix_power(overlap, 0.5, hermitian=False)[0]

            energies = np.concatenate([energies, energies_i])
            if arg.hermitian:
                #couplings_i = orth @ couplings_i
                left = np.concatenate([left, couplings_i], axis=1)
            else:
                left_i, right_i = util.unpack_vectors(couplings_i)
                #left_i = orth @ left_i
                #right_i = unorth @ right_i
                left = np.concatenate([left, left_i], axis=1)
                right = np.concatenate([right, right_i], axis=1)
        couplings = np.array([left, right]) if not args[0].hermitian else left

        # Solve the eigenvalue problem
        self_energy = Lehmann(energies, couplings)
        result = Spectral(
            *self_energy.diagonalise_matrix(static, overlap=overlap), nphys, chempot=chempot
        )

        return result
    
    def combine_dyson(self, *args: Spectral, chempot: float | None = None, ns_method: Literal["qr", "svd"] = 'svd') -> Spectral:
        """Combine multiple spectral representations by concatenating Dyson orbitals. 

        Args:
            args: Spectral representations to combine.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.

        Returns:
            Combined spectral representation.
        """
        args = (self, *args)
        if len(set(arg.nphys for arg in args)) != 1:
            raise ValueError(
                "All Spectral objects must have the same number of physical degrees of freedom."
            )
        nphys = args[0].nphys

        # Sum the overlap and static self-energy matrices -- double counting is not an issue
        # with shared static parts because the overlap matrix accounts for the separation
        static = sum([arg.get_static_self_energy() for arg in args], np.zeros((nphys, nphys)))
        overlap = sum([arg.get_overlap() for arg in args], np.zeros((nphys, nphys)))
        # Check the chemical potentials
        if chempot is None:
            if any(arg.chempot is not None for arg in args):
                chempots = [arg.chempot for arg in args if arg.chempot is not None]
                if not all(np.isclose(chempots[0], part) for part in chempots[1:]):
                    raise ValueError(
                        "If not chempot is passed to combine, all chemical potentials must be "
                        "equal in the inputs."
                    )
                chempot = chempots[0]

        # Get the dyson orbitals
        hermitian = np.all([arg.hermitian for arg in args])
        energies = []
        left = []
        right = []
        for arg in args:
            energies_i, orbitals_i = arg.get_dyson_orbitals()
            energies.append(energies_i)
            if arg.hermitian:
                left.append(orbitals_i)
            else:
                left_i, right_i = util.unpack_vectors(orbitals_i)
                left.append(left_i)
                right.append(right_i)

        energies = np.concatenate(energies, axis=0)

        if hermitian:
            print('Hermitian')
            orbitals = np.concatenate(left, axis=1)
            # Check orthonormality 
            #assert np.allclose(orbitals.T.conj()@orbitals - np.eye(orbitals.shape[1]), 0)

            rest = util.null_space_basis(orbitals, hermitian=hermitian, method=ns_method)[0]
            # Combine vectors:
            vectors = np.block([orbitals.T, rest]).T

        # assert np.allclose(vectors@vectors.T.conj()-np.eye(vectors.shape[0]), 0)
        # assert np.allclose(vectors.T.conj()@vectors-np.eye(vectors.shape[1]), 0)

        else: 
            left = np.concatenate(left, axis=1)
            right = np.concatenate(right, axis=1)

            #rest_l = util.null_space_basis(left.T.conj(), hermitian=hermitian, method=ns_method)[0]  
            #rest_r = util.null_space_basis(right, hermitian=hermitian, method=ns_method)[1]

            if ns_method in ['svd', 'qr']:
                _, rest_r = util.null_space_basis(right, hermitian=hermitian, method=ns_method)
                _, rest_l = util.null_space_basis(left, hermitian=hermitian, method=ns_method)
            elif ns_method in ['eig', 'eig-complement']:
                rest_l, rest_r = util.null_space_basis( left.conj().T @ right, hermitian=hermitian, method=ns_method)
            
            #mat = left.T.conj() @ right
            #rest_l, rest_r = util.null_space_basis(mat, hermitian=hermitian, method=ns_method)

            #rest_l, rest_r = util.biorthonormalise(rest_l, rest_r)
            print("L R - I : %s"%np.linalg.norm(rest_l.T.conj() @ rest_r - np.eye(rest_l.shape[1])))
            vectors_l = np.block([left.T, rest_l]).T
            vectors_r = np.block([right.T, rest_r]).T
            vectors = np.array([vectors_l, vectors_r])


        return Spectral(energies, vectors, nphys, sort=True) 
    

    def combine_from_poles(self, *args: Spectral, chempot: float | None = None, hermitize=True, tol=1e-12, use_svd=True) -> Spectral:
        args = (self, *args)
        if len(set(arg.nphys for arg in args)) != 1:
            raise ValueError(
                "All Spectral objects must have the same number of physical degrees of freedom."
            )
        nphys = args[0].nphys

        # Sum the overlap and static self-energy matrices -- double counting is not an issue
        # with shared static parts because the overlap matrix accounts for the separation
        static = sum([arg.get_static_self_energy() for arg in args], np.zeros((nphys, nphys)))
        overlap = sum([arg.get_overlap() for arg in args], np.zeros((nphys, nphys)))

        # Check the chemical potentials
        if chempot is None:
            if any(arg.chempot is not None for arg in args):
                chempots = [arg.chempot for arg in args if arg.chempot is not None]
                if not all(np.isclose(chempots[0], part) for part in chempots[1:]):
                    raise ValueError(
                        "If not chempot is passed to combine, all chemical potentials must be "
                        "equal in the inputs."
                    )
                chempot = chempots[0]

        # Get the auxiliaries
        energies = []
        residues = []
        for arg in args:
            energies.append(arg.eigvals)
            left, right = util.unpack_vectors(arg.get_dyson_orbitals()[1])
            res = util.einsum('pa,qa->apq', left, right.conj())
            if hermitize:
                res = 0.5 * (res + res.transpose(0,1,2))
            residues.append(res)

        energies = np.concatenate(energies)
        residues = np.concatenate(residues)

        return Spectral.from_poles(energies, residues, chempot=chempot, assume_non_degenerate=True, tol=tol, use_svd=use_svd)
    
    def hermitize(self, tol: float = 1e-12) -> Spectral:
        """ Convert a non-hermitian spectral representation to a hermitian one by hermitizing the
            corresponding Green's function.

            Args:
                tol: Tolerance for considering eigenvalues as positive.
            
            Returns:
                Hermitian spectral representation.

            Raises:
                ValueError: If the spectral representation is already Hermitian.
        """
        if self.hermitian:
            raise ValueError("Spectral representation is already Hermitian.")

        gf = self.get_greens_function()
        gfh = gf.hermitize(tol=tol)
        nphys = gfh.couplings.shape[0]
        orthogonalisation_metric = util.linalg.matrix_power(gfh.moment(0), -0.5)[0]
        couplings = orthogonalisation_metric @ gfh.couplings

        null = util.linalg.null_space_basis(couplings)[1]
        eigvecs = np.vstack((couplings, null.T))
        eigvecs[:nphys, :] = np.linalg.inv(orthogonalisation_metric) @ eigvecs[:nphys, :]
        
        return Spectral(gfh.energies, eigvecs, nphys, chempot=gfh.chempot)

    @cached_property
    def overlap(self) -> Array:
        """Get the overlap matrix (the zeroth moment of the Green's function)."""
        orbitals = self.get_dyson_orbitals()[1]
        left, right = util.unpack_vectors(orbitals)
        return util.einsum("pk,qk->pq", right, left.conj())

    @property
    def eigvals(self) -> Array:
        """Get the eigenvalues."""
        return self._eigvals

    @property
    def eigvecs(self) -> Array:
        """Get the eigenvectors."""
        return self._eigvecs

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys

    @property
    def neig(self) -> int:
        """Get the number of eigenvalues."""
        return self.eigvals.shape[0]

    @property
    def hermitian(self) -> bool:
        """Check if the spectrum is Hermitian."""
        return self.eigvecs.ndim == 2

    def __eq__(self, other: object) -> bool:
        """Check if two Lehmann representations are equal."""
        if not isinstance(other, Spectral):
            return NotImplemented
        if other.nphys != self.nphys:
            return False
        if other.neig != self.neig:
            return False
        if other.hermitian != self.hermitian:
            return False
        if other.chempot != self.chempot:
            return False
        return np.allclose(other.eigvals, self.eigvals) and np.allclose(other.eigvecs, self.eigvecs)

    def __hash__(self) -> int:
        """Hash the object."""
        return hash((tuple(self.eigvals), tuple(self.eigvecs.flatten()), self.nphys, self.chempot))
