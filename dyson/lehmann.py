"""
Containers for Lehmann representations.
"""

import numpy as np


class Lehmann:
    """
    Lehmann representations.

    Parameters
    ----------
    energies : numpy.ndarray
        Energies of the poles in the Lehmann representation.
    couplings : numpy.ndarray or tuple of numpy.ndarray
        Couplings of the poles in the Lehmann representation to a
        physical space (i.e. Dyson orbitals in the case of a Green's
        function).  Can be a tuple of `(left, right)` for a
        non-Hermitian representation.
    chempot : float, optional
        Chemical potential, indicating the position of the Fermi
        energy.  Default value is `0.0`.
    """

    def __init__(self, energies, couplings, chempot=0.0):
        # Input:
        self.energies = energies
        self.couplings = couplings
        self.chempot = chempot

    @classmethod
    def from_pyscf(cls, lehmann_pyscf):
        """
        Construct a Lehmann representation from a PySCF SelfEnergy
        or GreensFunction.

        Parameters
        ----------
        lehmann_pyscf : pyscf.agf2.aux.AuxSpace
            PySCF Lehmann representation.

        Returns
        -------
        lehmann : Lehmann
            Lehmann representation.
        """

        if isinstance(lehmann_pyscf, cls):
            return lehmann_pyscf
        else:
            return cls(
                lehmann_pyscf.energy,
                lehmann_pyscf.coupling,
                chempot=lehmann_pyscf.chempot,
            )

    def moment(self, order):
        """
        Get a spectral moment (or range of moments) of the Lehmann
        representation.

        Parameters
        ----------
        order : int or iterable of int
            Order(s) to calculate.

        Returns
        -------
        moment : numpy.ndarray
            Spectral moments, if `order` is an `int` then the moment
            is a 2D matrix, and if `order` is an `iterable` then it is
            a 3D matrix enumerating the orders.
        """

        squeeze = False
        if isinstance(order, int):
            order = [order]
            squeeze = True
        order = np.array(order)

        couplings_l, couplings_r = self._unpack_couplings()

        moment = np.einsum(
            "pk,qk,nk->npq",
            couplings_l,
            couplings_r.conj(),
            self.energies[None] ** order[:, None],
            optimize=True,
        )

        if squeeze:
            moment = moment[0]

        return moment

    def matrix(self, physical, chempot=False, out=None):
        """
        Build a dense matrix consisting of a matrix (i.e. a
        Hamiltonian) in the physical space coupling to a series of
        energies as defined by the Lehmann representation.

        Parameters
        ----------
        physical : numpy.ndarray
            Physical space part of the matrix.
        chempot : bool or float, optional
            Include the chemical potential in the energies.  If given
            as a `bool`, use `self.chempot`.  If a `float` then use
            this as the chemical potential.  Default value is `False`.

        Returns
        -------
        matrix : numpy.ndarray
            Dense matrix representation.
        """

        couplings_l, couplings_r = self._unpack_couplings()

        energies = self.energies
        if chempot:
            energies = energies - chempot

        if out is None:
            dtype = np.result_type(couplings_l.dtype, couplings_r.dtype, physical.dtype)
            out = np.zeros((self.nphys + self.naux, self.nphys + self.naux), dtype=dtype)

        out[: self.nphys, : self.nphys] = physical
        out[: self.nphys, self.nphys :] = couplings_l
        out[self.nphys :, : self.nphys] = couplings_r.T.conj()
        out[self.nphys :, self.nphys :] = np.diag(energies)

        return out

    def diagonalise_matrix(self, physical, chempot=False, out=None):
        """
        Diagonalise the dense matrix representation of the Lehmann
        representation.

        Parameters
        ----------
        physical : numpy.ndarray
            Physical space part of the matrix.
        chempot : bool or float, optional
            Include the chemical potential in the energies.  If given
            as a `bool`, use `self.chempot`.  If a `float` then use
            this as the chemical potential.  Default value is `False`.

        Returns
        -------
        eigenvalues : numpy.ndarray
            Eigenvalues of the matrix.
        eigenvectors : numpy.ndarray
            Eigenvectors of the matrix.
        """

        matrix = self.matrix(physical, chempot=chempot, out=out)

        if self.hermitian:
            w, v = np.linalg.eigh(matrix)
        else:
            w, v = np.linalg.eig(matrix)

        return w, v

    def diagonalise_matrix_with_projection(self, physical, chempot=False, out=None):
        """
        Diagonalise the dense matrix representation of the Lehmann
        representation, and project the eigenvectors back to the
        physical space.

        Parameters
        ----------
        physical : numpy.ndarray
            Physical space part of the matrix.
        chempot : bool or float, optional
            Include the chemical potential in the energies.  If given
            as a `bool`, use `self.chempot`.  If a `float` then use
            this as the chemical potential.  Default value is `False`.

        Returns
        -------
        eigenvalues : numpy.ndarray
            Eigenvalues of the matrix.
        eigenvectors : numpy.ndarray
            Eigenvectors of the matrix, projected into the physical
            space.
        """

        w, v = self.diagonalise_matrix(physical, chempot=chempot, out=out)

        if self.hermitian:
            v = v[: self.nphys]
        else:
            vl = v[: self.nphys]
            vr = np.linalg.inv(v).T.conj()[: self.nphys]
            v = (vl, vr)

        return w, v

    def weights(self, occupancy=1):
        """
        Get the weights of the residues in the Lehmann representation.

        Parameters
        ----------
        occupancy : int or float, optional
            Occupancy of the states.  Default value is `1`.

        Returns
        -------
        weights : numpy.ndarray
            Weights of the states.
        """

        if self.hermitian:
            wt = np.sum(self.couplings**2, axis=0) * occupancy
        else:
            wt = np.sum(self.couplings[0] * self.couplings[1].conj(), axis=0) * occupancy

        return wt

    def as_orbitals(self, occupancy=1, mo_coeff=None):
        """
        Convert the Lehmann representation to an orbital representation.

        Parameters
        ----------
        occupancy : int or float, optional
            Occupancy of the states.  Default value is `1`.
        mo_coeff : numpy.ndarray, optional
            Molecular orbital coefficients.  If given, the orbitals
            will be rotated into the basis of these coefficients.
            Default value is `None`.

        Returns
        -------
        orb_energy : numpy.ndarray
            Orbital energies.
        orb_coeff : numpy.ndarray
            Orbital coefficients.
        orb_occ : numpy.ndarray
            Orbital occupancies.
        """

        if not self.hermitian:
            raise NotImplementedError

        orb_energy = self.energies

        if mo_coeff is not None:
            orb_coeff = np.dot(mo_coeff, self.couplings)
        else:
            orb_coeff = self.couplings

        orb_occ = np.zeros_like(orb_energy)
        orb_occ[orb_energy < self.chempot] = self.occupied().weights(occupancy=occupancy)

        return orb_energy, orb_coeff, orb_occ

    @property
    def hermitian(self):
        """Boolean flag for the Hermiticity."""

        return not isinstance(self.couplings, tuple)

    def _unpack_couplings(self):
        if self.hermitian:
            couplings_l = couplings_r = self.couplings
        else:
            couplings_l, couplings_r = self.couplings

        return couplings_l, couplings_r

    def _mask(self, mask, deep=True):
        """Return a part of the Lehmann representation using a mask."""

        if deep:
            energies = self.energies[mask].copy()
            if self.hermitian:
                couplings = self.couplings[:, mask].copy()
            else:
                couplings = (
                    self.couplings[0][:, mask].copy(),
                    self.couplings[1][:, mask].copy(),
                )
        else:
            energies = self.energies[mask]
            couplings = self.couplings[:, mask]

        return self.__class__(energies, couplings, chempot=self.chempot)

    def occupied(self, deep=True):
        """Return the occupied part of the Lehmann representation."""

        return self._mask(self.energies < self.chempot)

    def virtual(self, deep=True):
        """Return the virtual part of the Lehmann representation."""

        return self._mask(self.energies >= self.chempot)

    def copy(self, chempot=None, deep=True):
        """Return a copy with optionally updated chemical potential."""

        if chempot is None:
            chempot = self.chempot

        if deep:
            energies = self.energies.copy()
            if self.hermitian:
                couplings = self.couplings.copy()
            else:
                couplings = (self.couplings[0].copy(), self.couplings[1].copy())
        else:
            energies = self.energies
            couplings = self.couplings

        return self.__class__(energies, couplings, chempot=chempot)

    @property
    def nphys(self):
        """Number of physical degrees of freedom."""
        return self._unpack_couplings()[0].shape[0]

    @property
    def naux(self):
        """Number of auxiliary degrees of freedom."""
        return self._unpack_couplings()[0].shape[1]

    def __add__(self, other):
        """Combine two Lehmann representations."""

        if self.nphys != other.nphys:
            raise ValueError("Number of physical degrees of freedom do not match.")

        if self.chempot != other.chempot:
            raise ValueError("Chemical potentials do not match.")

        energies = np.concatenate((self.energies, other.energies))

        couplings_a_l, couplings_a_r = self._unpack_couplings()
        couplings_b_l, couplings_b_r = other._unpack_couplings()

        if self.hermitian:
            couplings = np.concatenate((couplings_a_l, couplings_b_l), axis=1)
        else:
            couplings = (
                np.concatenate((couplings_a_l, couplings_b_l), axis=1),
                np.concatenate((couplings_a_r, couplings_b_r), axis=1),
            )

        return self.__class__(energies, couplings, chempot=self.chempot)
