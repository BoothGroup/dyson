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

    def chebyshev_moment(self, order, scaling=None):
        """
        Get a spectral Chebyshev moment (or range of moments) of the
        Lehmann representation.

        Parameters
        ----------
        order : int or iterable of int
            Order(s) to calculate
        scaling : tuple of float
            Scaling parameters, such that the energy scale of the
            Lehmann representation is scaled as
            `(energies - scaling[1]) / scaling[0]`.  If `None`, the
            scaling paramters are computed as
            `(max(energies) - min(energies)) / (2.0 - 1e-3)` and
            `(max(energies) + min(energies)) / 2.0`, respectively.

        Returns
        -------
        chebyshev : numpy.ndarray
            Spectral Chebyshev moments, if `order` is an `int` then the
            moment is a 2D matrix, and if `order` is an `iterable` then
            it is a 3D matrix enumerating the orders.
        """

        if scaling is not None:
            a, b = scaling
        else:
            emin = min(self.energies)
            emax = max(self.energies)
            a = (emax - emin) / (2.0 - 1e-3)
            b = (emax + emin) / 2.0

        nmoms = set((order,) if isinstance(order, int) else order)
        nmom_max = max(nmoms)

        couplings_l, couplings_r = self._unpack_couplings()
        energies_scaled = (self.energies - b) / a

        c = np.zeros((nmom_max + 1, self.nphys, self.naux), dtype=self.dtype)
        c[0] = couplings_l
        c[1] = couplings_l * energies_scaled

        for i in range(2, nmom_max + 1):
            c[i] = 2 * energies_scaled * c[i - 1] - c[i - 2]

        c = c[list(nmoms)]
        chebyshev = np.einsum("qk,npx->npq", couplings_r, c)

        return chebyshev

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

    def as_static_potential(self, mo_energy, eta=1e-2):
        """
        Convert the Lehmann representation to a static potential, for
        example for us in qsGW when the Lehmann representation is of a
        self-energy.

        Parameters
        ----------
        mo_energy : numpy.ndarray
            Molecular orbital energies.
        eta : float, optional
            Broadening parameter.  Default value is `1e-2`.

        Returns
        -------
        static_potential : numpy.ndarray
            Static potential.
        """

        energies = self.energies + np.sign(self.energies - self.chempot) * 1.0j * eta
        denom = mo_energy[:, None] - energies[None, :]

        couplings_l, couplings_r = self._unpack_couplings()

        static_potential = np.einsum("pk,qk,pk->pq", couplings_l, couplings_r, 1.0 / denom).real
        static_potential = 0.5 * (static_potential + static_potential.T)

        return static_potential

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

    @property
    def dtype(self):
        return np.result_type(self.energies, self.couplings)
