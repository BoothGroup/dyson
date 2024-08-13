"""
Expression base class.
"""

import numpy as np

from dyson import default_log, init_logging


class BaseExpression:
    """
    Base class for all expressions.
    """

    hermitian = True

    def __init__(self, mf, mo_energy=None, mo_coeff=None, mo_occ=None, log=None):
        self.log = log or default_log
        # init_logging(self.log)
        # self.log.info("")
        # self.log.info("%s", self.__class__.__name__)
        # self.log.info("%s", "*" * len(self.__class__.__name__))

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.mf = mf
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

    def apply_hamiltonian(self, vector):
        """Apply the Hamiltonian to a trial vector.

        Parameters
        ----------
        vector : numpy.ndarray
            Vector to apply Hamiltonian to.

        Returns
        -------
        output : numpy.ndarray
            Output vector.
        """

        raise NotImplementedError

    def apply_hamiltonian_left(self, vector):
        """Apply the Hamiltonian to a trial vector on the left.

        Parameters
        ----------
        vector : numpy.ndarray
            Vector to apply Hamiltonian to.

        Returns
        -------
        output : numpy.ndarray
            Output vector.
        """

        raise NotImplementedError

    def diagonal(self):
        """Get the diagonal of the Hamiltonian.

        Returns
        -------
        diag : numpy.ndarray
            Diagonal of the Hamiltonian.
        """

        raise NotImplementedError

    def get_wavefunction(self, orb):
        """Obtain the wavefunction as a vector, for a given orbital.

        Parameters
        ----------
        orb : int
            Orbital index.

        Returns
        -------
        wfn : numpy.ndarray
            Wavefunction vector.
        """

        raise NotImplementedError

    def get_wavefunction_bra(self, orb):
        return self.get_wavefunction(orb)

    def get_wavefunction_ket(self, orb):
        return self.get_wavefunction(orb)

    def build_gf_moments(self, nmom, store_vectors=True, left=False):
        """Build moments of the Green's function.

        Parameters
        ----------
        nmom : int or tuple of int
            Number of moments to compute.
        store_vectors : bool, optional
            Store all vectors on disk rather than storing them all
            ahead of time.  With `store_vectors=True`, the memory
            overhead of the vectors is O(N) larger.  With
            `store_vectors=False`, the CPU overhead of the vectors is
            O(N) larger.  Default value is `True`.
        left : bool, optional
            Use the left-handed Hamiltonian application instead of the
            right-handed one.  Default value is `False`.

        Returns
        -------
        t : numpy.ndarray
            Moments of the Green's function.
        """

        t = np.zeros((nmom, self.nphys, self.nphys))

        if left:
            get_wavefunction_bra = self.get_wavefunction_ket
            get_wavefunction_ket = self.get_wavefunction_bra
            apply_hamiltonian = self.apply_hamiltonian_left
        else:
            get_wavefunction_bra = self.get_wavefunction_bra
            get_wavefunction_ket = self.get_wavefunction_ket
            apply_hamiltonian = self.apply_hamiltonian

        if store_vectors:
            v = [get_wavefunction_bra(i) for i in range(self.nphys)]

        for i in range(self.nphys):
            u = get_wavefunction_ket(i)

            for n in range(nmom):
                for j in range(i if self.hermitian else 0, self.nphys):
                    if not store_vectors:
                        v = {j: get_wavefunction_bra(j)}

                    t[n, i, j] = np.dot(v[j], u)

                    if self.hermitian:
                        t[n, j, i] = t[n, i, j]

                if n != (nmom - 1):
                    u = apply_hamiltonian(u)

        if left:
            t = t.transpose(0, 2, 1).conj()

        return t

    def build_gf_chebyshev_moments(self, nmom, store_vectors=True, left=False, scaling=None):
        """Build moments of the Green's function using Chebyshev polynomials.

        Parameters
        ----------
        nmom : int or tuple of int
            Number of moments to compute.
        store_vectors : bool, optional
            Store all vectors on disk rather than storing them all
            ahead of time.  With `store_vectors=True`, the memory
            overhead of the vectors is O(N) larger.  With
            `store_vectors=False`, the CPU overhead of the vectors is
            O(N) larger.  Default value is `True`.
        left : bool, optional
            Use the left-handed Hamiltonian application instead of the
            right-handed one.  Default value is `False`.
        scaling : tuple of float
            Scaling parameters, such that the energy scale of the
            Lehmann representation is scaled as
            `(energies - scaling[1]) / scaling[0]`.  If `None`, the
            scaling paramters are computed as
            `(max(energies) - min(energies)) / (2.0 - 1e-3)` and
            `(max(energies) + min(energies)) / 2.0`, respectively.

        Returns
        -------
        t : numpy.ndarray
            Chebyshev moments of the Green's function.
        """

        if scaling is not None:
            a, b = scaling
        else:
            # Calculate the scaling parameters by the range of the
            # eigenvalues of the Hamiltonian. These can be approximated
            # using the diagonal of the Hamiltonian. A more effective
            # method would be to use the Lanczos or Davidson algorithms
            # to compute the extremum eigenvalues and pass them in as
            # an argument.
            diag = self.diagonal()
            emin = min(diag)
            emax = max(diag)
            a = (emax - emin) / (2.0 - 1e-3)
            b = (emax + emin) / 2.0

        t = np.zeros((nmom, self.nphys, self.nphys))

        if left:
            get_wavefunction_bra = self.get_wavefunction_ket
            get_wavefunction_ket = self.get_wavefunction_bra
            apply_hamiltonian = self.apply_hamiltonian_left
        else:
            get_wavefunction_bra = self.get_wavefunction_bra
            get_wavefunction_ket = self.get_wavefunction_ket
            apply_hamiltonian = self.apply_hamiltonian

        def apply_scaled_hamiltonian(v):
            # [(H - b) / a] v = H (v / a) - b (v / a)
            v_scaled = v / a
            return apply_hamiltonian(v_scaled) - b * v_scaled

        if store_vectors:
            v = [get_wavefunction_bra(i) for i in range(self.nphys)]

        for i in range(self.nphys):
            u = get_wavefunction_ket(i)

            for n in range(nmom):
                for j in range(i if self.hermitian else 0, self.nphys):
                    if not store_vectors:
                        v = {j: get_wavefunction_bra(j)}

                    t[n, i, j] = np.dot(v[j], u)

                    if self.hermitian:
                        t[n, j, i] = t[n, i, j]

                if n != (nmom - 1):
                    if n == 0:
                        # u_{1} = H u_{0}
                        u, u_prev = apply_scaled_hamiltonian(u), u
                    else:
                        # u_{i} = 2 H u_{i-1} - u_{i-2}
                        u, u_prev = 2.0 * apply_scaled_hamiltonian(u) - u_prev, u

        if left:
            t = t.transpose(1, 2).conj()

        return t

    def build_se_moments(self, nmom):
        """Build moments of the self-energy.

        Parameters
        ----------
        nmom : int or tuple of int
            Number of moments to compute.

        Returns
        -------
        t : numpy.ndarray
            Moments of the self-energy.
        """

        raise NotImplementedError

    @property
    def nmo(self):
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self):
        return np.sum(self.mo_occ > 0)

    @property
    def nvir(self):
        return np.sum(self.mo_occ == 0)

    @property
    def nalph(self):
        return self.nocc

    @property
    def nbeta(self):
        return self.nocc

    @property
    def nphys(self):
        return self.nmo
