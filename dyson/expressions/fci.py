"""
FCI expressions.
"""

import numpy as np
from pyscf import ao2mo, fci, lib

from dyson import default_log, util
from dyson.expressions import BaseExpression


def _fci_constructor(δalph, δbeta, func_sq, sign):
    """Construct FCI expressions classes for a given change in the
    number of alpha and beta electrons.
    """

    @util.inherit_docstrings
    class _FCI(BaseExpression):
        """
        FCI expressions.
        """

        hermitian = True

        def __init__(
            self,
            *args,
            h1e=None,
            h2e=None,
            e_ci=None,
            c_ci=None,
            chempot=None,
            nelec=None,
            **kwargs,
        ):
            if len(args):
                if nelec is not None:
                    raise ValueError(
                        "nelec keyword argument only valid when mean-field object is not " "passed."
                    )
                BaseExpression.__init__(self, *args, **kwargs)
            else:
                # Allow initialisation without MF object
                if h1e is None or h2e is None:
                    raise ValueError(
                        "h1e and h2e keyword arguments are required to initialise FCI "
                        "without mean-field object."
                    )
                self.log = kwargs.get("log", default_log)
                self.mf = None
                self._nelec = nelec
                self._nmo = h1e.shape[0]

            if e_ci is None:
                if h1e is None:
                    h1e = np.linalg.multi_dot(
                        (
                            self.mo_coeff.T,
                            self.mf.get_hcore(),
                            self.mo_coeff,
                        )
                    )
                if h2e is None:
                    h2e = ao2mo.kernel(self.mf._eri, self.mo_coeff)

                ci = fci.direct_spin1.FCI()
                ci.verbose = 0
                e_ci, c_ci = ci.kernel(h1e, h2e, self.nmo, (self.nalph, self.nbeta))

            assert e_ci is not None
            assert c_ci is not None
            assert h1e is not None
            assert h2e is not None

            self.e_ci = e_ci
            self.c_ci = c_ci
            self.chempot = e_ci if chempot is None else chempot

            self.link_index = (
                fci.cistring.gen_linkstr_index_trilidx(range(self.nmo), self.nalph + δalph),
                fci.cistring.gen_linkstr_index_trilidx(range(self.nmo), self.nbeta + δbeta),
            )

            self.hamiltonian = fci.direct_spin1.absorb_h1e(
                h1e,
                h2e,
                self.nmo,
                (self.nalph + δalph, self.nbeta + δbeta),
                0.5,
            )

            self.diag = fci.direct_spin1.make_hdiag(
                h1e,
                h2e,
                self.nmo,
                (self.nalph + δalph, self.nbeta + δbeta),
            )

        def diagonal(self):
            return sign * self.diag

        def apply_hamiltonian(self, vector):
            hvec = fci.direct_spin1.contract_2e(
                self.hamiltonian,
                vector,
                self.nmo,
                (self.nalph + δalph, self.nbeta + δbeta),
                self.link_index,
            )
            hvec -= self.chempot * vector

            return sign * hvec.ravel()

        def get_wavefunction(self, orb):
            wfn = func_sq(self.c_ci, self.nmo, (self.nalph, self.nbeta), orb)
            return wfn.ravel()

        # Override properties to allow initialisation without a mean-field

        @property
        def nmo(self):
            if self.mf is None:
                return self._nmo
            return self.mo_coeff.shape[-1]

        @property
        def nocc(self):
            if self.mf is None:
                return self._nelec // 2
            return np.sum(self.mo_occ > 0)

        @property
        def nvir(self):
            if self.mf is None:
                return self.nmo - self.nocc
            return np.sum(self.mo_occ == 0)

    return _FCI


FCI_1h = _fci_constructor(-1, 0, fci.addons.des_a, -1)

FCI_1p = _fci_constructor(1, 0, fci.addons.cre_a, 1)

FCI = {
    "1h": FCI_1h,
    "1p": FCI_1p,
}
