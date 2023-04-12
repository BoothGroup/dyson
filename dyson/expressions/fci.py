"""
FCI expressions.
"""

import numpy as np
from pyscf import ao2mo, fci, lib

from dyson import util
from dyson.expressions import BaseExpression


def _fci_constructor(δalph, δbeta, func_sq):
    """Construct FCI expressions classes for a given change in the
    number of alpha and beta electrons.
    """

    @util.inherit_docstrings
    class _FCI(BaseExpression):
        """
        FCI expressions.
        """

        hermitian = True

        def __init__(self, *args, h1e=None, h2e=None, e_ci=None, c_ci=None, chempot=0.0, **kwargs):
            BaseExpression.__init__(self, *args, **kwargs)

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
            self.chempot = chempot

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
            return self.diag

        def apply_hamiltonian(self, vector):
            hvec = fci.direct_spin1.contract_2e(
                self.hamiltonian,
                vector,
                self.nmo,
                (self.nalph + δalph, self.nbeta + δbeta),
                self.link_index,
            )
            hvec -= self.chempot * vector

            return hvec.ravel()

        def get_wavefunction(self, orb):
            wfn = func_sq(self.c_ci, self.nmo, (self.nalph, self.nbeta), orb)
            return wfn.ravel()

    return _FCI


FCI_1h = _fci_constructor(-1, 0, fci.addons.des_a)

FCI_1p = _fci_constructor(1, 0, fci.addons.cre_a)

FCI = {
    "1h": FCI_1h,
    "1p": FCI_1p,
}
