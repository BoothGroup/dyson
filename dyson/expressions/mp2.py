"""
MP2 expressions.
"""

import numpy as np
from pyscf import agf2, ao2mo, lib

from dyson import util
from dyson.expressions import BaseExpression


def _mp2_constructor(occ, vir):
    """Construct MP2 expressions classes for a given occupied and
    virtual mask.
    """

    @util.inherit_docstrings
    class _MP2(BaseExpression):
        """
        MP2 expressions.
        """

        def __init__(self, *args, non_dyson=False, **kwargs):
            BaseExpression.__init__(self, *args, **kwargs)

            self.non_dyson = non_dyson

        def get_static_part(self):
            if self.non_dyson:
                c = self.mo_coeff[:, occ(self)]
                e = self.mo_energy[occ(self)]
            else:
                c = self.mo_coeff
                e = self.mo_energy

            co = self.mo_coeff[:, occ(self)]
            cv = self.mo_coeff[:, vir(self)]
            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]

            e_ia = lib.direct_sum("i-a->ia", eo, ev)
            e_ijab = lib.direct_sum("ij+ab->iajb", e_ia, e_ia)

            xajb = ao2mo.incore.general(self.mf._eri, (c, cv, co, cv), compact=False)
            xajb = xajb.reshape([x.shape[-1] for x in (c, cv, co, cv)])
            t2 = xajb / e_ijab
            xajb = 2 * xajb - xajb.transpose(0, 3, 2, 1)

            h1 = lib.einsum("xajb,yajb->xy", xajb, t2) * 0.5
            h1 += h1.T
            h1 += np.diag(e)  # FIXME or C* F C?

            return h1

        def apply_hamiltonian(self, vector, static=None, xija=None):
            if static is None:
                static = self.get_static_part()

            if self.non_dyson:
                c = self.mo_coeff[:, occ(self)]
                e = self.mo_energy[occ(self)]
            else:
                c = self.mo_coeff
                e = self.mo_energy
            p = slice(None, e.size)
            a = slice(e.size, None)

            co = self.mo_coeff[:, occ(self)]
            cv = self.mo_coeff[:, vir(self)]
            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]

            e_ija = lib.direct_sum("i+j-a->ija", eo, eo, ev)

            if xija is None:
                xija = ao2mo.incore.general(self.mf._eri, (c, co, co, cv))
                xija = xija.reshape([x.shape[-1] for x in (c, co, co, cv)])

            r = np.zeros_like(vector)
            r[p] += np.dot(static, vector[p])
            r[p] += lib.einsum("xija,ija->x", xija, vector[a])
            r[a] += lib.einsum("xija,x->ija", xija, vector[p]).ravel() * 2.0
            r[a] -= lib.einsum("xija,x->jia", xija, vector[a]).ravel()
            r[a] += (vector[a] * e_ija).ravel()

            return r

        def diagonal(self, static=None):
            if static is None:
                static = self.get_static_part()

            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]
            e_ija = lib.direct_sum("i+j-a->ija", eo, eo, ev)

            r = np.concatenate([np.diag(static), e_ija.ravel()])

            return r

        def get_wavefunction(self, orb):
            nocc = np.sum(occ(self))
            nvir = np.sum(vir(self))
            nija = nocc * nocc * nvir

            if self.non_dyson:
                n = nocc
            else:
                n = nocc + nvir

            r = np.zeros((n + nija,))
            r[orb] = 1.0

            return r

        def build_se_moments(self, nmom, xija=None):
            if self.non_dyson:
                c = self.mo_coeff[:, occ(self)]
            else:
                c = self.mo_coeff

            co = self.mo_coeff[:, occ(self)]
            cv = self.mo_coeff[:, vir(self)]
            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]

            if xija is None:
                xija = ao2mo.incore.general(self.mf._eri, (c, co, co, cv), compact=False)
                xija = xija.reshape([x.shape[-1] for x in (c, co, co, cv)])

            t = []
            for n in range(nmom):
                tn = 0
                for i in range(eo.size):
                    vl = xija[:, i]
                    vr = 2.0 * xija[:, i] - xija[:, :, i]
                    eja = eo[i] + lib.direct_sum("j-a->ja", eo, ev)
                    tn += lib.einsum("xja,yja,ja->xy", vl, vr, eja**n)
                t.append(tn)

            return np.array(t)

    return _MP2


MP2_1h = _mp2_constructor(lambda self: self.mo_occ > 0, lambda self: self.mo_occ == 0)

MP2_1p = _mp2_constructor(lambda self: self.mo_occ == 0, lambda self: self.mo_occ > 0)

MP2 = {
    "1h": MP2_1h,
    "1p": MP2_1p,
}
