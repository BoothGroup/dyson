"""
MP2 expressions.
"""

import numpy as np
from pyscf import agf2, ao2mo, lib

from dyson import util
from dyson.expressions import BaseExpression

# TODO only separate for non-Dyson


def _mp2_constructor(occ, vir):
    """Construct MP2 expressions classes for a given occupied and
    virtual mask. These classes use a non-Dyson approximation.
    """

    @util.inherit_docstrings
    class _MP2(BaseExpression):
        """
        MP2 expressions with non-Dyson approximation.
        """

        hermitian = False

        def __init__(self, *args, **kwargs):
            BaseExpression.__init__(self, *args, **kwargs)

            self.ijka = self._integrals_for_hamiltonian()

        def get_static_part(self):
            co = self.mo_coeff[:, occ(self)]
            cv = self.mo_coeff[:, vir(self)]
            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]

            e_iajb = lib.direct_sum("i-a+j-b->iajb", eo, ev, eo, ev)

            iajb = ao2mo.incore.general(self.mf._eri, (co, cv, co, cv), compact=False)
            iajb = iajb.reshape([x.shape[-1] for x in (co, cv, co, cv)])
            t2 = iajb / e_iajb
            iajb = 2 * iajb - iajb.transpose(0, 3, 2, 1)

            h1 = lib.einsum("iakb,jakb->ij", iajb, t2) * 0.5
            h1 += h1.T
            h1 += np.diag(eo)  # FIXME or C* F C?

            return h1

        def _integrals_for_hamiltonian(self):
            c = self.mo_coeff[:, occ(self)]
            e = self.mo_energy[occ(self)]
            p = slice(None, e.size)
            a = slice(e.size, None)

            co = self.mo_coeff[:, occ(self)]
            cv = self.mo_coeff[:, vir(self)]

            ijka = ao2mo.incore.general(self.mf._eri, (co, co, co, cv), compact=False)
            ijka = ijka.reshape([x.shape[-1] for x in (co, co, co, cv)])

            return ijka

        def apply_hamiltonian(self, vector, static=None):
            if static is None:
                static = self.get_static_part()

            e = self.mo_energy[occ(self)]
            p = slice(None, e.size)
            a = slice(e.size, None)

            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]
            e_jka = lib.direct_sum("j+k-a->jka", eo, eo, ev)
            ijka = self.ijka

            r = np.zeros_like(vector)
            r[p] += np.dot(static, vector[p])
            r[p] += lib.einsum("ijka,jka->i", ijka, vector[a].reshape(e_jka.shape))
            r[a] += lib.einsum("ijka,i->jka", ijka, vector[p]).ravel() * 2.0
            r[a] -= lib.einsum("ikja,i->jka", ijka, vector[p]).ravel()
            r[a] += vector[a] * e_jka.ravel()

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

            r = np.zeros((nocc + nija,))
            r[orb] = 1.0

            return r

        def build_se_moments(self, nmom):
            eo = self.mo_energy[occ(self)]
            ev = self.mo_energy[vir(self)]
            ijka = self.ijka

            t = []
            for n in range(nmom):
                tn = 0
                for j in range(eo.size):
                    vl = ijka[:, j]
                    vr = 2.0 * ijka[:, j] - ijka[:, :, j]
                    eka = eo[j] + lib.direct_sum("k-a->ka", eo, ev)
                    tn += lib.einsum("ika,jka,ka->ij", vl, vr, eka**n)
                t.append(tn)

            return np.array(t)

    return _MP2


@util.inherit_docstrings
class MP2_Dyson(BaseExpression):
    """
    MP2 expressions without non-Dyson approximation.
    """

    def __init__(self, *args, **kwargs):
        BaseExpression.__init__(self, *args, **kwargs)

        self._agf2 = agf2.ragf2_slow.RAGF2(
            self.mf,
            mo_energy=self.mo_energy,
            mo_coeff=self.mo_coeff,
            mo_occ=self.mo_occ,
        )

    def get_static_part(self):
        raise NotImplementedError  # TODO

    def apply_hamiltonian(self, vector, static=None):
        raise NotImplementedError  # TODO

    def diagonal(self, static=None):
        raise NotImplementedError  # TODO

    def get_wavefunction(self, orb):
        nija = self.nocc * self.nocc * self.nvir
        niab = self.nocc * self.nvir * self.nvir

        r = np.zeros((self.nmo + nija + niab,))
        r[orb] = 1.0

        return r

    def build_se_moments(self, nmom):
        eo = self.mo_energy[: self.nocc]
        ev = self.mo_energy[self.nocc :]
        c = self.mo_coeff
        co = self.mo_coeff[:, : self.nocc]
        cv = self.mo_coeff[:, self.nocc :]

        xija = ao2mo.incore.general(self.mf._eri, (c, co, co, cv), compact=False)
        xija = xija.reshape([x.shape[-1] for x in (c, co, co, cv)])

        xabi = ao2mo.incore.general(self.mf._eri, (c, cv, cv, co), compact=False)
        xabi = xabi.reshape([x.shape[-1] for x in (c, cv, cv, co)])

        th = np.zeros((nmom, self.nmo, self.nmo))
        tp = np.zeros((nmom, self.nmo, self.nmo))
        for n in range(nmom):
            for i in range(eo.size):
                vl = xija[:, i]
                vr = 2.0 * xija[:, i] - xija[:, :, i]
                eja = eo[i] + lib.direct_sum("j-a->ja", eo, ev)
                th[n] += lib.einsum("xja,yja,ja->xy", vl, vr, eja**n)

            for a in range(ev.size):
                vl = xabi[:, a]
                vr = 2.0 * xabi[:, a] - xabi[:, :, a]
                ebi = ev[a] + lib.direct_sum("b-i->bi", ev, eo)
                tp[n] += lib.einsum("xbi,ybi,bi->xy", vl, vr, ebi**n)

        return th, tp


MP2_1h = _mp2_constructor(lambda self: self.mo_occ > 0, lambda self: self.mo_occ == 0)

MP2_1p = _mp2_constructor(lambda self: self.mo_occ == 0, lambda self: self.mo_occ > 0)

MP2 = {
    "Dyson": MP2_Dyson,
    "1h": MP2_1h,
    "1p": MP2_1p,
}
