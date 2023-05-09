"""
GW expressions.
"""

import numpy as np
from pyscf import agf2, ao2mo, lib

from dyson import util
from dyson.expressions import BaseExpression


@util.inherit_docstrings
class GW_Dyson(BaseExpression):
    """
    GW expressions without a non-Dyson approximation.
    """

    hermitian = True
    polarizability = "drpa"

    def __init__(self, *args, **kwargs):
        BaseExpression.__init__(self, *args, **kwargs)

        try:
            from momentGW import GW

            self._gw = GW(self.mf)
            self._gw.mo_occ = self.mo_occ
            self._gw.mo_coeff = self.mo_coeff
            self._gw.mo_energy = self.mo_energy
            self._gw.polarizability = self.polarizability
        except ImportError as e:
            raise ImportError("momentGW is required for GW expressions.")

    def get_static_part(self):
        static = self._gw.build_se_static()

        return static

    def apply_hamiltonian(self, vector, static=None):
        # From Bintrim & Berkelbach

        if static is None:
            static = self.get_static_part()

        i = slice(None, self.nocc)
        a = slice(self.nocc, self.nmo)
        ija = slice(self.nmo, self.nmo + self.nocc * self.nocc * self.nvir)
        iab = slice(self.nmo + self.nocc * self.nocc * self.nvir, None)

        Lpq = self._gw.ao2mo(self.mo_coeff)
        Lia = Lpq[:, i, a]
        Lai = Lpq[:, a, i]
        Lij = Lpq[:, i, i]
        Lab = Lpq[:, a, a]

        nocc, nvir = Lia.shape[1:]

        vi = vector[i]
        va = vector[a]
        vija = vector[ija].reshape(nocc, nocc, nvir)
        viab = vector[iab].reshape(nocc, nvir, nvir)

        eija = lib.direct_sum("i+j-a->ija", self.mo_energy[i], self.mo_energy[i], self.mo_energy[a])
        eiab = lib.direct_sum("a+b-i->iab", self.mo_energy[a], self.mo_energy[a], self.mo_energy[i])

        r = np.zeros_like(vector)

        if self.polarizability == "dtda":
            r[i] += lib.einsum("ij,j->i", static[i, i], vi)
            r[i] += lib.einsum("ib,b->i", static[i, a], va)
            r[i] += lib.einsum("Qik,Qcl,klc->i", Lij, Lai, vija)
            r[i] += lib.einsum("Qid,Qkc,kcd->i", Lia, Lia, viab)

            r[a] += lib.einsum("aj,j->a", static[a, i], vi)
            r[a] += lib.einsum("ab,b->a", static[a, a], va)
            r[a] += lib.einsum("Qak,Qcl,klc->a", Lai, Lai, vija)
            r[a] += lib.einsum("Qad,Qkc,kcd->a", Lab, Lia, viab)

            r[ija] += lib.einsum("Qki,Qaj,k->ija", Lij, Lai, vi).ravel()
            r[ija] += lib.einsum("Qbi,Qaj,b->ija", Lai, Lai, va).ravel()
            r[ija] += lib.einsum("ija,ija->ija", eija, vija).ravel()
            r[ija] -= lib.einsum("Qja,Qcl,ilc->ija", Lia, Lai, vija).ravel()

            r[iab] += lib.einsum("Qjb,Qia,j->iab", Lia, Lia, vi).ravel()
            r[iab] += lib.einsum("Qcb,Qia,c->iab", Lab, Lia, va).ravel()
            r[iab] += lib.einsum("iab,iab->iab", eiab, viab).ravel()
            r[iab] += lib.einsum("Qai,Qkc,kcb->iab", Lai, Lia, viab).ravel()

        elif self.polarizability == "drpa":
            raise NotImplementedError

        return r

    def diagonal(self, static=None):
        # From Bintrim & Berkelbach

        if static is None:
            static = self.get_static_part()

        i = slice(None, self.nocc)
        a = slice(self.nocc, self.nmo)
        ija = slice(self.nmo, self.nmo + self.nocc * self.nocc * self.nvir)
        iab = slice(self.nmo + self.nocc * self.nocc * self.nvir, None)

        Lpq, Lia = self._gw.ao2mo(self.mo_coeff)
        Lia = Lpq[:, i, a]
        Lai = Lpq[:, a, i]
        Lij = Lpq[:, i, i]
        Lab = Lpq[:, a, a]

        nocc, nvir = Lia.shape[1:]

        eija = lib.direct_sum("i+j-a->ija", self.mo_energy[i], self.mo_energy[i], self.mo_energy[a])
        eiab = lib.direct_sum("a+b-i->iab", self.mo_energy[a], self.mo_energy[a], self.mo_energy[i])

        diag = np.zeros((self.nmo + eija.size + eiab.size,))

        if self.polarizability == "dtda":
            diag[i] += np.diag(static[i, i])

            diag[a] += np.diag(static[a, a])

            diag[ija] += eija.ravel()
            diag[ija] -= lib.einsum("Qja,Qaj,ii->ija", Lia, Lai, np.eye(nocc)).ravel()

            diag[iab] += eiab.ravel()
            diag[iab] += lib.einsum("Qai,Qia,bb->iab", Lai, Lia, np.eye(nvir)).ravel()

        elif self.polarizability == "drpa":
            raise NotImplementedError

        return diag

    def get_wavefunction(self, orb):
        nija = self.nocc * self.nocc * self.nvir
        nabi = self.nocc * self.nvir * self.nvir

        r = np.zeros((self.nmo + nija + nabi,))
        r[orb] = 1.0

        return r

    def build_se_moments(self, nmom):
        Lpq, Lia = self._gw.ao2mo(self.mo_coeff)
        moments = self._gw.build_se_moments(nmom, Lpq, Lia)

        return moments


GW = {
    "Dyson": GW_Dyson,
}
