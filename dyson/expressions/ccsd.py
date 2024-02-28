"""
EOM-CCSD expressions.
"""

import numpy as np
from pyscf import ao2mo, cc, lib

from dyson import util
from dyson.expressions import BaseExpression


class CCSD_1h(BaseExpression):
    """
    IP-EOM-CCSD expressions.
    """

    hermitian = False

    def __init__(self, *args, ccsd=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
        BaseExpression.__init__(self, *args, **kwargs)

        if ccsd is None:
            ccsd = cc.CCSD(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            ccsd.verbose = 0
        
        # Update amplitudes if provided as kwargs
        self.t1 = t1 if t1 is not None else ccsd.t1
        self.t2 = t2 if t2 is not None else ccsd.t2 
        self.l1 = l1 if l1 is not None else ccsd.l1
        self.l2 = l2 if l2 is not None else ccsd.l2

        # Solve CCSD if amplitudes are not provided
        if ccsd.t1 is None or ccsd.t2 is None:
            ccsd.kernel()
            self.t1 = ccsd.t1
            self.t2 = ccsd.t2
        if ccsd.l1 is None or ccsd.l2 is None:
            self.l1, self.l2 = ccsd.solve_lambda()

        self.eris = ccsd.ao2mo()
        self.imds = cc.eom_rccsd._IMDS(ccsd, eris=self.eris)
        self.imds.make_ip()

        self.eom = lambda: None
        self.eom.nmo = self.nmo
        self.eom.nocc = self.nocc
        self.eom.vector_to_amplitudes = cc.eom_rccsd.vector_to_amplitudes_ip
        self.eom.amplitudes_to_vector = cc.eom_rccsd.amplitudes_to_vector_ip
        self.eom.partition = None

    def diagonal(self):
        diag = -cc.eom_rccsd.ipccsd_diag(self.eom, imds=self.imds)
        return diag

    def apply_hamiltonian(self, vector):
        hvec = -cc.eom_rccsd.ipccsd_matvec(self.eom, vector, imds=self.imds)
        return hvec

    def apply_hamiltonian_left(self, vector):
        hvec = -cc.eom_rccsd.lipccsd_matvec(self.eom, vector, imds=self.imds)
        return hvec

    def get_wavefunction_bra(self, orb):
        t1 = self.t1
        t2 = self.t2
        l1 = self.l1
        l2 = self.l2

        if orb < self.nocc:
            v1 = np.eye(self.nocc)[orb]
            v1 -= lib.einsum("ie,e->i", l1, t1[orb])
            tmp = t2[orb] * 2.0
            tmp -= t2[orb].swapaxes(1, 2)
            v1 -= lib.einsum("imef,mef->i", l2, tmp)

            tmp = -lib.einsum("ijea,e->ija", l2, t1[orb])
            v2 = tmp * 2.0
            v2 -= tmp.swapaxes(0, 1)
            tmp = lib.einsum("ja,i->ija", l1, np.eye(self.nocc)[orb])
            v2 += tmp * 2.0
            v2 -= tmp.swapaxes(0, 1)

        else:
            v1 = l1[:, orb - self.nocc].copy()
            v2 = l2[:, :, orb - self.nocc] * 2.0
            v2 -= l2[:, :, :, orb - self.nocc]

        return self.eom.amplitudes_to_vector(v1, v2)

    def get_wavefunction_ket(self, orb):
        t1 = self.t1
        t2 = self.t2
        l1 = self.l1
        l2 = self.l2

        if orb < self.nocc:
            v1 = np.eye(self.nocc)[orb]
            v2 = np.zeros((self.nocc, self.nocc, self.nvir))
        else:
            v1 = t1[:, orb - self.nocc]
            v2 = t2[:, :, orb - self.nocc]

        return self.eom.amplitudes_to_vector(v1, v2)


class CCSD_1p(BaseExpression):
    """
    EA-EOM-CCSD expressions.
    """

    hermitian = False

    def __init__(self, *args, ccsd=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
        BaseExpression.__init__(self, *args, **kwargs)

        if ccsd is None:
            ccsd = cc.CCSD(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            ccsd.verbose = 0

        # Update amplitudes if provided as kwargs
        self.t1 = t1 if t1 is not None else ccsd.t1
        self.t2 = t2 if t2 is not None else ccsd.t2 
        self.l1 = l1 if l1 is not None else ccsd.l1
        self.l2 = l2 if l2 is not None else ccsd.l2

        # Solve CCSD if amplitudes are not provided
        if ccsd.t1 is None or ccsd.t2 is None:
            ccsd.kernel()
            self.t1 = ccsd.t1
            self.t2 = ccsd.t2
        if ccsd.l1 is None or ccsd.l2 is None:
            self.l1, self.l2 = ccsd.solve_lambda()

        self.eris = ccsd.ao2mo()
        self.imds = cc.eom_rccsd._IMDS(ccsd, eris=self.eris)
        self.imds.make_ea()

        self.eom = lambda: None
        self.eom.nmo = self.nmo
        self.eom.nocc = self.nocc
        self.eom.vector_to_amplitudes = cc.eom_rccsd.vector_to_amplitudes_ea
        self.eom.amplitudes_to_vector = cc.eom_rccsd.amplitudes_to_vector_ea
        self.eom.partition = None

    def diagonal(self):
        diag = cc.eom_rccsd.eaccsd_diag(self.eom, imds=self.imds)
        return diag

    def apply_hamiltonian(self, vector):
        hvec = cc.eom_rccsd.eaccsd_matvec(self.eom, vector, imds=self.imds)
        return hvec

    def apply_hamiltonian_left(self, vector):
        hvec = cc.eom_rccsd.leaccsd_matvec(self.eom, vector, imds=self.imds)
        return hvec

    def get_wavefunction_bra(self, orb):
        t1 = self.t1
        t2 = self.t2
        l1 = self.l1
        l2 = self.l2

        if orb < self.nocc:
            v1 = -l1[orb]
            v2 = -l2[orb] * 2.0
            v2 += l2[:, orb]

        else:
            v1 = np.eye(self.nvir)[orb - self.nocc]
            v1 -= lib.einsum("mb,m->b", l1, t1[:, orb - self.nocc])
            tmp = t2[:, :, :, orb - self.nocc] * 2.0
            tmp -= t2[:, :, orb - self.nocc]
            v1 -= lib.einsum("kmeb,kme->b", l2, tmp)

            tmp = -lib.einsum("ikba,k->iab", l2, t1[:, orb - self.nocc])
            v2 = tmp * 2.0
            v2 -= tmp.swapaxes(1, 2)
            tmp = lib.einsum("ib,a->iab", l1, np.eye(self.nvir)[orb - self.nocc])
            v2 += tmp * 2.0
            v2 -= tmp.swapaxes(1, 2)

        return self.eom.amplitudes_to_vector(v1, v2)

    def get_wavefunction_ket(self, orb):
        t1 = self.t1
        t2 = self.t2
        l1 = self.l1
        l2 = self.l2

        if orb < self.nocc:
            v1 = t1[orb]
            v2 = t2[orb]
        else:
            v1 = -np.eye(self.nvir)[orb - self.nocc]
            v2 = np.zeros((self.nocc, self.nvir, self.nvir))

        return -self.eom.amplitudes_to_vector(v1, v2)


CCSD = {
    "1h": CCSD_1h,
    "1p": CCSD_1p,
}
