import numpy as np
from pyscf import gto, scf, lo

from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from dyson import default_log, util
from dyson.expressions.expression import BaseExpression

from time import time

def _dmrg_constructor(op_string):

    @util.inherit_docstrings
    class _DMRG(BaseExpression):
        """
        DMRG expressions.
        """

        hermitian = True
        
        def __init__(self, *args, bond_dims=None, noises=None, thrds=None, **kwargs):

            if len(args):
                # if nelec is not None:
                #     raise ValueError(
                #         "nelec keyword argument only valid when mean-field object is not " "passed."
                #     )
                BaseExpression.__init__(self, *args, **kwargs)
                self.op_string = op_string

                # Lowdin orthogonalization
                mo_coeff_old = self.mf.mo_coeff.copy()
                self.mf.mo_coeff = lo.orth.lowdin(self.mf.get_ovlp())


                # DMRG
                ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(self.mf, ncore=0, ncas=None, g2e_symm=1)
                self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
                self.driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

                self.bond_dims = [100] * 4 + [50] * 4 if bond_dims is None else bond_dims
                self.noises = [1e-4] * 4 + [1e-5] * 4 + [0] * 4 if noises is None else noises
                self.thrds = [1e-10] * 8 if thrds is None else thrds

                self.mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, integral_cutoff=1E-8, iprint=1)
                self.ket = self.driver.get_random_mps(tag="KET", bond_dim=100, nroots=1)
                self.energy = self.driver.dmrg(self.mpo, self.ket, n_sweeps=20, bond_dims=self.bond_dims, noises=self.noises, thrds=self.thrds, iprint=1)




        def get_wavefunction(self, orb):
            kets = []
            for xd, spin in zip(self.op_string, 'AB'):
                dmpo = self.driver.get_site_mpo(op=xd, site_index=orb, iprint=0)
                dket = self.driver.get_random_mps(tag="D%sKET-%s-H0" % (spin, orb), bond_dim=8,
                    center=self.ket.center, target=dmpo.op.q_label + self.ket.info.target)
                self.driver.multiply(dket, dmpo, self.ket, n_sweeps=10, bond_dims=[8], thrds=self.thrds, iprint=0)
                kets.append(dket)
            return kets


        def apply_hamiltonian(self, kets):
            hdkets = []
            for i, (xd, spin) in enumerate(zip(self.op_string, 'AB')):
                hdket = self.driver.copy_mps(kets[i], tag="HD%sKET-%s-%s" % (spin, i + 1, time()))
                #print(hdket, kets[i])
                self.driver.multiply(hdket, self.mpo, kets[i], n_sweeps=10, bond_dims=[8], thrds=[1E-10] * 10, iprint=0)
                hdkets.append(hdket)
            return hdkets

        def apply_hamiltonian_left(self, ket):
            pass

        def dot(self, bras, kets):
            impo = self.driver.get_identity_mpo()
            ret = np.array(np.sum([self.driver.expectation(bras[s], impo, kets[s]) for s in [0, 1]]))
            return ret

    return _DMRG

DMRG_1h = _dmrg_constructor('dD')
DMRG_1p = _dmrg_constructor('cC')

DMRG = {
    "1h": DMRG_1h,
    "1p": DMRG_1p,
}


def get_hn_ket(ket, n, xj, oper='cC'):
    kets = []
    for xd, spin in zip(oper, "AB"):
        dmpo = driver.get_site_mpo(op=xd, site_index=xj, iprint=0)
        dket = driver.get_random_mps(tag="D%sKET-%s-H0" % (spin, xj), bond_dim=40,
            center=ket.center, target=dmpo.op.q_label + ket.info.target)
        driver.multiply(dket, dmpo, ket, n_sweeps=10, bond_dims=[40], thrds=[1E-10] * 10, iprint=0)
        for i in range(n):
            hdket = driver.copy_mps(dket, tag="D%sKET-%s-H%s" % (spin, xj, i + 1))
            driver.multiply(hdket, self.hamiltonian_mpo, dket, n_sweeps=10, bond_dims=[40], thrds=[1E-10] * 10, iprint=0)
            dket = hdket
        kets.append(dket)
    return kets