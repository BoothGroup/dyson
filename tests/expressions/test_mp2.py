"""
Tests for MP2.
"""

import unittest
import pytest

from pyscf import gto, scf, adc, agf2, lib
import numpy as np
import scipy.linalg

from dyson import util, Lehmann, NullLogger
from dyson import MBLSE, MixedMBLSE, DensityRelaxation, SelfConsistent, Davidson
from dyson.expressions import MP2


@pytest.mark.regression
class MP2_Tests(unittest.TestCase):
    """
    Test the `MP2` expressions.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="cc-pvdz", verbose=0)
        mf = scf.RHF(mol).run()
        gf2 = agf2.AGF2(mf)
        gf2.conv_tol_rdm1 = gf2.conv_tol_nelec = gf2.conv_tol = 1e-10
        gf2.kernel()
        cls.mf, cls.gf2 = mf, gf2

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.gf2

    def test_agf2(self):
        # Tests AGF2 implementation using `dyson`, as done in
        # `examples/30-agf2.py`.

        mf = self.mf

        def get_fock(rdm1_mo):
            rdm1_ao = np.linalg.multi_dot((mf.mo_coeff, rdm1_mo, mf.mo_coeff.T))
            fock_ao = mf.get_fock(dm=rdm1_ao)
            fock_mo = np.linalg.multi_dot((mf.mo_coeff.T, fock_ao, mf.mo_coeff))
            return fock_mo

        diis = lib.diis.DIIS()

        def get_se(gf, se_prev=None):
            mo_energy, mo_coeff, mo_occ = gf.as_orbitals(mo_coeff=mf.mo_coeff, occupancy=2)
            fock = get_fock(gf.occupied().moment(0) * 2)

            mp2 = MP2["Dyson"](mf, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
            th, tp = mp2.build_se_moments(2)
            th = lib.einsum("...ij,pi,qj->...pq", th, gf.couplings, gf.couplings)
            tp = lib.einsum("...ij,pi,qj->...pq", tp, gf.couplings, gf.couplings)
            th, tp = diis.update(np.array([th, tp]), xerr=None)

            solverh = MBLSE(fock, th, log=NullLogger())
            solverp = MBLSE(fock, tp, log=NullLogger())
            solver = MixedMBLSE(solverh, solverp)
            solver.kernel()

            return solver.get_self_energy()

        gf = Lehmann(mf.mo_energy, np.eye(mf.mo_energy.size))

        solver = SelfConsistent(
                get_se,
                gf,
                relax_solver=DensityRelaxation,
                get_fock=get_fock,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        solver.kernel()

        ip1 = -solver.get_greens_function().occupied().energies[-1]
        ea1 = solver.get_greens_function().virtual().energies[0]

        ip2 = -self.gf2.gf.get_occupied().energy[-1]
        ea2 = self.gf2.gf.get_virtual().energy[0]

        self.assertAlmostEqual(ip1, ip2, 8)
        self.assertAlmostEqual(ea1, ea2, 8)

    def test_ip_adc2(self):
        mf = self.mf

        mp2 = MP2["1h"](mf)
        static = mp2.get_static_part()
        diag = mp2.diagonal(static=static)
        matvec = lambda v: mp2.apply_hamiltonian(v, static=static)

        solver = Davidson(matvec, diag, nroots=5, nphys=mp2.nocc, log=NullLogger())
        solver.conv_tol = 1e-10
        solver.kernel()
        ip1 = -solver.get_greens_function().energies[-3:][::-1]

        adc2 = adc.ADC(mf)
        ip2 = adc2.kernel(nroots=5)[0]

        self.assertAlmostEqual(ip1[0], ip2[0], 8)
        self.assertAlmostEqual(ip1[1], ip2[1], 8)
        self.assertAlmostEqual(ip1[2], ip2[2], 8)

    def test_ea_adc2(self):
        mf = self.mf

        mp2 = MP2["1p"](mf)
        static = mp2.get_static_part()
        diag = mp2.diagonal(static=static)
        matvec = lambda v: mp2.apply_hamiltonian(v, static=static)

        solver = Davidson(matvec, diag, nroots=5, nphys=mp2.nocc, log=NullLogger())
        solver.conv_tol = 1e-10
        solver.kernel()
        ea1 = solver.get_greens_function().energies[:3]

        adc2 = adc.ADC(mf)
        adc2.method_type = "ea"
        ea2 = adc2.kernel(nroots=5)[0]

        self.assertAlmostEqual(ea1[0], ea2[0], 8)
        self.assertAlmostEqual(ea1[1], ea2[1], 8)
        self.assertAlmostEqual(ea1[2], ea2[2], 8)


if __name__ == "__main__":
    unittest.main()
