"""
Tests for SelfConsistentField.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2, lib
import numpy as np

from dyson import util, SelfConsistentField, NullLogger
from dyson.lehmann import Lehmann


@pytest.mark.regression
class SelfConsistentField_Tests(unittest.TestCase):
    """
    Test the `SelfConsistentField` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy)
        se = agf2.AGF2(mf, nmom=(None, 2)).build_se().get_virtual()
        cls.mf, cls.mol = mf, mol
        cls.f, cls.se = f, se

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.mol, cls.f, cls.se

    def test_agf2(self):
        def get_fock(rdm1):
            rdm1_ao = np.linalg.multi_dot((self.mf.mo_coeff, rdm1, self.mf.mo_coeff.T))
            fock_ao = self.mf.get_fock(dm=rdm1_ao)
            fock = np.linalg.multi_dot((self.mf.mo_coeff.T, fock_ao, self.mf.mo_coeff))
            return fock

        solver = SelfConsistentField(get_fock, self.se, self.mol.nelectron, log=NullLogger())
        solver.conv_tol = 1e-10
        solver.chempot_opt_options = dict(conv_tol=1e-8)
        solver.kernel()

        rdm1 = solver.gf_res.occupied().moment(0) * 2
        fock = get_fock(rdm1)

        self.assertTrue(solver.converged)
        # FIXME seems horribly flaky?
        self.assertAlmostEqual(lib.fp(rdm1), 3.6878183766, 4)
        self.assertAlmostEqual(lib.fp(fock), -2.0714652088, 4)


if __name__ == "__main__":
    unittest.main()
