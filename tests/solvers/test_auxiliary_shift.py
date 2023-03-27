"""
Tests for AuxiliaryShift.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
import numpy as np

from dyson import util, AuxiliaryShift, NullLogger
from dyson.lehmann import Lehmann


@pytest.mark.regression
class AuxiliaryShift_Tests(unittest.TestCase):
    """
    Test the `AuxiliaryShift` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="cc-pvdz", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy)
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        cls.mf, cls.mol = mf, mol
        cls.f, cls.se = f, se

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.mol, cls.f, cls.se

    def test_agf2(self):
        solver = AuxiliaryShift(self.f, self.se, self.mol.nelectron, log=NullLogger())
        solver.conv_tol = 1e-6
        solver.kernel()

        self.assertTrue(solver.converged)
        self.assertAlmostEqual(solver.error, 0.0, 5)


if __name__ == "__main__":
    unittest.main()
