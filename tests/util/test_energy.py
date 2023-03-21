"""
Tests for energy functionals.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2, lib
import numpy as np

from dyson import util


@pytest.mark.regression
class Energy_Tests(unittest.TestCase):
    """
    Tests for the `util.energy` module.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        h = np.linalg.multi_dot((mf.mo_coeff.T, mol.get_hcore(), mf.mo_coeff))
        f = np.diag(mf.mo_energy)
        gf2 = agf2.AGF2(mf, nmom=(None, None))
        se = gf2.build_se()
        se.coupling[:gf2.nocc, se.energy > se.chempot] = 0
        se.coupling[gf2.nocc:, se.energy < se.chempot] = 0
        gf = se.get_greens_function(f)
        cls.gf2, cls.mf, cls.h, cls.f, cls.se, cls.gf = gf2, mf, h, f, se, gf

    @classmethod
    def tearDownClass(cls):
        del cls.gf2, cls.mf, cls.h, cls.f, cls.se, cls.gf

    def test_greens_function_galitskii_migdal(self):
        moments = self.gf.get_occupied().moment(range(2))
        e_gm = util.greens_function_galitskii_migdal(moments, self.h)
        e_gm += self.mf.mol.energy_nuc()

        e_ref = self.gf2.energy_1body(self.gf2.ao2mo(), self.gf)
        e_ref += self.gf2.energy_2body(self.gf, self.se)

        self.assertAlmostEqual(e_gm, e_ref, 8)


if __name__ == "__main__":
    unittest.main()
