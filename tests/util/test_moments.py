"""
Tests for moment utilities.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
import numpy as np

from dyson import util


@pytest.mark.regression
class Moments_Tests(unittest.TestCase):
    """
    Test for the `util.moments` module.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy[mf.mo_occ > 0])
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        se.coupling = se.coupling[mf.mo_occ > 0]
        gf = se.get_greens_function(f)
        cls.f, cls.se, cls.gf = f, se, gf

    @classmethod
    def tearDownClass(cls):
        del cls.f, cls.se, cls.gf

    def test_self_energy_to_greens_function(self):
        t_se = self.se.moment(range(10))
        t_gf = self.gf.moment(range(12))
        t_gf_recov = util.self_energy_to_greens_function(self.f, t_se)
        for i, (a, b) in enumerate(zip(t_gf, t_gf_recov)):
            self.assertAlmostEqual(util.scaled_error(a, b), 0, 10)


if __name__ == "__main__":
    unittest.main()
