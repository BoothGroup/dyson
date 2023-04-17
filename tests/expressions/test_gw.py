"""
Tests for GW.
"""

import unittest
import pytest

from pyscf import gto, dft, adc, agf2, lib
import numpy as np
import scipy.linalg

try:
    import momentGW
except ImportError:
    momentGW = None

from dyson import util, Lehmann, NullLogger
from dyson import MBLSE, MixedMBLSE, Davidson
from dyson.expressions import GW


@pytest.mark.regression
@pytest.mark.skipif(momentGW is None, reason="Moment GW tests require momentGW")
class GW_Tests(unittest.TestCase):
    """
    Test the `GW` expressions.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = dft.RKS(mol, xc="hf").density_fit().run()
        cls.mf = mf

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_moment_gw(self):
        gw = GW["Dyson"](self.mf)
        static = gw.get_static_part()
        th, tp = gw.build_se_moments(9)

        solverh = MBLSE(static, th, log=NullLogger())
        solverp = MBLSE(static, tp, log=NullLogger())
        solver = MixedMBLSE(solverh, solverp)
        solver.kernel()

        gf = solver.get_greens_function()
        gf = gf.physical()

        import momentGW
        gw_ref = momentGW.GW(self.mf)
        _, gf_ref, se_ref = gw_ref.kernel(9)
        gf_ref.remove_uncoupled(tol=0.1)

        np.testing.assert_allclose(gf_ref.moment(0), gf.moment(0), rtol=1e10, atol=1e-10)
        np.testing.assert_allclose(gf_ref.moment(1), gf.moment(1), rtol=1e10, atol=1e-10)

    def test_tda_gw(self):
        gw = GW["Dyson"](self.mf)
        gw.polarizability = "dtda"
        static = gw.get_static_part()
        matvec = lambda v: gw.apply_hamiltonian(v, static=static)
        diag = gw.diagonal(static=static)
        # TODO


if __name__ == "__main__":
    unittest.main()
