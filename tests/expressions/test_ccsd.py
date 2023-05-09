"""
Tests for CCSD.
"""

import unittest
import pytest

from pyscf import gto, scf, cc, lib
import numpy as np
import scipy.linalg

from dyson import util, Lehmann, NullLogger
from dyson import MBLGF, Davidson
from dyson.expressions import CCSD


@pytest.mark.regression
class CCSD_Tests(unittest.TestCase):
    """
    Test the `CCSD` expressions.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="cc-pvdz", verbose=0)
        mf = scf.RHF(mol).run()
        ccsd = cc.CCSD(mf).run()
        ccsd.solve_lambda()
        cls.mf, cls.ccsd = mf, ccsd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd

    def test_ip_ccsd(self):
        mf = self.mf

        ccsd = CCSD["1h"](mf)
        diag = ccsd.diagonal()
        matvec = ccsd.apply_hamiltonian

        solver = Davidson(matvec, diag, nroots=5, nphys=ccsd.nocc, log=NullLogger())
        solver.conv_tol = 1e-10
        solver.kernel()
        ip1 = -solver.get_greens_function().energies[-3:][::-1]

        ip2 = self.ccsd.ipccsd(nroots=5)[0]

        self.assertAlmostEqual(ip1[0], ip2[0], 7)
        self.assertAlmostEqual(ip1[1], ip2[1], 7)
        self.assertAlmostEqual(ip1[2], ip2[2], 7)

    def test_ea_ccsd(self):
        mf = self.mf

        ccsd = CCSD["1p"](mf)
        diag = ccsd.diagonal()
        matvec = ccsd.apply_hamiltonian

        solver = Davidson(matvec, diag, nroots=5, nphys=ccsd.nocc, log=NullLogger())
        solver.conv_tol = 1e-10
        solver.kernel()
        ea1 = solver.get_greens_function().energies[:3]

        ea2 = self.ccsd.eaccsd(nroots=5)[0]

        self.assertAlmostEqual(ea1[0], ea2[0], 7)
        self.assertAlmostEqual(ea1[1], ea2[1], 7)
        self.assertAlmostEqual(ea1[2], ea2[2], 7)

    def test_momgfccsd(self):
        mf = self.mf
        ccsd = self.ccsd
        nmom = 6

        expr = CCSD["1h"](mf, t1=ccsd.t1, t2=ccsd.t2, l1=ccsd.l1, l2=ccsd.l2)
        th = expr.build_gf_moments(nmom)
        expr = CCSD["1p"](mf, t1=ccsd.t1, t2=ccsd.t2, l1=ccsd.l1, l2=ccsd.l2)
        tp = expr.build_gf_moments(nmom)

        solverh = MBLGF(th, hermitian=False, log=NullLogger())
        solverh.kernel()
        gfh = solverh.get_greens_function()
        solverp = MBLGF(tp, hermitian=False, log=NullLogger())
        solverp.kernel()
        gfp = solverp.get_greens_function()
        gf = gfh + gfp

        grid = np.linspace(-5, 5, 1024)
        eta = 1e-1
        sf = util.build_spectral_function(gf.energies, gf.couplings, grid, eta=eta)

        momgfcc = cc.momgfccsd.MomGFCCSD(ccsd, ((nmom-2)//2, (nmom-2)//2))
        eh, vh, ep, vp = momgfcc.kernel()
        e = np.concatenate((eh, ep), axis=0)
        v = np.concatenate((vh[0], vp[0]), axis=1)
        u = np.concatenate((vh[1], vp[1]), axis=1)
        sf_ref = util.build_spectral_function(e, (v, u), grid, eta=eta)

        np.testing.assert_allclose(sf, sf_ref, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
