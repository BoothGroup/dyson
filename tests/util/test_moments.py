"""
Tests for moment utilities.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2, lib
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
        cls.f, cls.se, cls.gf, = f, se, gf

    @classmethod
    def tearDownClass(cls):
        del cls.f, cls.se, cls.gf

    def test_se_moments_to_gf_moments(self):
        t_se = self.se.moment(range(10))
        t_gf = self.gf.moment(range(12))
        t_gf_recov = util.se_moments_to_gf_moments(self.f, t_se)
        for i, (a, b) in enumerate(zip(t_gf, t_gf_recov)):
            self.assertAlmostEqual(util.scaled_error(a, b), 0, 10)

    def test_gf_moments_to_se_moments(self):
        t_gf = self.gf.moment(range(12))
        t_se = self.se.moment(range(10))
        static_recov, t_se_recov = util.gf_moments_to_se_moments(t_gf)
        self.assertAlmostEqual(util.scaled_error(static_recov, self.f), 0, 10)
        for i, (a, b) in enumerate(zip(t_se, t_se_recov)):
            self.assertAlmostEqual(util.scaled_error(a, b), 0, 10)

    def test_matvec_to_greens_function(self):
        h = np.block([[self.f, self.se.coupling], [self.se.coupling.T, np.diag(self.se.energy)]])
        matvec = lambda v: np.dot(h, v)
        bra = np.eye(self.se.nphys, self.se.nphys + self.se.naux)
        t_gf = self.gf.moment(range(10))
        t_gf_matvec = util.matvec_to_greens_function(matvec, 10, bra)
        for i, (a, b) in enumerate(zip(t_gf, t_gf_matvec)):
            self.assertAlmostEqual(util.scaled_error(a, b), 0, 10)

    def test_matvec_to_greens_function_chebyshev(self):
        emin = self.gf.energy.min()
        emax = self.gf.energy.max()
        a = (emax - emin) / (2.0 - 1e-2)
        b = (emax + emin) / 2.0
        energy_scaled = (self.gf.energy - b) / a
        c = np.zeros((100, self.gf.nphys, energy_scaled.size))
        c[0] = self.gf.coupling
        c[1] = self.gf.coupling * energy_scaled
        for i in range(2, 100):
            c[i] = 2.0 * c[i-1] * energy_scaled - c[i-2]
        t_gf = lib.einsum("qx,npx->npq", self.gf.coupling, c)
        h = np.block([[self.f, self.se.coupling], [self.se.coupling.T, np.diag(self.se.energy)]])
        matvec = lambda v: np.dot(h, v)
        bra = np.eye(self.se.nphys, self.se.nphys + self.se.naux)
        t_gf_matvec = util.matvec_to_greens_function_chebyshev(matvec, 100, (a, b), bra)
        for i, (a, b) in enumerate(zip(t_gf, t_gf_matvec)):
            self.assertAlmostEqual(util.scaled_error(a, b), 0, 10)


if __name__ == "__main__":
    unittest.main()
