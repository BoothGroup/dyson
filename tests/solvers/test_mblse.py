"""
Tests for MBLSE.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
import numpy as np
import scipy.linalg

from dyson import util, MBLSE, NullLogger


@pytest.mark.regression
class MBLSE_Tests(unittest.TestCase):
    """
    Test the `MBLSE` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy)
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        cls.f, cls.se = f, se

    @classmethod
    def tearDownClass(cls):
        del cls.f, cls.se

    def test_hermitian(self):
        f = self.f
        e = self.se.energy
        v = self.se.coupling
        h = np.block([[f, v], [v.T, np.diag(e)]])
        t = np.einsum("pk,qk,nk->npq", v, v, e[None]**np.arange(16)[:, None])
        w0, v0 = np.linalg.eigh(h)

        solver = MBLSE(f, t, max_cycle=0, log=NullLogger())
        w, v = solver.kernel()
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 3)

        solver = MBLSE(f, t, max_cycle=1, log=NullLogger())
        w, v = solver.kernel()
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 5)

        solver = MBLSE(f, t, max_cycle=3, log=NullLogger())
        w, v = solver.kernel()
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 7)

    def test_nonhermitian(self):
        f = self.f
        e = self.se.energy
        v = self.se.coupling
        pert = (np.random.random(v.shape) - 0.5) / 100
        h = np.block([[f, v+pert], [v.T, np.diag(e)]])
        t = np.einsum("pk,qk,nk->npq", v+pert, v, e[None]**np.arange(16)[:, None])
        w0, v0 = np.linalg.eigh(h)

        solver = MBLSE(f, t, max_cycle=0, log=NullLogger())
        w, v = solver.kernel()
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 3)

        solver = MBLSE(f, t, max_cycle=1, log=NullLogger())
        w, v = solver.kernel()
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 4)



if __name__ == "__main__":
    unittest.main()
