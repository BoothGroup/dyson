"""
Tests for the self-consistent solvers.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
import numpy as np
import scipy.linalg

from dyson import NullLogger, SelfConsistent


@pytest.mark.regression
class SelfConsistent_Tests(unittest.TestCase):
    """
    Test the `SelfConsistent` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy)
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        e = se.energy
        v = se.coupling
        h = np.block([[f, v], [v.T, np.diag(e)]])
        cls.e, cls.v, cls.f, cls.h = e, v, f, h
        cls.w0, cls.v0 = np.linalg.eigh(h)

    @classmethod
    def tearDownClass(cls):
        del cls.e, cls.v, cls.f, cls.h
        del cls.w0, cls.v0

    def test_orbital_target(self):
        m = lambda w: np.einsum("pk,qk,k->pq", self.v, self.v, 1/(w-self.e))
        solver = SelfConsistent(self.f, m, log=NullLogger())

        solver.target = 0
        w, v = solver.kernel()
        self.assertAlmostEqual(w[0], self.w0[0], 8)

        solver.target = 1
        w, v = solver.kernel()
        self.assertAlmostEqual(w[1], self.w0[1], 8)

        solver.target = 2
        w, v = solver.kernel()
        self.assertAlmostEqual(w[2], self.w0[2], 8)

        solver.target = 3
        w, v = solver.kernel()
        self.assertAlmostEqual(w[3], self.w0[3], 8)

    def test_min_target(self):
        m = lambda w: np.einsum("pk,qk,k->pq", self.v, self.v, 1/(w-self.e))
        solver = SelfConsistent(self.f, m, log=NullLogger())
        solver.target = "min"
        w, v = solver.kernel()
        self.assertAlmostEqual(w[0], self.w0[0], 8)

    def test_mindif_target(self):
        m = lambda w: np.einsum("pk,qk,k->pq", self.v, self.v, 1/(w-self.e))
        solver = SelfConsistent(self.f, m, log=NullLogger())
        solver.target = "mindif"
        solver.guess = self.w0[3] + 0.01
        w, v = solver.kernel()
        self.assertAlmostEqual(w[3], self.w0[3], 8)


if __name__ == "__main__":
    unittest.main()
