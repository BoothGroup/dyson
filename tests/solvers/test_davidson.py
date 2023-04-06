"""
Tests for the Davidson solver.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
from pyscf.lib.linalg_helper import pick_real_eigs
import numpy as np
import scipy.linalg

from dyson import NullLogger, Davidson


@pytest.mark.regression
class Davidson_Tests(unittest.TestCase):
    """
    Test for the `Davidson` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy[mf.mo_occ > 0])
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        e = se.energy
        v = se.coupling[mf.mo_occ > 0]
        h = np.block([[f, v], [v.T, np.diag(e)]])
        cls.e, cls.v, cls.f, cls.h = e, v, f, h
        cls.w0, cls.v0 = np.linalg.eigh(h)

    @classmethod
    def tearDownClass(cls):
        del cls.e, cls.v, cls.f, cls.h
        del cls.w0, cls.v0

    def test_hermitian(self):
        m = lambda v: np.dot(self.h, v)
        d = np.diag(self.h)
        solver = Davidson(m, d, picker=pick_real_eigs, log=NullLogger())
        w, v = solver.kernel()
        self.assertAlmostEqual(w[0], self.w0[0], 8)
        self.assertAlmostEqual(w[1], self.w0[1], 8)

    def test_hermitian_guess(self):
        m = lambda v: np.dot(self.h, v)
        d = np.diag(self.h)
        guess = np.zeros((1, d.size))
        guess[0, np.argmin(d)] = 1
        solver = Davidson(m, d, picker=pick_real_eigs, guess=guess, nroots=1, log=NullLogger())
        w, v = solver.kernel()
        self.assertAlmostEqual(w[0], self.w0[0], 8)

    def test_nonhermitian(self):
        pert = (np.random.random(self.v.shape) - 0.5) / 100
        h = np.block([[self.f, self.v+pert], [self.v.T, np.diag(self.e)]])
        m = lambda v: np.dot(h, v)
        d = np.diag(h)
        solver = Davidson(m, d, picker=pick_real_eigs, hermitian=False, log=NullLogger())
        w, v = solver.kernel()
        w0, v0 = np.linalg.eig(h)
        w0 = w0[np.argsort(w0.real)]
        self.assertAlmostEqual(w[0], w0[0], 8)
        self.assertAlmostEqual(w[1], w0[1], 8)

    def test_nonhermitian_guess(self):
        pert = (np.random.random(self.v.shape) - 0.5) / 100
        h = np.block([[self.f, self.v+pert], [self.v.T, np.diag(self.e)]])
        m = lambda v: np.dot(h, v)
        d = np.diag(h)
        guess = np.zeros((1, d.size))
        guess[0, np.argmin(d)] = 1
        solver = Davidson(m, d, picker=pick_real_eigs, hermitian=False, guess=guess, log=NullLogger())
        w, v = solver.kernel()
        w0, v0 = np.linalg.eig(h)
        w0 = w0[np.argsort(w0.real)]
        self.assertAlmostEqual(w[0], w0[0], 8)
        self.assertAlmostEqual(w[1], w0[1], 8)



if __name__ == "__main__":
    unittest.main()
