"""
Tests for MBLGF.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
import numpy as np
import scipy.linalg

from dyson import util, MBLGF, NullLogger


@pytest.mark.regression
class MBLGF_Tests(unittest.TestCase):
    """
    Test the `MBLGF` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy[mf.mo_occ == 0])
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        se.coupling = se.coupling[mf.mo_occ == 0]
        cls.f, cls.se = f, se

    @classmethod
    def tearDownClass(cls):
        del cls.f, cls.se

    def test_hermitian(self):
        f = self.f
        e = self.se.energy
        v = self.se.coupling
        h = np.block([[f, v], [v.T, np.diag(e)]])
        w0, v0 = np.linalg.eigh(h)
        nmo = self.se.nphys
        t = np.einsum("pk,qk,nk->npq", v0[:nmo], v0[:nmo], w0[None]**np.arange(16)[:, None])
        w0, v0 = util.remove_unphysical(v0, nmo, eigvals=w0, tol=1e-2)

        solver = MBLGF(t, max_cycle=2, log=NullLogger())
        w, v = solver.kernel()
        w, v = util.remove_unphysical(v, nmo, eigvals=w, tol=1e-2)
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 1)

        solver = MBLGF(t, max_cycle=3, log=NullLogger())
        w, v = solver.kernel()
        w, v = util.remove_unphysical(v, nmo, eigvals=w, tol=1e-2)
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 2)

        solver = MBLGF(t, max_cycle=4, log=NullLogger())
        w, v = solver.kernel()
        w, v = util.remove_unphysical(v, nmo, eigvals=w, tol=1e-2)
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)
        self.assertAlmostEqual(w[0], w0[0], 3)

    def test_nonhermitian(self):
        f = self.f
        e = self.se.energy
        v = self.se.coupling
        pert = (np.ones(v.shape) - 0.5) / 200
        h = np.block([[f, v+pert], [v.T, np.diag(e)]])
        w0, v0 = np.linalg.eig(h)
        mask = np.argsort(w0.real)
        w0, v0 = w0[mask], v0[:, mask]
        v0i = np.linalg.inv(v0).T
        nmo = self.se.nphys
        t = np.einsum("pk,qk,nk->npq", v0[:nmo], v0i[:nmo], w0[None]**np.arange(16)[:, None])
        w0, v0 = util.remove_unphysical(v0, nmo, eigvals=w0, tol=1e-2)

        solver = MBLGF(t, max_cycle=2, log=NullLogger())
        w, v = solver.kernel()
        w, v = util.remove_unphysical(v, nmo, eigvals=w, tol=1e-2)
        error = solver._check_moment_error()
        self.assertAlmostEqual(error, 0.0, 10)

        # FIXME: these tests are incredibly flaky

        #solver = MBLGF(t, max_cycle=3, log=NullLogger())
        #w, v = solver.kernel()
        #w, v = util.remove_unphysical(v, nmo, eigvals=w, tol=1e-2)
        #error = solver._check_moment_error()
        #self.assertAlmostEqual(error, 0.0, 10)

        #solver = MBLGF(t, max_cycle=4, log=NullLogger())
        #w, v = solver.kernel()
        #w, v = util.remove_unphysical(v, nmo, eigvals=w, tol=1e-2)
        #error = solver._check_moment_error()
        #self.assertAlmostEqual(error, 0.0, 10)


if __name__ == "__main__":
    unittest.main()
