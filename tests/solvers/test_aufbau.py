"""
Tests for AufbauPrinciple.
"""

import unittest
import pytest

from pyscf import gto, scf, agf2
import numpy as np

from dyson import util, AufbauPrinciple, NullLogger
from dyson.lehmann import Lehmann


@pytest.mark.regression
class AufbauPrinciple_Tests(unittest.TestCase):
    """
    Test the `AufbauPrinciple` solver.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        f = np.diag(mf.mo_energy)
        se = agf2.AGF2(mf, nmom=(None, None)).build_se().get_virtual()
        cls.mf, cls.mol = mf, mol
        cls.f, cls.se = f, se

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.mol, cls.f, cls.se

    def test_hf(self):
        gf = Lehmann(self.mf.mo_energy, np.eye(self.mf.mo_energy.size))

        solver = AufbauPrinciple(gf, self.mol.nelectron, log=NullLogger())
        solver.kernel()

        self.assertTrue(solver.converged)
        self.assertAlmostEqual(solver.error, 0.0, 7)
        self.assertAlmostEqual(solver.homo, self.mf.mo_energy[self.mf.mo_occ > 0].max(), 7)
        self.assertAlmostEqual(solver.lumo, self.mf.mo_energy[self.mf.mo_occ == 0].min(), 7)

    def test_agf2(self):
        f = self.f
        e = self.se.energy
        v = self.se.coupling
        h = np.block([[f, v], [v.T, np.diag(e)]])
        w, v = np.linalg.eigh(h)
        v = v[:f.shape[0]]
        gf = Lehmann(w, v)

        solver = AufbauPrinciple(gf, self.mol.nelectron, log=NullLogger())
        solver.kernel()

        self.assertTrue(solver.converged)
        self.assertAlmostEqual(solver.error, 0.017171058925, 7)


if __name__ == "__main__":
    unittest.main()
