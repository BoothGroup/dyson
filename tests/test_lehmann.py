"""
Tests for Lehmann representations.
"""

import unittest
import pytest

import numpy as np
from pyscf import lib

from dyson import Lehmann


@pytest.mark.regression
class Lehmann_Hermitian_Tests(unittest.TestCase):
    """
    Tests for the `Lehmann` class.
    """

    @classmethod
    def setUpClass(cls):
        e = np.cos(np.arange(100))
        c = np.sin(np.arange(1000)).reshape(10, 100)
        cls.aux = Lehmann(e, c)

    @classmethod
    def tearDownClass(cls):
        del cls.aux

    def test_moments(self):
        t = self.aux.moment([0, 1])
        self.assertAlmostEqual(lib.fp(t), -38.97393642159078, 10)
        t = self.aux.moment(1)
        self.assertAlmostEqual(lib.fp(t),   0.09373842776339, 10)

    def test_chebyshev_moments(self):
        t = self.aux.chebyshev_moment(range(10))
        self.assertAlmostEqual(lib.fp(t), -59.24704483050994, 10)
        t = self.aux.chebyshev_moment(5)
        self.assertAlmostEqual(lib.fp(t),  -0.89044258131632, 10)

    def test_matrix(self):
        phys = np.diag(np.cos(np.arange(self.aux.nphys)))
        mat = self.aux.matrix(phys)
        self.assertAlmostEqual(lib.fp(mat), -1.7176781717484837, 10)
        mat = self.aux.matrix(phys, chempot=0.1, out=mat)
        self.assertAlmostEqual(lib.fp(mat), -1.6486800825995238, 10)

    def test_diagonalise_matrix(self):
        phys = np.diag(np.cos(np.arange(self.aux.nphys)))
        e, c = self.aux.diagonalise_matrix(phys)
        self.assertAlmostEqual(lib.fp(e), -27.601125782799805, 10)
        self.assertAlmostEqual(lib.fp(c),   5.734873329655418, 10)

    def test_diagonalise_matrix_with_projection(self):
        phys = np.diag(np.cos(np.arange(self.aux.nphys)))
        e, c = self.aux.diagonalise_matrix_with_projection(phys)
        self.assertAlmostEqual(lib.fp(e), -27.601125782799805, 10)
        self.assertAlmostEqual(lib.fp(c),   0.893365900726610, 10)

    def test_weights(self):
        w = self.aux.weights()
        self.assertAlmostEqual(lib.fp(w), -3.958396736529412, 10)

    def test_as_orbitals(self):
        mo_energy, mo_coeff, mo_occ = self.aux.as_orbitals()
        self.assertAlmostEqual(lib.fp(mo_energy), -1.3761236354579696, 10)
        self.assertAlmostEqual(lib.fp(mo_coeff), 16.769969516330693, 10)
        self.assertAlmostEqual(lib.fp(mo_occ), -0.8204576360667741, 10)
        mo_energy, mo_coeff, mo_occ = self.aux.as_orbitals(mo_coeff=np.eye(self.aux.nphys))
        self.assertAlmostEqual(lib.fp(mo_energy), -1.3761236354579696, 10)
        self.assertAlmostEqual(lib.fp(mo_coeff), 16.769969516330693, 10)
        self.assertAlmostEqual(lib.fp(mo_occ), -0.8204576360667741, 10)

    def test_as_static_potential(self):
        mo_energy = np.cos(np.arange(self.aux.nphys))
        v = self.aux.as_static_potential(mo_energy)
        self.assertAlmostEqual(lib.fp(v), 92.45915312825181, 10)

    def test_as_perturbed_mo_energy(self):
        mo_energy = self.aux.as_perturbed_mo_energy()
        self.assertAlmostEqual(lib.fp(mo_energy), -1.3838965817318036, 10)


@pytest.mark.regression
class Lehmann_NonHermitian_Tests(unittest.TestCase):
    """
    Tests for the `Lehmann` class without hermiticity.
    """

    @classmethod
    def setUpClass(cls):
        e = np.cos(np.arange(100))
        c = (
            np.sin(np.arange(1000)).reshape(10, 100),
            np.cos(np.arange(1000, 2000)).reshape(10, 100),
        )
        cls.aux = Lehmann(e, c)

    @classmethod
    def tearDownClass(cls):
        del cls.aux

    def test_moments(self):
        t = self.aux.moment([0, 1])
        self.assertAlmostEqual(lib.fp(t), 5.500348836608749, 10)
        t = self.aux.moment(1)
        self.assertAlmostEqual(lib.fp(t), 0.106194238305493, 10)

    def test_chebyshev_moments(self):
        t = self.aux.chebyshev_moment(range(10))
        self.assertAlmostEqual(lib.fp(t), 22.449864768273073, 10)
        t = self.aux.chebyshev_moment(5)
        self.assertAlmostEqual(lib.fp(t),  0.481350207154633, 10)

    def test_matrix(self):
        phys = np.diag(np.cos(np.arange(self.aux.nphys)))
        mat = self.aux.matrix(phys)
        self.assertAlmostEqual(lib.fp(mat), 0.3486818284611074, 10)
        mat = self.aux.matrix(phys, chempot=0.1, out=mat)
        self.assertAlmostEqual(lib.fp(mat), 0.4176799176100642, 10)

    def test_weights(self):
        w = self.aux.weights()
        self.assertAlmostEqual(lib.fp(w), 3.360596091200564, 10)

    def test_as_static_potential(self):
        mo_energy = np.cos(np.arange(self.aux.nphys))
        v = self.aux.as_static_potential(mo_energy)
        self.assertAlmostEqual(lib.fp(v), 16.51344106239947, 10)

    def test_as_perturbed_mo_energy(self):
        mo_energy = self.aux.as_perturbed_mo_energy()
        self.assertAlmostEqual(lib.fp(mo_energy), -1.4195771834933937, 10)


if __name__ == "__main__":
    unittest.main()
