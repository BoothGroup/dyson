"""
Tests for the exact solver.
"""

import unittest
import pytest

import numpy as np
import scipy.linalg

from dyson import NullLogger, Exact


@pytest.mark.regression
class Exact_Tests(unittest.TestCase):
    """
    Test the `Exact` solver.
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_real_hermitian(self):
        m = np.random.random((100, 100))
        m = 0.5 * (m + m.T.conj())
        solver = Exact(m, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], v.T.conj())
        np.testing.assert_almost_equal(m, m0)

    def test_complex_hermitian(self):
        m = np.random.random((100, 100)) + np.random.random((100, 100)) + 1.0j
        m = 0.5 * (m + m.T.conj())
        solver = Exact(m, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], v.T.conj())
        np.testing.assert_almost_equal(m, m0)

    def test_real_nonhermitian(self):
        m = np.random.random((100, 100))
        solver = Exact(m, hermitian=False, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], np.linalg.inv(v))
        np.testing.assert_almost_equal(m, m0)

    def test_complex_nonhermitian(self):
        m = np.random.random((100, 100)) + np.random.random((100, 100)) + 1.0j
        solver = Exact(m, hermitian=False, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], np.linalg.inv(v))
        np.testing.assert_almost_equal(m, m0)

    def test_real_hermitian_generalised(self):
        m = np.random.random((100, 100))
        m = 0.5 * (m + m.T.conj())
        s = np.random.random((100, 100))
        s = 0.5 * (s + s.T.conj())
        solver = Exact(m, overlap=s, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], v.T.conj())
        np.testing.assert_almost_equal(m, m0)

    def test_complex_hermitian_generalised(self):
        m = np.random.random((100, 100)) + np.random.random((100, 100)) + 1.0j
        m = 0.5 * (m + m.T.conj())
        s = np.random.random((100, 100))
        s = 0.5 * (s + s.T.conj())
        solver = Exact(m, overlap=s, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], v.T.conj())
        np.testing.assert_almost_equal(m, m0)

    def test_real_nonhermitian_generalised(self):
        m = np.random.random((100, 100))
        s = np.random.random((100, 100))
        solver = Exact(m, overlap=s, hermitian=False, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], np.linalg.inv(v))
        np.testing.assert_almost_equal(m, m0)

    def test_complex_nonhermitian_generalised(self):
        m = np.random.random((100, 100)) + np.random.random((100, 100)) + 1.0j
        s = np.random.random((100, 100))
        solver = Exact(m, overlap=s, hermitian=False, log=NullLogger())
        w, v = solver.kernel()
        m0 = np.dot(v * w[None], np.linalg.inv(v))
        np.testing.assert_almost_equal(m, m0)


if __name__ == "__main__":
    unittest.main()
