"""Configuration for :mod:`pytest`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyscf import gto, scf

from dyson import numpy as np
from dyson.expressions import ADC2, CCSD, FCI, HF, TDAGW, ADC2x
from dyson.representations.lehmann import Lehmann
from dyson.solvers import Exact
from dyson.representations.spectral import Spectral

if TYPE_CHECKING:
    from typing import Callable, Hashable

    from dyson.expressions.expression import BaseExpression, ExpressionCollection
    from dyson.typing import Array

    ExactGetter = Callable[[scf.hf.RHF, type[BaseExpression]], Exact]


MOL_CACHE = {
    "h2-631g": gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="6-31g",
        verbose=0,
    ),
    "lih-631g": gto.M(
        atom="Li 0 0 0; H 0 0 1.64",
        basis="6-31g",
        verbose=0,
    ),
    "h2o-sto3g": gto.M(
        atom="O 0 0 0; H 0 0 1; H 0 1 0",
        basis="sto-3g",
        verbose=0,
    ),
    "he-ccpvdz": gto.M(
        atom="He 0 0 0",
        basis="cc-pvdz",
        verbose=0,
    ),
}

MF_CACHE = {
    "h2-631g": scf.RHF(MOL_CACHE["h2-631g"]).run(conv_tol=1e-12),
    "lih-631g": scf.RHF(MOL_CACHE["lih-631g"]).run(conv_tol=1e-12),
    "h2o-sto3g": scf.RHF(MOL_CACHE["h2o-sto3g"]).run(conv_tol=1e-12),
    "he-ccpvdz": scf.RHF(MOL_CACHE["he-ccpvdz"]).run(conv_tol=1e-12),
}

METHODS = [HF, CCSD, FCI, ADC2, ADC2x, TDAGW]
METHOD_NAMES = ["HF", "CCSD", "FCI", "ADC2", "ADC2x", "TDAGW"]


def pytest_generate_tests(metafunc):  # type: ignore
    if "mf" in metafunc.fixturenames:
        metafunc.parametrize("mf", MF_CACHE.values(), ids=MF_CACHE.keys())
    if "expression_cls" in metafunc.fixturenames:
        expressions = []
        ids = []
        for method, name in zip(METHODS, METHOD_NAMES):
            for sector, expression in method.items():
                expressions.append(expression)
                ids.append(f"{name}-{sector}")
        metafunc.parametrize("expression_cls", expressions, ids=ids)
    if "expression_method" in metafunc.fixturenames:
        expressions = []
        ids = []
        for method, name in zip(METHODS, METHOD_NAMES):
            expressions.append(method)
            ids.append(name)
        metafunc.parametrize("expression_method", expressions, ids=ids)


class Helper:
    """Helper class for tests."""

    @staticmethod
    def are_equal_arrays(moment1: Array, moment2: Array, tol: float = 1e-8) -> bool:
        """Check if two arrays are equal to within a threshold."""
        print(
            f"Error in {object.__repr__(moment1)} and {object.__repr__(moment2)}: "
            f"{np.max(np.abs(moment1 - moment2))}"
        )
        return np.allclose(moment1, moment2, atol=tol)

    @staticmethod
    def have_equal_moments(
        lehmann1: Lehmann | Array, lehmann2: Lehmann | Array, num: int, tol: float = 1e-8
    ) -> bool:
        """Check if two :class:`Lehmann` objects have equal moments to within a threshold."""
        moments1 = lehmann1.moments(range(num)) if isinstance(lehmann1, Lehmann) else lehmann1
        moments2 = lehmann2.moments(range(num)) if isinstance(lehmann2, Lehmann) else lehmann2
        checks: list[bool] = []
        for i, (m1, m2) in enumerate(zip(moments1, moments2)):
            errors = np.abs(m1 - m2)
            errors_scaled = errors / np.maximum(np.max(np.abs(m1)), 1.0)
            checks.append(bool(np.all(errors_scaled < tol)))
            print(
                f"Error in moment {i} of {object.__repr__(lehmann1)} and "
                f"{object.__repr__(lehmann2)}: {np.max(errors_scaled)} ({np.max(errors)})"
            )
        return all(checks)

    @staticmethod
    def recovers_greens_function(
        static: Array,
        self_energy: Lehmann,
        greens_function: Lehmann,
        num: int = 2,
        tol: float = 1e-8,
    ) -> bool:
        """Check if a self-energy recovers the Green's function to within a threshold."""
        overlap = greens_function.moment(0)
        greens_function_other = Lehmann(
            *self_energy.diagonalise_matrix_with_projection(static, overlap=overlap)
        )
        return Helper.have_equal_moments(greens_function, greens_function_other, num, tol=tol)

    @staticmethod
    def has_orthonormal_couplings(greens_function: Lehmann, tol: float = 1e-8) -> bool:
        """Check if the Green's function Dyson orbitals are orthonormal to within a threshold."""
        return Helper.are_equal_arrays(
            greens_function.moment(0), np.eye(greens_function.nphys), tol=tol
        )


@pytest.fixture(scope="session")
def helper() -> Helper:
    """Fixture for the :class:`Helper` class."""
    return Helper()


_EXACT_CACHE: dict[Hashable, Exact] = {}


def get_exact(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> Exact:
    """Get the exact solver for a given mean-field object and expression."""
    key = (mf.__class__, mf.mol.dumps(), expression_cls)
    if key not in _EXACT_CACHE:
        expression = expression_cls.from_mf(mf)
        exact = Exact.from_expression(expression)
        exact.kernel()
        _EXACT_CACHE[key] = exact

    exact = _EXACT_CACHE[key]
    assert exact.result is not None

    return exact


@pytest.fixture(scope="session")
def exact_cache() -> ExactGetter:
    """Fixture for a getter function for cached :class:`Exact` classes."""
    return get_exact


def _get_central_result(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    allow_hermitian: bool = True,
) -> Spectral:
    """Get the central result for the given mean-field method."""
    if "dyson" in expression_method:
        expression = expression_method.dyson.from_mf(mf)
        if expression.nconfig > 1024:
            pytest.skip("Skipping test for large Hamiltonian")
        if not expression.hermitian and not allow_hermitian:
            pytest.skip("Skipping test for non-Hermitian Hamiltonian with negative weights")
        exact = exact_cache(mf, expression_method.dyson)
        assert exact.result is not None
        return exact.result

    # Combine hole and particle results
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    if not expression_h.hermitian and not allow_hermitian:
        pytest.skip("Skipping test for non-Hermitian Hamiltonian with negative weights")
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None
    return Spectral.combine(exact_h.result, exact_p.result)
