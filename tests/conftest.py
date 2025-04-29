"""Configuration for :mod:`pytest`."""

from __future__ import annotations

from pyscf import gto, scf

from dyson import numpy as np
from dyson.expressions import HF, CCSD, FCI


MOL_CACHE = {
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
    "lih-631g": scf.RHF(MOL_CACHE["lih-631g"]).run(),
    "h2o-sto3g": scf.RHF(MOL_CACHE["h2o-sto3g"]).run(),
    "he-ccpvdz": scf.RHF(MOL_CACHE["he-ccpvdz"]).run(),
}


def pytest_generate_tests(metafunc):  # type: ignore
    if "mf" in metafunc.fixturenames:
        metafunc.parametrize("mf", MF_CACHE.values(), ids=MF_CACHE.keys())
    if "expressions" in metafunc.fixturenames:
        metafunc.parametrize(
            "expressions",
            [HF, CCSD, FCI],
            ids=["HF", "CCSD", "FCI"],
        )
    if "sector" in metafunc.fixturenames:
        metafunc.parametrize("sector", ["1h", "1p"], ids=["1h", "1p"])
