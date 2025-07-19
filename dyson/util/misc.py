"""Miscellaneous utility functions."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pyscf import gto, scf

if TYPE_CHECKING:
    from typing import Iterator
    from warnings import WarningMessage


@contextmanager
def catch_warnings(warning_type: type[Warning] = Warning) -> Iterator[list[WarningMessage]]:
    """Context manager to catch warnings.

    Returns:
        A list of caught warnings.
    """
    # Remove any user filters
    user_filters = warnings.filters[:]
    warnings.simplefilter("always", warning_type)

    # Catch warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        yield caught_warnings

    # Restore user filters
    warnings.filters[:] = user_filters  # type: ignore[index]


def get_mean_field(atom: str, basis: str, charge: int = 0, spin: int = 0) -> scf.RHF:
    """Get a mean-field object for a given system.

    Intended as a convenience function for examples.

    Args:
        atom: The atomic symbol of the system.
        basis: The basis set to use.
        charge: The total charge of the system.
        spin: The total spin of the system.

    Returns:
        A mean-field object for the system.
    """
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    return mf
