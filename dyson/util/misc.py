"""Miscellaneous utility functions."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

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
