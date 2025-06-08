"""Miscellaneous utility functions."""

from __future__ import annotations

import warnings
from contextlib import contextmanager


@contextmanager
def catch_warnings(warning_type: type[Warning] = Warning) -> list[Warning]:
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
    warnings.filters = user_filters
