"""
Miscellaneous utilities.
"""

import numpy as np


def cache(function):
    """
    Caches return values according to positional and keyword arguments
    in the `_cache` property of an object.
    """

    def wrapper(obj, *args, **kwargs):
        if (function.__name__, args, tuple(kwargs.items())) in obj._cache:
            return obj._cache[function.__name__, args, tuple(kwargs.items())]
        else:
            out = function(obj, *args, **kwargs)
            obj._cache[function.__name__, args, tuple(kwargs.items())] = out
            return out

    return wrapper
