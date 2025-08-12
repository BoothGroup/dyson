"""Backend management for :mod:`dyson`."""

from __future__ import annotations

import functools
import importlib
import os
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

_BACKEND = os.environ.get("DYSON_BACKEND", "numpy")
_BACKEND_WARNINGS = os.environ.get("DYSON_BACKEND_WARNINGS", "0") == "1"

_MODULE_CACHE: dict[tuple[str, str], ModuleType] = {}
_BACKENDS = {
    "numpy": {
        "numpy": "numpy",
        "scipy": "scipy",
    },
    "jax": {
        "numpy": "jax.numpy",
        "scipy": "jax.scipy",
    },
}


def set_backend(backend: str) -> None:
    """Set the backend for :mod:`dyson`."""
    global _BACKEND  # noqa: PLW0603
    if backend not in _BACKENDS:
        raise ValueError(
            f"Invalid backend: {backend}. Available backends are: {list(_BACKENDS.keys())}"
        )
    _BACKEND = backend


def cast_returned_array(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a function to coerce its returned array to the backend type."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(numpy.asarray(r) for r in result)
        return numpy.asarray(result)

    return wrapper


class ProxyModule(ModuleType):
    """Dynamic proxy module for backend-specific imports."""

    def __init__(self, key: str) -> None:
        """Initialise the object."""
        super().__init__(f"{__name__}.{key}")
        self._key = key

    def __getattr__(self, attr: str) -> ModuleType:
        """Get the attribute from the backend module."""
        mod = self._load()
        return getattr(mod, attr)

    def _load(self) -> ModuleType:
        """Load the backend module."""
        # Check the cache
        key = (self._key, _BACKEND)
        if key in _MODULE_CACHE:
            return _MODULE_CACHE[key]

        # Load the module
        keys = self._key.split(".")
        module = _BACKENDS[_BACKEND][keys[0]]
        if len(keys) > 1:
            module += "." + ".".join(keys[1:])
        _MODULE_CACHE[key] = importlib.import_module(module)

        return _MODULE_CACHE[key]


if TYPE_CHECKING:
    import numpy
    import scipy
else:
    numpy = ProxyModule("numpy")  # type: ignore[assignment]
    scipy = ProxyModule("scipy")  # type: ignore[assignment]
    scipy.optimize = ProxyModule("scipy.optimize")  # type: ignore[assignment]
