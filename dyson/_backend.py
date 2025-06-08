"""Backend management for :mod:`dyson`."""

from __future__ import annotations

import importlib
import os

from types import ModuleType


try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

_BACKEND = os.environ.get("DYSON_BACKEND", "numpy")
_module_cache: dict[tuple[str, str], ModuleType] = {}

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
    global _BACKEND
    if backend not in _BACKENDS:
        raise ValueError(
            f"Invalid backend: {backend}. Available backends are: {list(_BACKENDS.keys())}"
        )
    _BACKEND = backend


class ProxyModule(ModuleType):
    """Dynamic proxy module for backend-specific imports."""

    def __init__(self, key: str):
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
        if key in _module_cache:
            return _module_cache[key]

        # Load the module
        module = _BACKENDS[_BACKEND][self._key]
        _module_cache[key] = importlib.import_module(module)

        return _module_cache[key]


numpy = ProxyModule("numpy")
scipy = ProxyModule("scipy")
