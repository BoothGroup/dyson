"""
Spectral function utilities.
"""

import numpy as np

from pyscf import lib


def build_spectral_function(energy, coupling, grid, eta=1e-1, trace=True, imag=True):
    """Build a spectral function."""

    if not trace:
        subscript = "pk,qk,wk->wpq"
    else:
        subscript = "pk,pk,wk->w"

    denom = 1.0 / (grid[:, None] - energy[None] + 1.0j * eta)
    sf = -lib.einsum(subscript, coupling, coupling, denom) / np.pi

    if imag:
        sf = sf.imag

    return sf
