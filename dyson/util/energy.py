"""Energy functionals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util

if TYPE_CHECKING:
    from dyson.typing import Array


def gf_moments_galitskii_migdal(gf_moments_hole: Array, hcore: Array, factor: float = 1.0) -> float:
    """Compute the Galitskii--Migdal energy in terms of the moments of the hole Green's function.

    Args:
        gf_moments_hole: Moments of the hole Green's function. Only the first two (zeroth and first
            moments) are required.
        hcore: Core Hamiltonian.
        factor: Factor to scale energy. For UHF and GHF calculations, this should likely be 0.5,
            for RHF it is 1.0.

    Returns:
        Galitskii--Migdal energy.
    """
    e_gm = util.einsum("pq,qp->", gf_moments_hole[0], hcore)
    e_gm += np.trace(gf_moments_hole[1])
    return e_gm * factor
