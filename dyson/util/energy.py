"""
Energy functionals.
"""

import numpy as np


def greens_function_galitskii_migdal(gf_moments_hole, hcore, factor=1.0):
    """
    Compute the energy using the Galitskii-Migdal formula in terms of
    the hole Green's function moments and the core Hamiltonian.

    Parameters
    ----------
    gf_moments : numpy.ndarray (m, n, n)
        Moments of the hole Green's function. Only the first two
        (n=0 and n=1) are required.
    hcore : numpy.ndarray (n, n)
        Core Hamiltonian.
    factor : float, optional
        Factor to scale energy. For UHF and GHF calculations, this
        should likely be 0.5, for RHF it is 1.0.  Default value is
        `1.0`.

    Returns
    -------
    e_gm : float
        Galitskii-Migdal energy.
    """

    e_gm = np.einsum("pq,qp->", gf_moments_hole[0], hcore)
    e_gm += np.trace(gf_moments_hole[1])

    return e_gm
