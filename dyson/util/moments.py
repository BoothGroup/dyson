"""
Moment utilities.
"""

import numpy as np


def self_energy_to_greens_function(se_static, se_moments):
    """
    Convert moments of the Green's function to those of the
    self-energy. The first m moments of the self-energy, along with
    the static part, are sufficient to define the first m+2 moments
    of the Green's function. See Eqns 2.103-105 of Backhouse's thesis.

    Parameters
    ----------
    se_static : numpy.ndarray (n, n)
        Static part of the self-energy.
    se_moments : numpy.ndarray (m, n, n)
        Moments of the self-energy.

    Returns
    -------
    gf_moments : numpy.ndarray (m+2, n, n)
        Moments of the Green's function.
    """

    nmom, nphys, _ = se_moments.shape
    gf_moments = np.zeros((nmom+2, nphys, nphys), dtype=se_moments.dtype)

    for i in range(nmom+2):
        gf_moments[i] += np.linalg.matrix_power(se_static, i)
        for n in range(i-1):
            for m in range(i-n-1):
                k = i - n - m - 2
                gf_moments[i] += np.linalg.multi_dot((
                    np.linalg.matrix_power(se_static, n),
                    se_moments[m],
                    gf_moments[k],
                ))

    return gf_moments
