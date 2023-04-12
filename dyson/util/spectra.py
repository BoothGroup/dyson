"""
Spectral function utilities.
"""

import numpy as np
from pyscf import lib


def build_spectral_function(energy, coupling, grid, eta=1e-1, trace=True, imag=True):
    """
    Build a spectral function.

    Parameters
    ----------
    energy : numpy.ndarray
        Energies of the states.
    coupling : numpy.ndarray or tuple of numpy.ndarray
        Coupling of the states to the spectral function. If a tuple
        is given, the first element is the left coupling and the
        second element is the right coupling.
    grid : numpy.ndarray
        Grid on which to evaluate the spectral function.
    eta : float, optional
        Broadening parameter. Default value is `1e-1`.
    trace : bool, optional
        Whether to trace over the spectral function before returning.
        Default value is `True`.
    imag : bool, optional
        Whether to return only the imaginary part of the spectral
        function.  Default value is `True`.

    Returns
    -------
    sf : numpy.ndarray
        Spectral function.
    """

    if isinstance(coupling, tuple):
        coupling_l, coupling_r = coupling
    else:
        coupling_l = coupling_r = coupling

    if not trace:
        subscript = "pk,qk,wk->wpq"
    else:
        subscript = "pk,pk,wk->w"

    denom = 1.0 / (grid[:, None] - energy[None] + 1.0j * eta)
    sf = -lib.einsum(subscript, coupling_l, coupling_r.conj(), denom) / np.pi

    if imag:
        sf = sf.imag

    return sf
