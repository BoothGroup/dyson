"""
Spectral function utilities.
"""

import numpy as np
from pyscf import lib
from scipy.sparse.linalg import LinearOperator, gcrotmk


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


def build_exact_spectral_function(expression, grid, eta=1e-1, trace=True, imag=True, conv_tol=1e-8):
    """
    Build a spectral function exactly for a given expression.

    Parameters
    ----------
    expression : BaseExpression
        Expression to build the spectral function for.
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
    conv_tol : float, optional
        Threshold for convergence. Default value is `1e-8`.

    Returns
    -------
    sf : numpy.ndarray
        Spectral function.

    Notes
    -----
    If convergence isn't met for elements, they are set to NaN.
    """

    if not trace:
        subscript = "pk,qk,wk->wpq"
    else:
        subscript = "pk,pk,wk->w"

    # FIXME: Consistent interface
    apply_kwargs = {}
    if hasattr(expression, "get_static_part"):
        apply_kwargs["static"] = expression.get_static_part()
    diag = expression.diagonal(**apply_kwargs)

    def matvec_dynamic(freq, vec):
        """Compute (freq - H - i\eta) * vec."""
        out = (freq - 1.0j * eta) * vec
        out -= expression.apply_hamiltonian(vec.real, **apply_kwargs)
        if np.any(np.abs(vec.imag) > 1e-14):
            out -= expression.apply_hamiltonian(vec.imag, **apply_kwargs) * 1.0j
        return out

    def matdiv_dynamic(freq, vec):
        """Approximate vec / (freq - H - i\eta)."""
        out = vec / (freq - diag - 1.0j * eta)
        out[np.isinf(out)] = np.nan
        return out

    shape = (grid.size,)
    if not trace:
        shape += (expression.nmo, expression.nmo)
    sf = np.zeros(shape, dtype=np.complex128)

    bras = []
    for p in range(expression.nmo):
        bras.append(expression.get_wavefunction_bra(p))

    for p in range(expression.nmo):
        ket = expression.get_wavefunction_ket(p)

        for w in range(grid.size):
            shape = (diag.size, diag.size)
            ax = LinearOperator(shape, lambda x: matvec_dynamic(grid[w], x), dtype=np.complex128)
            mx = LinearOperator(shape, lambda x: matdiv_dynamic(grid[w], x), dtype=np.complex128)
            x0 = matdiv_dynamic(grid[w], ket)
            x, info = gcrotmk(ax, ket, x0=x0, M=mx, atol=0.0, rtol=conv_tol, m=30)

            if info != 0:
                sf[w] = np.nan
            elif not trace:
                for q in range(expression.nmo):
                    sf[w, p, q] = np.dot(bras[q], x)
            else:
                sf[w] += np.dot(bras[p], x)

    sf = sf / np.pi
    if imag:
        sf = sf.imag

    return sf
