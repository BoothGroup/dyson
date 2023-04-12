"""
Linear algebra utilities.
"""

import numpy as np


def matrix_power(m, power, hermitian=True, threshold=1e-10, return_error=False):
    """
    Compute the power of the matrix `m` via the eigenvalue
    decomposition.

    Parameters
    ----------
    m : numpy.ndarray (n, n)
        The matrix to be raised to a power.
    power : float
        The power to which the matrix is to be raised.
    hermitian : bool, optional
        Whether the matrix is hermitian.  Default value is `True`.
    threshold : float, optional
        Threshold for removing singularities.  Default value is
        `1e-10`.
    return_error : bool, optional
        Whether to return the error in the power.  Default value is
        `False`.

    Returns
    -------
    m_pow : numpy.ndarray (n, n)
        The matrix raised to the power.
    error : float, optional
        The error in the power.  Only returned if `return_error` is
        `True`.
    """

    if hermitian:
        # assert np.allclose(m, m.T.conj())
        eigvals, eigvecs = np.linalg.eigh(m)
    else:
        eigvals, eigvecs = np.linalg.eig(m)

    if power < 0:
        # Remove singularities
        mask = np.abs(eigvals) > threshold
    else:
        mask = np.ones_like(eigvals, dtype=bool)

    if hermitian and not np.iscomplexobj(m):
        if np.abs(power) < 1:
            mask = np.logical_and(mask, eigvals > 0)
        eigvecs_right = eigvecs.T.conj()
    elif hermitian and np.iscomplexobj(m):
        power = power + 0.0j
        eigvecs_right = eigvecs.T.conj()
    else:
        power = power + 0.0j
        eigvecs_right = np.linalg.inv(eigvecs)

    left = eigvecs[:, mask] * eigvals[mask][None] ** power
    right = eigvecs_right[mask]
    m_pow = np.dot(left, right)

    if return_error:
        left = eigvecs[:, ~mask] * eigvals[~mask][None]
        right = eigvecs_right[~mask]
        m_res = np.dot(left, right)
        error = np.linalg.norm(np.linalg.norm(m_res))
        return m_pow, error
    else:
        return m_pow


def hermi_sum(m):
    """
    Return m + m^†

    Parameters
    ----------
    m : numpy.ndarray (n, n)
        The matrix to be summed with its hermitian conjugate.

    Returns
    -------
    m_sum : numpy.ndarray (n, n)
        The sum of the matrix with its hermitian conjugate.
    """

    return m + m.T.conj()


def scaled_error(a, b):
    """
    Return the scaled error between two matrices.

    Parameters
    ----------
    a : numpy.ndarray (n, n)
        The first matrix.
    b : numpy.ndarray (n, n)
        The second matrix.

    Returns
    -------
    error : float
        The scaled error between the two matrices.
    """

    a = a / max(np.max(np.abs(a)), 1)
    b = b / max(np.max(np.abs(b)), 1)

    return np.linalg.norm(a - b)


def remove_unphysical(eigvecs, nphys, eigvals=None, tol=1e-8):
    """
    Remove eigenvectors with a small physical component.

    Parameters
    ----------
    eigvecs : numpy.ndarray or tuple of numpy.ndarray
        Eigenvectors.  If a tuple, the first element is the left
        eigenvectors and the second element is the right
        eigenvectors.
    nphys : int
        Number of physical orbitals.
    eigvals : numpy.ndarray, optional
        Eigenvalues.  Default value is `None`.
    tol : float, optional
        Threshold for removing eigenvectors.  Default value is
        `1e-8`.

    Returns
    -------
    eigvals : numpy.ndarray, optional
        Eigenvalues.  Only returned if `eigvals` is not `None`.
    eigvecs : numpy.ndarray or tuple of numpy.ndarray
        Eigenvectors.  If a tuple, the first element is the left
        eigenvectors and the second element is the right
        eigenvectors.
    """

    if isinstance(eigvecs, tuple):
        eigvecs_l, eigvecs_r = eigvecs
    else:
        eigvecs_l = eigvecs_r = eigvecs

    mask = np.abs(np.sum(eigvecs_l[:nphys] * eigvecs_r.conj()[:nphys], axis=0)) > tol

    if isinstance(eigvecs, tuple):
        eigvecs_out = (eigvecs_l[:, mask], eigvecs_r[:, mask])
    else:
        eigvecs_out = eigvecs[:, mask]

    if eigvals is not None:
        return eigvals[mask], eigvecs_out
    else:
        return eigvecs_out
