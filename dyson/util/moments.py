"""
Moment utilities.
"""

import numpy as np


def se_moments_to_gf_moments(se_static, se_moments):
    """
    Convert moments of the self-energy to those of the Green's
    function. The first m moments of the self-energy, along with
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
    gf_moments = np.zeros((nmom + 2, nphys, nphys), dtype=se_moments.dtype)

    for i in range(nmom + 2):
        gf_moments[i] += np.linalg.matrix_power(se_static, i)
        for n in range(i - 1):
            for m in range(i - n - 1):
                k = i - n - m - 2
                gf_moments[i] += np.linalg.multi_dot(
                    (
                        np.linalg.matrix_power(se_static, n),
                        se_moments[m],
                        gf_moments[k],
                    )
                )

    return gf_moments


def gf_moments_to_se_moments(gf_moments):
    """
    Convert moments of the Green's function to those of the
    self-energy. The first m+2 moments of the Green's function
    are sufficient to define the first m moments of the self-energy,
    along with the static part. See Eqns 2.103-105 of Backhouse's
    thesis.

    Parameters
    ----------
    gf_moments : numpy.ndarray (m+2, n, n)
        Moments of the Green's function.

    Returns
    -------
    se_static : numpy.ndarray (n, n)
        Static part of the self-energy.
    se_moments : numpy.ndarray (m, n, n)
        Moments of the self-energy.
    """

    nmom, nphys, _ = gf_moments.shape

    if nmom < 2:
        raise ValueError(
            "At least 2 moments of the Green's function are required to "
            "find those of the self-energy."
        )

    if not np.allclose(gf_moments[0], np.eye(nphys)):
        raise ValueError("The first moment of the Green's function must be the identity.")

    se_moments = np.zeros((nmom - 2, nphys, nphys), dtype=gf_moments.dtype)
    se_static = gf_moments[1]

    # Invert the recurrence relations:
    #
    #   G_{n} = F^{n} + \sum_{l+m+k}^{n-2} F^{l} \Sigma_{m} G_{k}
    #   \Sigma_{n} = G_{n} - F^{n} - \sum_{l+m+k}^{n-2} F^{l} \Sigma_{m} G_{k}
    #
    # where the sum is over all possible combinations of l, m, and k but
    # with the constraint that m != n. This case is F^{0} \Sigma_{n} G_{0}
    # which is equal to the desired LHS.

    for i in range(nmom - 2):
        se_moments[i] = gf_moments[i + 2].copy()
        se_moments[i] -= np.linalg.matrix_power(se_static, i + 2)
        for l in range(i + 1):
            for m in range(i + 1 - l):
                k = i - l - m
                if m != i:
                    se_moments[i] -= np.linalg.multi_dot(
                        (
                            np.linalg.matrix_power(se_static, l),
                            se_moments[m],
                            gf_moments[k],
                        )
                    )

    return se_static, se_moments


def build_block_tridiagonal(on_diagonal, off_diagonal_upper, off_diagonal_lower=None):
    """
    Build a block tridiagonal matrix.

    Parameters
    ----------
    on_diagonal : numpy.ndarray (m+1, n, n)
        On-diagonal blocks.
    off_diagonal_upper : numpy.ndarray (m, n, n)
        Off-diagonal blocks for the upper half of the matrix.
    off_diagonal_lower : numpy.ndarray (m, n, n), optional
        Off-diagonal blocks for the lower half of the matrix. If
        `None`, use the transpose of `off_diagonal_upper`.
    """

    zero = np.zeros_like(on_diagonal[0])

    if off_diagonal_lower is None:
        off_diagonal_lower = [m.T.conj() for m in off_diagonal_upper]

    m = np.block(
        [
            [
                on_diagonal[i]
                if i == j
                else off_diagonal_upper[j]
                if j == i - 1
                else off_diagonal_lower[i]
                if i == j - 1
                else zero
                for j in range(len(on_diagonal))
            ]
            for i in range(len(on_diagonal))
        ]
    )

    return m


def matvec_to_greens_function(matvec, nmom, bra, ket=None):
    """
    Build a set of moments using the matrix-vector product for a
    given Hamiltonian and a bra and ket vector.

    Parameters
    ----------
    matvec : callable
        Matrix-vector product function, takes a vector as input.
    nmom : int
        Number of moments to compute.
    bra : numpy.ndarray (n, m)
        Bra vector.
    ket : numpy.ndarray (n, m), optional
        Ket vector, if `None` then use the bra.
    """

    nphys, nconf = bra.shape
    moments = np.zeros((nmom, nphys, nphys))

    if ket is None:
        ket = bra
    ket = ket.copy()

    for n in range(nmom):
        moments[n] = np.dot(bra, ket.T.conj())
        if n != (nmom - 1):
            for i in range(nphys):
                ket[i] = matvec(ket[i])

    return moments


matvec_to_greens_function_monomial = matvec_to_greens_function


def matvec_to_greens_function_chebyshev(matvec, nmom, scale_factors, bra, ket=None):
    """
    Build a set of Chebyshev moments using the matrix-vector product
    for a given Hamiltonian and a bra and ket vector.

    Parameters
    ----------
    matvec : callable
        Matrix-vector product function, takes a vector as input.
    nmom : int
        Number of moments to compute.
    scale_factors : tuple of int
        Factors to scale the Hamiltonian as `(H - b) / a`, in order
        to keep the spectrum within [-1, 1]. These are typically
        defined as
            `a = (emax - emin) / (2 - eps)`
            `b = (emax + emin) / 2`
        where `emin` and `emax` are the minimum and maximum eigenvalues
        of H, and `eps` is a small number.
    bra : numpy.ndarray (n, m)
        Bra vector.
    ket : numpy.ndarray (n, m), optional
        Ket vector, if `None` then use the bra.
    """

    nphys, nconf = bra.shape
    moments = np.zeros((nmom, nphys, nphys))
    a, b = scale_factors

    if ket is None:
        ket = bra

    ket0 = ket.copy()
    ket1 = np.zeros_like(ket0)
    for i in range(nphys):
        ket1[i] = (matvec(ket0[i]) - b * ket0[i]) / a

    moments[0] = np.dot(bra, ket0.T.conj())

    for n in range(1, nmom):
        moments[n] = np.dot(bra, ket1.T.conj())
        if n != (nmom - 1):
            for i in range(nphys):
                ket2i = 2.0 * (matvec(ket1[i]) - b * ket1[i]) / a - ket0[i]
                ket0[i], ket1[i] = ket1[i], ket2i

    return moments
