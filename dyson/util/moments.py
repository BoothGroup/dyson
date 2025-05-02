"""Moment utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np

if TYPE_CHECKING:
    from typing import Callable

    from dyson.typing import Array


def se_moments_to_gf_moments(static: Array, se_moments: Array) -> Array:
    """Convert moments of the self-energy to those of the Green's function.

    Args:
        static: Static part of the self-energy.
        moments: Moments of the self-energy.

    Returns:
        Moments of the Green's function.

    Notes:
        The first :math:`m` moments of the self-energy, along with the static part, are sufficient
        to define the first :math:`m+2` moments of the Green's function.
    """
    nmom, nphys, _ = se_moments.shape
    gf_moments = np.zeros((nmom + 2, nphys, nphys), dtype=se_moments.dtype)

    # Get the powers of the static part
    powers = [np.eye(nphys, dtype=se_moments.dtype)]
    for i in range(1, nmom + 2):
        powers.append(powers[i - 1] @ static)

    # Perform the recursion
    for i in range(nmom + 2):
        gf_moments[i] += powers[i]
        for n in range(i - 1):
            for m in range(i - n - 1):
                k = i - n - m - 2
                gf_moments[i] += powers[n] @ se_moments[m] @ gf_moments[k]

    return gf_moments


def gf_moments_to_se_moments(
    gf_moments: Array, allow_non_identity: bool = False
) -> tuple[Array, Array]:
    """Convert moments of the Green's function to those of the self-energy.

    Args:
        gf_moments: Moments of the Green's function.
        allow_non_identity: If `True`, allow the zeroth moment of the Green's function to be
            non-identity.

    Returns:
        static: Static part of the self-energy.
        moments: Moments of the self-energy.

    Notes:
        The first :math:`m+2` moments of the Green's function are sufficient to define the first
        :math:`m` moments of the self-energy, along with the static part.

    Raises:
        ValueError: If the zeroth moment of the Green's function is not the identity matrix.
    """
    nmom, nphys, _ = gf_moments.shape
    if nmom < 2:
        raise ValueError(
            "Need at least 2 moments of the Green's function to compute those of the self-energy."
        )
    if not allow_non_identity and not np.allclose(gf_moments[0], np.eye(nphys)):
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

    # Get the powers of the static part
    powers = [np.eye(nphys, dtype=gf_moments.dtype), se_static]
    for i in range(2, nmom):
        powers.append(powers[i - 1] @ se_static)

    # Perform the recursion
    for i in range(nmom - 2):
        se_moments[i] = gf_moments[i + 2] - powers[i + 2]
        for l in range(i + 1):
            for m in range(i + 1 - l):
                k = i - l - m
                if m != i:
                    se_moments[i] -= powers[l] @ se_moments[m] @ gf_moments[k]

    return se_static, se_moments


def build_block_tridiagonal(
    on_diagonal: list[Array],
    off_diagonal_upper: list[Array],
    off_diagonal_lower: list[Array] | None = None,
) -> Array:
    """Build a block tridiagonal matrix.

    Args:
        on_diagonal: On-diagonal blocks.
        off_diagonal_upper: Off-diagonal blocks for the upper half of the matrix.
        off_diagonal_lower: Off-diagonal blocks for the lower half of the matrix. If
            `None`, use the transpose of `off_diagonal_upper`.

    Returns:
        A block tridiagonal matrix with the given blocks.

    Notes:
        The number of on-diagonal blocks should be one greater than the number of off-diagonal
        blocks.
    """
    zero = np.zeros_like(on_diagonal[0])
    if off_diagonal_lower is None:
        off_diagonal_lower = [matrix.T.conj() for matrix in off_diagonal_upper]

    def _block(i: int, j: int) -> Array:
        """Return the block at position (i, j)."""
        if i == j:
            return on_diagonal[i]
        elif j == i - 1:
            return off_diagonal_upper[j]
        elif i == j - 1:
            return off_diagonal_lower[i]
        return zero

    # Construct the block tridiagonal matrix
    matrix = np.block(
        [[_block(i, j) for j in range(len(on_diagonal))] for i in range(len(on_diagonal))]
    )

    return matrix


def matvec_to_gf_moments(
    matvec: Callable[[Array], Array], nmom: int, bra: Array, ket: Array | None = None
) -> Array:
    """Build moments of a Green's function using the matrix-vector operation.

    Args:
        matvec: Matrix-vector product function.
        nmom: Number of moments to compute.
        bra: Bra vectors.
        ket: Ket vectors, if `None` then use `bra`.

    Returns:
        Moments of the Green's function.

    Notes:
        This function is functionally identical to :method:`Expression.build_gf_moments`, but the
        latter is optimised for :class:`Expression` objects.
    """
    nphys, nconf = bra.shape
    moments = np.zeros((nmom, nphys, nphys), dtype=bra.dtype)
    if ket is None:
        ket = bra
    ket = ket.copy()

    # Build the moments
    for n in range(nmom):
        part = bra.conj() @ ket.T
        if np.iscomplexobj(part) and not np.iscomplexobj(moments):
            moments = moments.astype(np.complex128)
        moments[n] = part
        if n != (nmom - 1):
            ket = np.array([matvec(vector) for vector in ket])

    return moments


def matvec_to_gf_moments_chebyshev(
    matvec: Callable[[Array], Array],
    nmom: int,
    scaling: tuple[float, float],
    bra: Array,
    ket: Array | None = None,
) -> Array:
    """Build Chebyshev moments of a Green's function using the matrix-vector operation.

    Args:
        matvec: Matrix-vector product function.
        nmom: Number of moments to compute.
        scaling: Scaling factors to ensure the energy scale of the Lehmann representation is in
            `[-1, 1]`. The scaling is applied as `(energies - scaling[1]) / scaling[0]`.
        bra: Bra vectors.
        ket: Ket vectors, if `None` then use `bra`.

    Returns:
        Moments of the Green's function.

    Notes:
        This function is functionally identical to :method:`Expression.build_gf_chebyshev_moments`,
        but the latter is optimised for :class:`Expression` objects.
    """
    nphys, nconf = bra.shape
    moments = np.zeros((nmom, nphys, nphys), dtype=bra.dtype)
    a, b = scaling
    ket0 = ket.copy() if ket is not None else bra.copy()
    ket1 = np.array([matvec(vector) - scaling[1] * vector for vector in ket0]) / scaling[0]

    # Build the moments
    moments[0] = bra @ ket0.T.conj()
    for n in range(1, nmom):
        moments[n] = bra @ ket1.T.conj()
        if n != (nmom - 1):
            ket2 = np.array([matvec(vector) - scaling[1] * vector for vector in ket1]) / scaling[0]
            ket0, ket1 = ket1, ket2

    return moments
