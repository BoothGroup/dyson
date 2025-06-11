"""Moment utilities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson.util.linalg import einsum, matrix_power

if TYPE_CHECKING:
    from dyson.typing import Array


def se_moments_to_gf_moments(
    static: Array, se_moments: Array, overlap: Array | None = None, check_error: bool = True
) -> Array:
    """Convert moments of the self-energy to those of the Green's function.

    Args:
        static: Static part of the self-energy.
        se_moments: Moments of the self-energy.
        overlap: The overlap matrix (zeroth moment of the Green's function). If `None`, the zeroth
            moment of the Green's function is assumed to be the identity matrix.
        check_error: Whether to check the errors in the orthogonalisation of the moments.

    Returns:
        Moments of the Green's function.

    Notes:
        The first :math:`m` moments of the self-energy, along with the static part, are sufficient
        to define the first :math:`m+2` moments of the Green's function.
    """
    nmom, nphys, _ = se_moments.shape

    # Orthogonalise the moments
    if overlap is not None:
        hermitian = np.allclose(overlap, overlap.T.conj())
        orth, error_orth = matrix_power(
            overlap, -0.5, hermitian=hermitian, return_error=check_error
        )
        unorth, error_unorth = matrix_power(
            overlap, 0.5, hermitian=hermitian, return_error=check_error
        )
        if check_error:
            assert error_orth is not None and error_unorth is not None
            error = max(error_orth, error_unorth)
            if error > 1e-10:
                warnings.warn(
                    "Space contributing non-zero weight to the zeroth moments "
                    f"({max(error_orth, error_unorth)}) was removed during moment conversion.",
                    UserWarning,
                    2,
                )
        static = orth @ static @ orth
        se_moments = einsum("npq,ip,qj->nij", se_moments, orth, orth)

    # Get the powers of the static part
    powers = [np.eye(static.shape[-1], dtype=static.dtype), static]
    for i in range(2, nmom + 2):
        powers.append(powers[i - 1] @ static)
    gf_moments = np.zeros(
        (nmom + 2, nphys, nphys), dtype=np.result_type(se_moments.dtype, powers[0].dtype)
    )

    # Perform the recursion
    for i in range(nmom + 2):
        gf_moments[i] += powers[i]
        for n in range(i - 1):
            for m in range(i - n - 1):
                k = i - n - m - 2
                gf_moments[i] += powers[n] @ se_moments[m] @ gf_moments[k]

    # Unorthogonalise the moments
    if overlap is not None:
        gf_moments = einsum("npq,ip,qj->nij", gf_moments, unorth, unorth)

    return gf_moments


def gf_moments_to_se_moments(gf_moments: Array, check_error: bool = True) -> tuple[Array, Array]:
    """Convert moments of the Green's function to those of the self-energy.

    Args:
        gf_moments: Moments of the Green's function.
        check_error: Whether to check the errors in the orthogonalisation of the moments.

    Returns:
        static: Static part of the self-energy.
        moments: Moments of the self-energy.

    Notes:
        The first :math:`m+2` moments of the Green's function are sufficient to define the first
        :math:`m` moments of the self-energy, along with the static part.
    """
    nmom, nphys, _ = gf_moments.shape
    if nmom < 2:
        raise ValueError(
            "Need at least 2 moments of the Green's function to compute those of the self-energy."
        )

    # Orthogonalise the moments
    ident = np.allclose(gf_moments[0], np.eye(nphys))
    if not ident:
        hermitian = np.allclose(gf_moments[0], gf_moments[0].T.conj())
        orth, error_orth = matrix_power(
            gf_moments[0], -0.5, hermitian=hermitian, return_error=check_error
        )
        unorth, error_unorth = matrix_power(
            gf_moments[0], 0.5, hermitian=hermitian, return_error=check_error
        )
        if check_error:
            assert error_orth is not None and error_unorth is not None
            error = max(error_orth, error_unorth)
            if error > 1e-10:
                warnings.warn(
                    "Space contributing non-zero weight to the zeroth moments "
                    f"({max(error_orth, error_unorth)}) was removed during moment conversion.",
                    UserWarning,
                    2,
                )
        gf_moments = einsum("npq,ip,qj->nij", gf_moments, orth, orth)

    # Get the static part and the moments of the self-energy
    se_static = gf_moments[1]
    se_moments = np.zeros((nmom - 2, nphys, nphys), dtype=gf_moments.dtype)

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

    # Unorthogonalise the moments
    if not ident:
        se_static = unorth @ se_static @ unorth
        se_moments = einsum("npq,ip,qj->nij", se_moments, unorth, unorth)

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
    if len(on_diagonal) == 0:
        return np.zeros((0, 0))
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


def get_chebyshev_scaling_parameters(
    min_value: float, max_value: float, epsilon: float = 1e-3
) -> tuple[float, float]:
    """Get the Chebyshev scaling parameters.

    Args:
        min_value: Minimum value of the range.
        max_value: Maximum value of the range.
        epsilon: Small value to avoid division by zero.

    Returns:
        A tuple containing the scaling factor and the shift.
    """
    return (
        (max_value - min_value) / (2.0 - epsilon),
        (max_value + min_value) / 2.0,
    )
