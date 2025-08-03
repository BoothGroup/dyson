"""Linear algebra."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, cast

import scipy.linalg

from dyson import numpy as np
from dyson.util import cache_by_id

if TYPE_CHECKING:
    from typing import Literal

    from dyson.typing import Array

einsum = functools.partial(np.einsum, optimize=True)

"""Flag to avoid using :func:`scipy.linalg.eig` and :func:`scipy.linalg.eigh`.

On some platforms, mixing :mod:`numpy` and :mod:`scipy` eigenvalue solvers can lead to performance
issues, likely from repeating warm-up overhead from conflicting BLAS and/or LAPACK libraries.
"""
AVOID_SCIPY_EIG: bool = True

"""Default biorthonormalisation method."""
BIORTH_METHOD: Literal["lu", "eig", "eig-balanced"] = "lu"


def is_orthonormal(vectors_left: Array, vectors_right: Array | None = None) -> bool:
    """Check if a set of vectors is orthonormal.

    Args:
        vectors_left: The left set of vectors to be checked.
        vectors_right: The right set of vectors to be checked. If `None`, use the left vectors.

    Returns:
        A boolean array indicating whether each vector is orthonormal.
    """
    if vectors_right is None:
        vectors_right = vectors_left
    if vectors_left.ndim == 1:
        vectors_left = vectors_left[:, None]
    if vectors_right.ndim == 1:
        vectors_right = vectors_right[:, None]
    overlap = einsum("ij,ik->jk", vectors_left.conj(), vectors_right)
    return np.allclose(overlap, np.eye(overlap.shape[0]), atol=1e-10, rtol=0.0)


@cache_by_id
def orthonormalise(vectors: Array, transpose: bool = False) -> Array:
    """Orthonormalise a set of vectors.

    Args:
        vectors: The set of vectors to be orthonormalised.
        transpose: Whether to transpose the vectors before and after orthonormalisation.

    Returns:
        The orthonormalised set of vectors.
    """
    if transpose:
        vectors = vectors.T.conj()

    overlap = vectors.T.conj() @ vectors
    orth = matrix_power(overlap, -0.5, hermitian=False)
    vectors = vectors @ orth.T.conj()

    if transpose:
        vectors = vectors.T.conj()

    return vectors


def biorthonormalise_with_overlap(
    left: Array,
    right: Array,
    overlap: Array,
    method: Literal["eig", "eig-balanced", "lu"] = BIORTH_METHOD,
) -> tuple[Array, Array]:
    """Biorthonormalise two sets of vectors with a given overlap matrix.

    Args:
        left: The left set of vectors.
        right: The right set of vectors.
        overlap: The overlap matrix to be used for biorthonormalisation.
        method: The method to use for biorthonormalisation. See :func:`biorthonormalise` for
            available methods.

    Returns:
        The biorthonormalised left and right sets of vectors.

    See Also:
        :func:`biorthonormalise` for details on the available methods.
    """
    if method == "eig":
        orth, error = matrix_power(overlap, -1, hermitian=False, return_error=True)
        right = right @ orth
    elif method == "eig-balanced":
        orth, error = matrix_power(overlap, -0.5, hermitian=False, return_error=True)
        left = left @ orth.T.conj()
        right = right @ orth
    elif method == "lu":
        l, u = scipy.linalg.lu(overlap, permute_l=True)
        try:
            left = left @ np.linalg.inv(l).T.conj()
            right = right @ np.linalg.inv(u)
        except np.linalg.LinAlgError as e:
            warnings.warn(
                f"Inverse of LU decomposition failed with error: {e}. "
                "Falling back to eigenvalue decomposition.",
                UserWarning,
            )
            return biorthonormalise_with_overlap(left, right, overlap, method="eig-balanced")
    else:
        raise ValueError(f"Unknown biorthonormalisation method: {method}")

    return left, right


@cache_by_id
def biorthonormalise(
    left: Array,
    right: Array,
    transpose: bool = False,
    method: Literal["eig", "eig-balanced", "lu"] = BIORTH_METHOD,
) -> tuple[Array, Array]:
    """Biorthonormalise two sets of vectors.

    Args:
        left: The left set of vectors.
        right: The right set of vectors.
        transpose: Whether to transpose the vectors before and after biorthonormalisation.
        method: The method to use for biorthonormalisation. The ``"eig"`` method uses the
            eigenvalue decomposition, the ``"eig-balanced"`` method uses the same decomposition but
            applies a balanced transformation to the left- and right-hand vectors, and the ``"lu"``
            method uses the LU decomposition.

    Returns:
        The biorthonormalised left and right sets of vectors.

    See Also:
        :func:`biorthonormalise_with_overlap` for a more general method that allows for a custom
        overlap matrix.
    """
    if transpose:
        left = left.T.conj()
        right = right.T.conj()

    overlap = left.T.conj() @ right
    left, right = biorthonormalise_with_overlap(left, right, overlap, method=method)

    if transpose:
        left = left.T.conj()
        right = right.T.conj()

    return left, right


def _sort_eigvals(eigvals: Array, eigvecs: Array, threshold: float = 1e-11) -> tuple[Array, Array]:
    """Sort eigenvalues and eigenvectors.

    Args:
        eigvals: The eigenvalues to be sorted.
        eigvecs: The eigenvectors to be sorted.
        threshold: Threshold for rounding the eigenvalues to avoid numerical noise.

    Returns:
        The sorted eigenvalues and eigenvectors.

    Note:
        The indirect sort attempts to sort the eigenvalues such that complex conjugate pairs are
        ordered correctly, regardless of any numerical noise in the real part. This is done by
        first ordering based on the rounded real and imaginary parts of the eigenvalues, and then
        sorting by the true real and imaginary parts.
    """
    decimals = round(-np.log10(threshold))
    real_approx = np.round(eigvals.real, decimals=decimals)
    imag_approx = np.round(eigvals.imag, decimals=decimals)
    idx = np.lexsort((eigvals.imag, eigvals.real, imag_approx, real_approx))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


@cache_by_id
def eig(matrix: Array, hermitian: bool = True, overlap: Array | None = None) -> tuple[Array, Array]:
    """Compute the eigenvalues and eigenvectors of a matrix.

    Args:
        matrix: The matrix to be diagonalised.
        hermitian: Whether the matrix is hermitian.
        overlap: An optional overlap matrix to be used for the eigenvalue decomposition.

    Returns:
        The eigenvalues and eigenvectors of the matrix.
    """
    # Find the eigenvalues and eigenvectors
    if AVOID_SCIPY_EIG and overlap is not None:
        matrix = np.linalg.solve(overlap, matrix)
    if AVOID_SCIPY_EIG and hermitian:
        eigvals, eigvecs = np.linalg.eigh(matrix)
    elif AVOID_SCIPY_EIG:
        eigvals, eigvecs = np.linalg.eig(matrix)
    elif hermitian:
        eigvals, eigvecs = scipy.linalg.eigh(matrix, b=overlap)
    else:
        eigvals, eigvecs = scipy.linalg.eig(matrix, b=overlap)

    # See if we can remove the imaginary part of the eigenvalues
    if not hermitian and np.all(eigvals.imag == 0.0):
        eigvals = eigvals.real

    return _sort_eigvals(eigvals, eigvecs)


@cache_by_id
def eig_lr(
    matrix: Array, hermitian: bool = True, overlap: Array | None = None
) -> tuple[Array, tuple[Array, Array]]:
    """Compute the eigenvalues and biorthogonal left- and right-hand eigenvectors of a matrix.

    Args:
        matrix: The matrix to be diagonalised.
        hermitian: Whether the matrix is hermitian.
        overlap: An optional overlap matrix to be used for the eigenvalue decomposition.

    Returns:
        The eigenvalues and biorthogonal left- and right-hand eigenvectors of the matrix.
    """
    # Find the eigenvalues and eigenvectors
    eigvals_left: Array | None = None
    if AVOID_SCIPY_EIG and hermitian:
        if overlap is not None:
            matrix = np.linalg.solve(overlap, matrix)
        eigvals, eigvecs_right = _sort_eigvals(*np.linalg.eigh(matrix))
        eigvecs_left = eigvecs_right
    elif AVOID_SCIPY_EIG:
        matrix_right = matrix
        matrix_left = matrix.T.conj()
        if overlap is not None:
            matrix_right = np.linalg.solve(overlap, matrix_right)
            matrix_left = np.linalg.solve(overlap.T.conj(), matrix_left)
        eigvals, eigvecs_right = _sort_eigvals(*np.linalg.eig(matrix_right))
        eigvals_left, eigvecs_left = np.linalg.eig(matrix_left)
        eigvals_left, eigvecs_left = _sort_eigvals(eigvals_left.conj(), eigvecs_left)
    elif hermitian:
        eigvals, eigvecs_right = _sort_eigvals(*scipy.linalg.eigh(matrix, b=overlap))
        eigvecs_left = eigvecs_right
    else:
        eigvals_raw, eigvecs_left, eigvecs_right = scipy.linalg.eig(
            matrix,
            left=True,
            right=True,
            b=overlap,
        )
        eigvals, eigvecs_right = _sort_eigvals(eigvals_raw, eigvecs_right)
        eigvals, eigvecs_left = _sort_eigvals(eigvals_raw, eigvecs_left)
    if not hermitian:
        eigvecs_left, eigvecs_right = biorthonormalise(eigvecs_left, eigvecs_right)

    # See if we can remove the imaginary part of the eigenvalues
    if not hermitian and np.all(eigvals.imag == 0.0):
        eigvals = eigvals.real

    return eigvals, (eigvecs_left, eigvecs_right)


@cache_by_id
def null_space_basis(
    matrix: Array, threshold: float = 1e-11, hermitian: bool | None = None
) -> tuple[Array, Array]:
    r"""Find a basis for the null space of a matrix.

    Args:
        matrix: The matrix for which to find the null space.
        threshold: Threshold for removing vectors to obtain the null space.
        hermitian: Whether the matrix is hermitian. If `None`, infer from the matrix.

    Returns:
        The basis for the null space.

    Note:
        The full vector space may not be biorthonormal.
    """
    if hermitian is None:
        hermitian = np.allclose(matrix, matrix.T.conj())

    # Find the null space
    null = np.eye(matrix.shape[1]) - matrix

    # Diagonalise the null space to find the basis
    weights, (left, right) = eig_lr(null, hermitian=hermitian)
    mask = (1 - np.abs(weights)) < threshold
    left = left[:, mask]
    right = right[:, mask]

    return (left, right) if hermitian else (left, left)


@cache_by_id
def matrix_power(
    matrix: Array,
    power: int | float,
    hermitian: bool = True,
    threshold: float = 1e-10,
    return_error: bool = False,
    ord: int | float = np.inf,
) -> tuple[Array, float | None]:
    """Compute the power of a matrix via the eigenvalue decomposition.

    Args:
        matrix: The matrix to be exponentiated.
        power: The power to which the matrix is to be raised.
        hermitian: Whether the matrix is hermitian.
        threshold: Threshold for removing singularities.
        return_error: Whether to return the error in the power.
        ord: The order of the norm to be used for the error.

    Returns:
        The matrix raised to the power, and the error if requested.
    """
    # TODO: Check if scipy.linalg.fractional_matrix_power is better

    # Get the eigenvalues and eigenvectors -- don't need to be biorthogonal, avoid recursive calls
    eigvals, right = eig(matrix, hermitian=hermitian)
    if hermitian:
        left = right
    else:
        left = np.linalg.inv(right).T.conj()

    # Get the mask for removing singularities
    if power < 0:
        mask = np.abs(eigvals) > threshold
    else:
        mask = np.ones_like(eigvals, dtype=bool)

    # Get the mask for removing negative eigenvalues
    if hermitian and not np.iscomplexobj(matrix):
        if np.abs(power) < 1:
            mask &= eigvals > 0
    else:
        if np.any(eigvals < 0):
            power: complex = power + 0.0j  # type: ignore[no-redef]

    # Contract the eigenvalues and eigenvectors
    matrix_power: Array = (right[:, mask] * eigvals[mask][None] ** power) @ left[:, mask].T.conj()

    # Get the error if requested
    error: float | None = None
    if return_error:
        null = (right[:, ~mask] * eigvals[~mask][None]) @ left[:, ~mask].T.conj()
        if null.size == 0:
            error = 0.0
        else:
            error = cast(float, np.linalg.norm(null, ord=ord))

    # See if we can remove the imaginary part of the matrix power
    if np.iscomplexobj(matrix_power) and np.all(np.isclose(matrix_power.imag, 0.0)):
        matrix_power = matrix_power.real

    return matrix_power, error


def hermi_sum(matrix: Array) -> Array:
    """Return the sum of a matrix with its Hermitian conjugate.

    Args:
        matrix: The matrix to be summed with its hermitian conjugate.

    Returns:
        The sum of the matrix with its hermitian conjugate.
    """
    return matrix + matrix.T.conj()


def scaled_error(matrix1: Array, matrix2: Array, ord: int | float = np.inf) -> float:
    """Return the scaled error between two matrices.

    Args:
        matrix1: The first matrix.
        matrix2: The second matrix.
        ord: The order of the norm to be used for the error.

    Returns:
        The scaled error between the two matrices.
    """
    matrix1 = np.atleast_1d(matrix1 / max(np.max(np.abs(matrix1)), 1))
    matrix2 = np.atleast_1d(matrix2 / max(np.max(np.abs(matrix2)), 1))
    return cast(float, np.linalg.norm(matrix1 - matrix2, ord=ord))


def as_trace(matrix: Array, ndim: int, axis1: int = -2, axis2: int = -1) -> Array:
    """Return the trace of a matrix, unless it has been passed as a trace.

    Args:
        matrix: The matrix to be traced.
        ndim: The number of dimensions of the matrix before the trace.
        axis1: The first axis of the trace.
        axis2: The second axis of the trace.

    Returns:
        The trace of the matrix.
    """
    if matrix.ndim == ndim:
        return matrix
    elif matrix.ndim > ndim:
        return np.trace(matrix, axis1=axis1, axis2=axis2)
    else:
        raise ValueError(f"Matrix has invalid shape {matrix.shape} for trace.")


def as_diagonal(matrix: Array, ndim: int) -> Array:
    """Return the diagonal of a matrix, unless it has been passed as a diagonal.

    Args:
        matrix: The matrix to be diagonalised.
        ndim: The number of dimensions of the matrix before the diagonal.

    Returns:
        The diagonal of the matrix.
    """
    if matrix.ndim == ndim:
        return matrix
    elif matrix.ndim > ndim:
        return np.diagonal(matrix, axis1=-2, axis2=-1)
    else:
        raise ValueError(f"Matrix has invalid shape {matrix.shape} for diagonal.")


def unit_vector(size: int, index: int, dtype: str = "float64") -> Array:
    """Return a unit vector of size `size` with a 1 at index `index`.

    Args:
        size: The size of the vector.
        index: The index of the vector.
        dtype: The data type of the vector.

    Returns:
        The unit vector.
    """
    return np.eye(1, size, k=index, dtype=dtype).ravel()


def concatenate_paired_vectors(vectors: list[Array], size: int) -> Array:
    r"""Concatenate vectors that are partitioned into two spaces, the first of which is common.

    Args:
        vectors: The vectors to be concatenated.
        size: The size of the first space.

    Returns:
        The concatenated vectors.

    Note:
        The concatenation is

        .. math::
            \begin{pmatrix}
                p_1 & p_2 & \cdots & p_n \\
                a_1 &     &        &     \\
                    & a_2 &        &     \\
                    &     & \ddots &     \\
                    &     &        & a_n \\
            \end{pmatrix}
            =
            \begin{pmatrix} p_1 \\ a_1 \end{pmatrix}
            + \begin{pmatrix} p_2 \\ a_2 \end{pmatrix}
            + \cdots
            + \begin{pmatrix} p_n \\ a_n \end{pmatrix}

        where :math:`p_i` are the vectors in the first space and :math:`a_i` are the vectors in the
        second space.

        This is useful for combining couplings between a common physical space and a set of
        auxiliary degrees of freedom.
    """
    space1 = slice(0, size)
    space2 = slice(size, None)
    vectors1 = np.concatenate([vector[space1] for vector in vectors], axis=1)
    vectors2 = block_diag(*[vector[space2] for vector in vectors])
    return np.concatenate([vectors1, vectors2], axis=0)


def unpack_vectors(vector: Array) -> tuple[Array, Array]:
    """Unpack a block vector in the :mod:`dyson` convention.

    Args:
        vector: The vector to be unpacked. The vector should either be a 2D array `(n, m)` or a 3D
            array `(2, n, m)`. The latter case is non-Hermitian.

    Returns:
        Left- and right-hand vectors.
    """
    if vector.ndim == 2:
        return vector, vector
    elif vector.ndim == 3:
        return vector[0], vector[1]
    raise ValueError(
        f"Vector has invalid shape {vector.shape} for unpacking. Must be 2D or 3D array."
    )


def block_diag(*arrays: Array) -> Array:
    """Return a block diagonal matrix from a list of arrays.

    Args:
        arrays: The arrays to be combined into a block diagonal matrix.

    Returns:
        The block diagonal matrix.
    """
    if not all(array.ndim == 2 for array in arrays):
        raise ValueError("All arrays must be 2D.")
    rows = [array.shape[0] for array in arrays]
    cols = [array.shape[1] for array in arrays]
    arrays_full = [[np.zeros((row, col)) for col in cols] for row in rows]
    for i, array in enumerate(arrays):
        arrays_full[i][i] = array
    return np.block(arrays_full)


def set_subspace(vectors: Array, subspace: Array) -> Array:
    """Set the subspace of a set of vectors.

    Args:
        vectors: The vectors to be set.
        subspace: The subspace to be applied to the vectors.

    Returns:
        The vectors with the subspace applied.

    Note:
        This operation is equivalent to applying `vectors[: n] = subspace` where `n` is the size of
        both dimensions in the subspace.
    """
    size = subspace.shape[0]
    return np.concatenate([subspace, vectors[size:]], axis=0)


def rotate_subspace(vectors: Array, rotation: Array) -> Array:
    """Rotate the subspace of a set of vectors.

    Args:
        vectors: The vectors to be rotated.
        rotation: The rotation matrix to be applied to the vectors.

    Returns:
        The rotated vectors.

    Note:
        This operation is equivalent to applying `vectors[: n] = rotation @ vectors[: n]` where `n`
        is the size of both dimensions in the rotation matrix.
    """
    if rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"Rotation matrix must be square, got shape {rotation.shape}.")
    size = rotation.shape[0]
    subspace = rotation @ vectors[:size]
    return set_subspace(vectors, subspace)
