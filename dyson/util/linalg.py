"""Linear algebra."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast, overload

import scipy.linalg

from dyson import numpy as np

if TYPE_CHECKING:
    from dyson.typing import Array

einsum = functools.partial(np.einsum, optimize=True)


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
    vectors = vectors @ orth
    if transpose:
        vectors = vectors.T.conj()
    return vectors


def biorthonormalise(left: Array, right: Array, transpose: bool = False) -> tuple[Array, Array]:
    """Biorthonormalise two sets of vectors.

    Args:
        left: The left set of vectors.
        right: The right set of vectors.
        transpose: Whether to transpose the vectors before and after biorthonormalisation.

    Returns:
        The biorthonormalised left and right sets of vectors.
    """
    if transpose:
        left = left.T.conj()
        right = right.T.conj()
    overlap = left.T.conj() @ right
    orth, error = matrix_power(overlap, -1, hermitian=False, return_error=True)
    right = right @ orth
    if transpose:
        left = left.T.conj()
        right = right.T.conj()
    return left, right


def eig(matrix: Array, hermitian: bool = True) -> tuple[Array, Array]:
    """Compute the eigenvalues and eigenvectors of a matrix.

    Args:
        matrix: The matrix to be diagonalised.
        hermitian: Whether the matrix is hermitian.

    Returns:
        The eigenvalues and eigenvectors of the matrix.
    """
    # Find the eigenvalues and eigenvectors
    if hermitian:
        # assert np.allclose(m, m.T.conj())
        eigvals, eigvecs = np.linalg.eigh(matrix)
    else:
        eigvals, eigvecs = np.linalg.eig(matrix)

    # Sort the eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs


def eig_lr(matrix: Array, hermitian: bool = True) -> tuple[Array, tuple[Array, Array]]:
    """Compute the eigenvalues and biorthogonal left- and right-hand eigenvectors of a matrix.

    Args:
        matrix: The matrix to be diagonalised.
        hermitian: Whether the matrix is hermitian.

    Returns:
        The eigenvalues and biorthogonal left- and right-hand eigenvectors of the matrix.
    """
    # Find the eigenvalues and eigenvectors
    if hermitian:
        eigvals, eigvecs_left = np.linalg.eigh(matrix)
        eigvecs_right = eigvecs_left
    else:
        eigvals, eigvecs_left, eigvecs_right = scipy.linalg.eig(matrix, left=True, right=True)
        eigvecs_left, eigvecs_right = biorthonormalise(eigvecs_left, eigvecs_right)

    # Sort the eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_left = eigvecs_left[:, idx]
    eigvecs_right = eigvecs_right[:, idx]

    return eigvals, (eigvecs_left, eigvecs_right)


def null_space_basis(
    bra: Array, ket: Array | None = None, threshold: float = 1e-11
) -> tuple[Array, Array]:
    r"""Find a basis for the null space of :math:`\langle \text{bra} | \text{ket} \rangle`.

    Args:
        bra: The bra vectors.
        ket: The ket vectors. If `None`, use the same vectors as `bra`.
        threshold: Threshold for removing vectors to obtain the null space.

    Returns:
        The basis for the null space for the `bra` and `ket` vectors.

    Note:
        The full vector space may not be biorthonormal.
    """
    hermitian = ket is None or bra is ket
    if ket is None:
        ket = bra

    # Find the null space
    proj = bra.T.conj() @ ket
    null = np.eye(bra.shape[1]) - proj

    # Diagonalise the null space to find the basis
    weights, (left, right) = eig_lr(null, hermitian=hermitian)
    mask = (1 - np.abs(weights)) < 1e-10
    left = left[:, mask].T.conj()
    right = right[:, mask].T.conj()

    return (left, right) if hermitian else (left, left)


def matrix_power(
    matrix: Array,
    power: int | float,
    hermitian: bool = True,
    threshold: float = 1e-10,
    return_error: bool = False,
    ord: int | float = np.inf,
) -> Array | tuple[Array, float]:
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
        power: complex = power + 0.0j  # type: ignore[no-redef]

    # Contract the eigenvalues and eigenvectors
    matrix_power: Array = (right[:, mask] * eigvals[mask][None] ** power) @ left[:, mask].T.conj()

    # Get the error if requested
    if return_error:
        null = (right[:, ~mask] * eigvals[~mask][None]) @ left[:, ~mask].T.conj()
        if null.size == 0:
            error = 0.0
        else:
            error = cast(float, np.linalg.norm(null, ord=ord))

    return (matrix_power, error) if return_error else matrix_power


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

    Returns:
        The scaled error between the two matrices.
    """
    matrix1 = matrix1 / max(np.max(np.abs(matrix1)), 1)
    matrix2 = matrix2 / max(np.max(np.abs(matrix2)), 1)
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
    elif (matrix.ndim + 2) == ndim:
        return np.trace(matrix, axis1=axis1, axis2=axis2)
    else:
        raise ValueError(f"Matrix has invalid shape {matrix.shape} for trace.")


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
    vectors2 = scipy.linalg.block_diag(*[vector[space2] for vector in vectors])
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
