"""Linear algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import scipy.linalg

from dyson import numpy as np

if TYPE_CHECKING:
    from dyson.typing import Array


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


def eig_biorth(matrix: Array, hermitian: bool = True) -> tuple[Array, tuple[Array, Array]]:
    """Compute the eigenvalues and biorthogonal eigenvectors of a matrix.

    Args:
        matrix: The matrix to be diagonalised.
        hermitian: Whether the matrix is hermitian.

    Returns:
        The eigenvalues and biorthogonal eigenvectors of the matrix.
    """
    # Find the eigenvalues and eigenvectors
    if hermitian:
        eigvals, eigvecs_left = np.linalg.eigh(matrix)
        eigvecs_right = eigvecs_left
    else:
        eigvals, eigvecs_left, eigvecs_right = scipy.linalg.eig(matrix, left=True, right=True)
        norm = eigvecs_right.T.conj() @ eigvecs_left
        eigvecs_left = eigvecs_left @ np.linalg.inv(norm)

    # Sort the eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs_left = eigvecs_left[:, idx]
    eigvecs_right = eigvecs_right[:, idx]

    return eigvals, (eigvecs_left, eigvecs_right)


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
    # Get the eigenvalues and eigenvectors
    eigvals, (left, right) = eig_biorth(matrix, hermitian=hermitian)

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
        null = (right[:, ~mask] * eigvals[~mask][None] ** power) @ left[:, ~mask].T.conj()
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
