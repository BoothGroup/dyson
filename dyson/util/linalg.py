"""Linear algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
    eigvals, eigvecs = eig(matrix, hermitian=hermitian)

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

    # Get the left and right eigenvalues
    if hermitian:
        left = right = eigvecs
    else:
        left = eigvecs
        right = np.linalg.inv(eigvecs).T.conj()

    # Contract the eigenvalues and eigenvectors
    matrix_power: Array = (left[:, mask] * eigvals[mask][None] ** power) @ right[:, mask].T.conj()

    # Get the error if requested
    if return_error:
        null = (left[:, ~mask] * eigvals[~mask][None] ** power) @ right[:, ~mask].T.conj()
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
