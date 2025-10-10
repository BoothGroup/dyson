"""Fourier transformation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import util
from dyson import numpy as np
from dyson.representations.enums import Component

if TYPE_CHECKING:
    from dyson.representations.dynamic import Dynamic
    from dyson.grids.frequency import GridIF
    from dyson.grids.time import GridIT
    from dyson.typing import Array


def _apply_imag_factor(
    array: Array, grid_it: GridIT, grid_if: GridIF, inverse: bool = False
) -> Array:
    """Apply the exponential factor for Fourier transform on imaginary axes."""
    factor = np.exp(1j * np.pi * grid_it.points / grid_if.beta) * grid_if.beta
    if inverse:
        factor = 1 / factor
    return util.einsum("w...,w->w...", array, factor)


def fourier_transform_imag(
    greens_function_it: Dynamic[GridIT],
    grid_if: GridIF,
    tail_moments: tuple[Array, ...] | None = None,
) -> Dynamic[GridIF]:
    """Forward Fourier transform from imaginary time to imaginary frequency domain.

    Args:
        gf_if: Dynamic quantity in imaginary time domain.
        tail_moments: Moments of the high-frequency tail.

    Returns:
        Dynamic quantity in imaginary frequency domain.
    """
    grid_it = greens_function_it.grid
    if not np.isclose(grid_it.beta, grid_if.beta):
        raise ValueError("the beta of the two grids must be the same.")
    if not (grid_it.uniformly_spaced and grid_if.uniformly_spaced):
        raise ValueError("only uniform grids are supported.")
    if not (grid_it.uniformly_weighted and grid_if.uniformly_weighted):
        raise ValueError("only uniform weights are supported.")
    if greens_function_it.component != Component.FULL:
        raise ValueError("only FFT for full component is supported.")
    if tail_moments is None:
        tail_moments = (np.eye(greens_function_it.nphys),)

    # Subtract tail (treated analytically)
    array_it = greens_function_it.array - grid_it.evaluate_tail(tail_moments)

    # Perform FFT
    array_it = _apply_imag_factor(array_it, grid_it, grid_if, inverse=False)
    array_if = np.fft.fft(array_it, len(grid_if), axis=0)

    # Add analytic tail
    array_if += grid_if.evaluate_tail(tail_moments)

    # Include normalisation from grid weights
    array_if *= np.sum(grid_if.weights) / np.sum(grid_it.weights)

    return greens_function_it.__class__(
        grid_if,
        array_if,
        reduction=greens_function_it.reduction,
        hermitian=greens_function_it.hermitian,
    )


def inverse_fourier_transform_imag(
    greens_function_if: Dynamic[GridIF],
    grid_it: GridIT,
    tail_moments: tuple[Array, ...] | None = None,
) -> Dynamic[GridIT]:
    """Inverse Fourier transform from imaginary frequency to imaginary time domain.

    Args:
        gf_if: Dynamic quantity in imaginary frequency domain.
        tail_moments: Moments of the high-frequency tail.

    Returns:
        Dynamic quantity in imaginary time domain.
    """
    grid_if = greens_function_if.grid
    if not np.isclose(grid_if.beta, grid_it.beta):
        raise ValueError("the beta of the two grids must be the same.")
    if greens_function_if.component != Component.FULL:
        raise ValueError("only IFFT for full component is supported.")
    if not (grid_it.uniformly_spaced and grid_if.uniformly_spaced):
        raise ValueError("only uniform grids are supported.")
    if not (grid_it.uniformly_weighted and grid_if.uniformly_weighted):
        raise ValueError("only uniform weights are supported.")
    if tail_moments is None:
        tail_moments = (np.eye(greens_function_if.nphys),)

    # Subtract tail (treated analytically)
    array_if = greens_function_if.array - grid_if.evaluate_tail(tail_moments)

    # Perform IFFT
    array_it = np.fft.ifft(array_if, len(grid_it), axis=0)
    array_it = _apply_imag_factor(array_it, grid_it, grid_if, inverse=True)

    # Add analytic tail
    array_it += grid_it.evaluate_tail(tail_moments)

    # Include normalisation from grid weights
    array_it *= np.sum(grid_it.weights) / np.sum(grid_if.weights)

    return greens_function_if.__class__(
        grid_it,
        array_it,
        reduction=greens_function_if.reduction,
        hermitian=greens_function_if.hermitian,
    )
