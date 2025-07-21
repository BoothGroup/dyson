"""Plotting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import scipy.constants

from dyson import numpy as np
from dyson.representations.enums import Component, Reduction

if TYPE_CHECKING:
    from typing import Any, Literal

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    from dyson.grids.frequency import BaseFrequencyGrid
    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann


theme = {
    # Lines
    "lines.linewidth": 3.0,
    "lines.markersize": 10.0,
    "lines.markeredgewidth": 1.0,
    "lines.markeredgecolor": "black",
    # Font
    "font.size": 12,
    "font.family": "sans-serif",
    "font.weight": "medium",
    # Axes
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    "axes.facecolor": "whitesmoke",
    "axes.linewidth": 1.5,
    "axes.unicode_minus": False,
    "axes.prop_cycle": plt.cycler(
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    ),
    # Ticks
    "xtick.labelsize": 12,
    "xtick.major.pad": 7,
    "xtick.major.size": 7,
    "xtick.major.width": 1.2,
    "xtick.minor.size": 4,
    "xtick.minor.width": 0.6,
    "ytick.labelsize": 12,
    "ytick.major.pad": 7,
    "ytick.major.size": 7,
    "ytick.major.width": 1.2,
    "ytick.minor.size": 4,
    "ytick.minor.width": 0.6,
    # Grid
    "grid.linewidth": 1.3,
    "grid.alpha": 0.5,
    # Legend
    "legend.fontsize": 11,
    # Figure
    "figure.figsize": (8, 6),
    "figure.facecolor": "white",
    "figure.autolayout": True,
    # LaTeX
    "pgf.texsystem": "pdflatex",
}

plt.rcParams.update(theme)


def _unit_name(unit: str) -> str:
    """Return the name of the unit for SciPy."""
    if unit == "Ha":
        return "hartree"
    elif unit == "eV":
        return "electron volt"
    else:
        raise ValueError(f"Unknown energy unit: {unit}. Use 'Ha' or 'eV'.")


def _convert(energy: float, unit_from: str, unit_to: str) -> float:
    """Convert energies between Hartree and eV."""
    if unit_from == unit_to:
        return energy
    unit_from = _unit_name(unit_from)
    unit_to = _unit_name(unit_to)
    return energy * scipy.constants.physical_constants[f"{unit_from}-{unit_to} relationship"][0]


def plot_lehmann(
    lehmann: Lehmann,
    ax: Axes | None = None,
    energy_unit: Literal["Ha", "eV"] = "eV",
    height_by_weight: bool = True,
    height_factor: float = 1.0,
    fmt: str = "k-",
    **kwargs: Any,
) -> list[Line2D]:
    """Plot a Lehmann representation as delta functions.

    Args:
        lehmann: The Lehmann representation to plot.
        ax: The axes to plot on. If ``None``, a new figure and axes are created.
        energy_unit: The unit of the energy values.
        height_by_weight: If ``True``, the height of each delta function is scaled by its weight.
            If ``False``, all delta functions have the same height.
        height_factor: A factor to scale the height of the delta functions.
        fmt: The format string for the lines.
        **kwargs: Additional keyword arguments passed to ``ax.plot``.

    Returns:
        A list of Line2D objects representing the plotted delta functions.
    """
    if ax is None:
        fig, ax = plt.subplots()
    lines: list[Line2D] = []
    for i, (energy, weight) in enumerate(zip(lehmann.energies, lehmann.weights())):
        energy = _convert(energy, "Ha", energy_unit)
        height = weight * height_factor if height_by_weight else height_factor
        lines += ax.plot([energy, energy], [0, height], fmt, **kwargs)
    return lines


def plot_dynamic(
    dynamic: Dynamic,
    ax: Axes | None = None,
    energy_unit: Literal["Ha", "eV"] = "eV",
    normalise: bool = False,
    height_factor: float = 1.0,
    fmt: str = "k-",
    **kwargs: Any,
) -> list[Line2D]:
    """Plot a dynamic representation as a line plot.

    Args:
        dynamic: The dynamic representation to plot.
        ax: The axes to plot on. If ``None``, a new figure and axes are created.
        energy_unit: The unit of the energy values.
        normalise: If ``True``, the representation is normalised to have a maximum value of 1.
        height_factor: A factor to scale the height of the line.
        fmt: The format string for the lines.
        **kwargs: Additional keyword arguments passed to ``ax.plot``.

    Returns:
        A list of Line2D objects representing the plotted dynamic.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if dynamic.reduction != Reduction.TRACE:
        raise ValueError(
            f"Dynamic object reduction must be {Reduction.TRACE.name} to plot as a line plot, but "
            f"got {dynamic.reduction.name}. If you intended to plot the trace, use "
            '`dynamic.copy(reduction="trace")` to create a copy with the trace reduction.'
        )
    if dynamic.component == Component.FULL:
        raise ValueError(
            f"Dynamic object component must be {Component.REAL.name} or {Component.IMAG.name} to "
            f"plot as a line plot, but got {dynamic.component.name}. If you intended to plot the "
            'real or imaginary part, use `dynamic.copy(component="real")` or '
            '`dynamic.copy(component="imag")` to create a copy with the desired component.'
        )
    grid = _convert(dynamic.grid, "Ha", energy_unit)
    array = dynamic.array
    if normalise:
        array = array / np.max(np.abs(array))
    return ax.plot(grid, array * height_factor, fmt, **kwargs)


def format_axes_spectral_function(
    grid: BaseFrequencyGrid,
    ax: Axes | None = None,
    energy_unit: Literal["Ha", "eV"] = "eV",
    xlabel: str = "Frequency ({})",
    ylabel: str = "Spectral function",
) -> None:
    """Format the axes for a spectral function plot.

    Args:
        grid: The frequency grid used for the spectral function.
        ax: The axes to format. If ``None``, the current axes are used.
        energy_unit: The unit of the energy values.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(xlabel.format(energy_unit))
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.set_xlim(_convert(grid.min(), "Ha", energy_unit), _convert(grid.max(), "Ha", energy_unit))


def unknown_pleasures(dynamics: list[Dynamic]) -> Axes:
    """Channel your inner Ian Curtis."""
    fig, ax = plt.subplots(figsize=(5, 7), facecolor="black")
    norm = max([np.max(np.abs(d.array)) for d in dynamics])
    xmin = min([d.grid.min() for d in dynamics])
    xmax = max([d.grid.max() for d in dynamics])
    xmin -= (xmax - xmin) * 0.05  # Add some padding
    xmax += (xmax - xmin) * 0.05  # Add some padding
    ymax = 0.0
    spacing = 0.2
    zorder = 1
    for i, dynamic in list(enumerate(dynamics))[::-1]:
        grid = _convert(dynamic.grid, "Ha", "eV")
        array = dynamic.array / norm
        array += i * spacing
        array += np.random.uniform(-0.015, 0.015, size=array.shape)  # Add some noise
        ymax = max(ymax, np.max(array))
        ax.fill_between(grid, i * spacing, array, color="k", zorder=zorder)
        ax.plot(grid, array, "-", color="white", linewidth=2.0, zorder=zorder + 1)
        zorder += 2
    ax.axis("off")
    ax.set_xlim(_convert(xmin, "Ha", "eV"), _convert(xmax, "Ha", "eV"))
    ax.set_ylim(-0.1, ymax + spacing)
    return ax
