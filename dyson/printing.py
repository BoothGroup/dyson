"""Printing utilities."""

from __future__ import annotations

import importlib
import os
import subprocess
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.errors import LiveError
from rich.progress import Progress
from rich.table import Table
from rich.theme import Theme

from dyson import __version__

if TYPE_CHECKING:
    from typing import Any, Literal

    from rich.progress import TaskID


theme = Theme(
    {
        "good": "green",
        "okay": "yellow",
        "bad": "red",
        "output": "cyan",
        "input": "bright_magenta",
        "method": "bold underline",
        "header": "bold",
    }
)

console = Console(
    highlight=False,
    theme=theme,
    log_path=False,
    quiet=os.environ.get("DYSON_QUIET", "").lower() in ("1", "true"),
)

HEADER = r"""     _
    | |
  __| | _   _  ___   ___   _ __
 / _` || | | |/ __| / _ \ | '_ \
| (_| || |_| |\__ \| (_) || | | |
 \__,_| \__, ||___/ \___/ |_| |_|
         __/ |
        |___/  %s
"""


def init_console() -> None:
    """Initialise the console with a header."""
    if globals().get("_DYSON_LOG_INITIALISED", False):
        return

    # Print header
    header_with_version = "[header]" + HEADER + "[/header]"
    header_with_version %= " " * (18 - len(__version__)) + "[input]" + __version__ + "[/input]"
    console.print(header_with_version)

    # Print versions of dependencies and ebcc
    def get_git_hash(directory: str) -> str:
        git_directory = os.path.join(directory, ".git")
        cmd = ["git", "--git-dir=%s" % git_directory, "rev-parse", "--short", "HEAD"]
        try:
            git_hash = subprocess.check_output(
                cmd, universal_newlines=True, stderr=subprocess.STDOUT
            ).rstrip()
        except subprocess.CalledProcessError:
            git_hash = "N/A"
        return git_hash

    for name in ["numpy", "pyscf", "dyson"]:
        module = importlib.import_module(name)
        if module.__file__ is None:
            git_hash = "N/A"
        else:
            git_hash = get_git_hash(os.path.join(os.path.dirname(module.__file__), ".."))
        console.print(f"{name}:")
        console.print(f" > Version:  [input]{module.__version__}[/]")
        console.print(f" > Git hash: [input]{git_hash}[/]")

    console.print("OMP_NUM_THREADS = [input]%s[/]" % os.environ.get("OMP_NUM_THREADS", ""))

    globals()["_DYSON_LOG_INITIALISED"] = True


class Quiet:
    """Context manager to disable console output."""

    def __init__(self, console: Console = console):
        """Initialise the object."""
        self._memo: list[bool] = []
        self._console = console

    def __enter__(self) -> None:
        """Enter the context manager."""
        self._memo.append(self.console.quiet)
        self.console.quiet = True

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager."""
        quiet = self._memo.pop()
        self.console.quiet = quiet

    def __call__(self) -> None:
        """Call the context manager."""
        self.console.quiet = True

    @property
    def console(self) -> Console:
        """Get the console."""
        return self._console


quiet = Quiet(console)


def rate_error(
    value: float | complex, threshold: float, threshold_okay: float | None = None
) -> Literal["good", "okay", "bad"]:
    """Rate the error based on a threshold.

    Args:
        value: The value to rate.
        threshold: The threshold for the rating.
        threshold_okay: Separate threshold for `"okay"` rating. Default is 10 times
            :param:`threshold`.

    Returns:
        str: The rating, one of "good", "okay", or "bad".
    """
    if threshold_okay is None:
        threshold_okay = 10 * threshold
    if abs(value) < threshold:
        return "good"
    elif abs(value) < threshold_okay:
        return "okay"
    else:
        return "bad"


def format_float(
    value: float | complex | None,
    precision: int = 10,
    scientific: bool = False,
    threshold: float | None = None,
) -> str:
    """Format a float or complex number to a string with a given precision.

    Args:
        value: The value to format.
        precision: The number of decimal places to include.
        scientific: Whether to use scientific notation for large or small values.
        threshold: If provided, the value will be rated based on this threshold.

    Returns:
        str: The formatted string.
    """
    if isinstance(value, complex):
        real = format_float(value.real, precision, scientific, threshold)
        if abs(value.imag) < (1e-1**precision):
            return real
        sign = "+" if value.imag >= 0 else "-"
        imag = format_float(abs(value.imag), precision, scientific, threshold)
        return f"{real}{sign}{imag}i"
    if value is None:
        return "N/A"
    if value.imag < (1e-1**precision):
        value = value.real
    out = f"{value:.{precision}g}" if scientific else f"{value:.{precision}f}"
    if threshold is not None:
        rating = rate_error(value, threshold)
        out = f"[{rating}]{out}[/]"
    return out


class ConvergencePrinter:
    """Table for printing convergence information."""

    def __init__(
        self,
        quantities: tuple[str, ...],
        quantity_errors: tuple[str, ...],
        thresholds: tuple[float, ...],
        console: Console = console,
        cycle_name: str = "Cycle",
    ):
        """Initialise the object."""
        self._console = console
        self._table = Table(box=box.SIMPLE)
        self._table.add_column(cycle_name, style="dim", justify="left")
        for quantity in quantities:
            self._table.add_column(quantity, justify="right")
        for quantity_error in quantity_errors:
            self._table.add_column(quantity_error, justify="right")
        self._thresholds = thresholds

    def add_row(
        self,
        cycle: int,
        quantities: tuple[float | None, ...],
        quantity_errors: tuple[float | None, ...],
    ) -> None:
        """Add a row to the table."""
        self._table.add_row(
            str(cycle),
            *[format_float(quantity) for quantity in quantities],
            *[
                format_float(error, precision=4, scientific=True, threshold=threshold)
                for error, threshold in zip(quantity_errors, self._thresholds)
            ],
        )

    def print(self) -> None:
        """Print the table."""
        self._console.print(self._table)

    @property
    def thresholds(self) -> tuple[float, ...]:
        """Get the thresholds."""
        return self._thresholds


class IterationsPrinter:
    """Progress bar for iterations."""

    def __init__(self, max_cycle: int, console: Console = console, description: str = "Iteration"):
        """Initialise the object."""
        self._max_cycle = max_cycle
        self._console = console
        self._description = description
        self._ignore = False
        self._progress = Progress(transient=True)
        self._task: TaskID | None = None

    def start(self) -> None:
        """Start the progress bar."""
        if self.console.quiet:
            return
        self._ignore = False
        try:
            self.progress.start()
        except LiveError:
            # If there is already a live print, don't start a progress bar
            self._ignore = True
            return
        self._task = self.progress.add_task(
            f"{self.description} 0 / {self.max_cycle}", total=self.max_cycle
        )

    def update(self, cycle: int) -> None:
        """Update the progress bar for the given cycle."""
        if self.console.quiet or self._ignore:
            return
        if self.task is None:
            raise RuntimeError("Progress bar has not been started. Call start() first.")
        self.progress.update(
            self.task, advance=1, description=f"{self.description} {cycle} / {self.max_cycle}"
        )

    def stop(self) -> None:
        """Stop the progress bar."""
        if self.console.quiet or self._ignore:
            return
        if self.task is None:
            raise RuntimeError("Progress bar has not been started. Call start() first.")
        self.progress.stop()

    @property
    def max_cycle(self) -> int:
        """Get the maximum number of cycles."""
        return self._max_cycle

    @property
    def console(self) -> Console:
        """Get the console."""
        return self._console

    @property
    def description(self) -> str:
        """Get the description of the progress bar."""
        return self._description

    @property
    def progress(self) -> Progress:
        """Get the progress bar."""
        return self._progress

    @property
    def task(self) -> TaskID | None:
        """Get the current task."""
        return self._task
