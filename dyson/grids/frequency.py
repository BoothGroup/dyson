"""Frequency grids."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import scipy.special

from dyson import numpy as np
from dyson import util
from dyson.grids.grid import BaseGrid

if TYPE_CHECKING:
    from typing import Any, Literal

    from dyson.lehmann import Lehmann
    from dyson.typing import Array


class BaseFrequencyGrid(BaseGrid):
    """Base class for frequency grids."""

    def evaluate_lehmann(self, lehmann: Lehmann, trace: bool = False, **kwargs: Any) -> Array:
        r"""Evaluate a Lehmann representation on the grid.

        The imaginary frequency representation is defined as

        .. math::
            \sum_{k} \frac{v_{pk} u_{qk}^*}{i \omega - \epsilon_k},

        and the real frequency representation is defined as

        .. math::
            \sum_{k} \frac{v_{pk} u_{qk}^*}{\omega - \epsilon_k \pm i \eta},

        where :math:`\omega` is the frequency grid, :math:`\epsilon_k` are the poles, and the sign
        of the broadening factor is determined by the time ordering.

        Args:
            lehmann: Lehmann representation to evaluate.
            trace: Return only the trace of the evaluated Lehmann representation.
            kwargs: Additional keyword arguments for the resolvent.

        Returns:
            Lehmann representation, realised on the grid.
        """
        left, right = lehmann.unpack_couplings()
        resolvent = self.resolvent(lehmann.energies, lehmann.chempot, **kwargs)
        inp, out = ("qk", "wpq") if not trace else ("pk", "w")
        return util.einsum(f"pk,{inp},wk->{out}", right, left.conj(), resolvent)

    @property
    def domain(self) -> str:
        """Get the domain of the grid.

        Returns:
            Domain of the grid.
        """
        return "frequency"

    @abstractmethod
    def resolvent(  # noqa: D417
        self, energies: Array, chempot: float | Array, **kwargs: Any
    ) -> Array:
        """Get the resolvent of the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.

        Returns:
            Resolvent of the grid.
        """
        pass


class RealFrequencyGrid(BaseFrequencyGrid):
    """Real frequency grid."""

    _eta: float = 1e-2

    def __new__(cls, *args: Any, eta: float | None = None, **kwargs: Any) -> RealFrequencyGrid:
        """Create a new instance of the grid.

        Args:
            args: Positional arguments for :class:`BaseGrid`.
            eta: Broadening factor, used as a small imaginary part to shift poles away from the real
                axis.
            kwargs: Keyword arguments for :class:`BaseGrid`.

        Returns:
            New instance of the grid.
        """
        obj = super().__new__(cls, *args, **kwargs).view(cls)
        if eta is not None:
            obj._eta = eta
        return obj

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return True

    @property
    def eta(self) -> float:
        """Get the broadening factor.

        Returns:
            Broadening factor.
        """
        return self._eta

    @eta.setter
    def eta(self, value: float) -> None:
        """Set the broadening factor.

        Args:
            value: Broadening factor.
        """
        self._eta = value

    @staticmethod
    def _resolvent_signs(
        energies: Array, ordering: Literal["time-ordered", "advanced", "retarded"]
    ) -> Array:
        """Get the signs for the resolvent based on the time ordering."""
        if ordering == "time-ordered":
            signs = np.where(energies >= 0, 1.0, -1.0)
        elif ordering == "advanced":
            signs = -np.ones_like(energies)
        elif ordering == "retarded":
            signs = np.ones_like(energies)
        else:
            raise ValueError(
                f"Invalid ordering: {ordering}. Must be 'time-ordered', 'advanced', or 'retarded'."
            )
        return signs

    def resolvent(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Literal["time-ordered", "advanced", "retarded"] = "time-ordered",
        invert: bool = True,
        **kwargs: Any,
    ) -> Array:
        r"""Get the resolvent of the grid.

        For real frequency grids, the resolvent is given by

        .. math::
            R(\omega) = \frac{1}{\omega - E \pm i \eta},

        where :math:`\eta` is a small broadening factor, and :math:`E` are the pole energies. The
        sign of :math:`i \eta` depends on the time ordering of the resolvent.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.
            invert: Whether to apply the inversion in the resolvent formula.

        Returns:
            Resolvent of the grid.
        """
        if kwargs:
            raise TypeError(f"resolvent() got unexpected keyword argument: {next(iter(kwargs))}")
        signs = self._resolvent_signs(energies - chempot, ordering)
        grid = np.expand_dims(self, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        denominator = grid + (signs * 1.0j * self.eta - energies)
        return 1.0 / denominator if invert else denominator

    @classmethod
    def from_uniform(
        cls, start: float, stop: float, num: int, eta: float | None = None
    ) -> RealFrequencyGrid:
        """Create a uniform real frequency grid.

        Args:
            start: Start of the grid.
            stop: End of the grid.
            num: Number of points in the grid.
            eta: Broadening factor.

        Returns:
            Uniform real frequency grid.
        """
        grid = np.linspace(start, stop, num, endpoint=True).view(cls)
        if eta is not None:
            grid.eta = eta
        return grid

    def __array_finalize__(self, obj: Array | None, *args: Any, **kwargs: Any) -> None:
        """Finalize the array.

        Args:
            obj: Array to finalize.
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """
        if obj is None:
            return
        super().__array_finalize__(obj, *args, **kwargs)
        self._weights = getattr(obj, "_weights", None)
        self._eta = getattr(obj, "_eta", RealFrequencyGrid._eta)


GridRF = RealFrequencyGrid


class ImaginaryFrequencyGrid(BaseFrequencyGrid):
    """Imaginary frequency grid."""

    _beta: float = 256

    def __new__(
        cls, *args: Any, beta: float | None = None, **kwargs: Any
    ) -> ImaginaryFrequencyGrid:
        """Create a new instance of the grid.

        Args:
            args: Positional arguments for :class:`BaseGrid`.
            beta: Inverse temperature.
            kwargs: Keyword arguments for :class:`BaseGrid`.

        Returns:
            New instance of the grid.
        """
        obj = super().__new__(cls, *args, **kwargs).view(cls)
        if beta is not None:
            obj._beta = beta
        return obj

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return False

    @property
    def beta(self) -> float:
        """Get the inverse temperature.

        Returns:
            Inverse temperature.
        """
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the inverse temperature.

        Args:
            value: Inverse temperature.
        """
        self._beta = value

    def resolvent(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        invert: bool = True,
        **kwargs: Any,
    ) -> Array:
        r"""Get the resolvent of the grid.

        For imaginary frequency grids, the resolvent is given by

        .. math::
            R(i \omega_n) = \frac{1}{i \omega_n - E},

        where :math:`E` are the pole energies.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            invert: Whether to apply the inversion in the resolvent formula.

        Returns:
            Resolvent of the grid.
        """
        if kwargs:
            raise TypeError(f"resolvent() got unexpected keyword argument: {next(iter(kwargs))}")
        grid = np.expand_dims(self, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        denominator = 1.0j * grid - energies
        return 1.0 / denominator if invert else denominator

    @classmethod
    def from_uniform(cls, num: int, beta: float | None = None) -> ImaginaryFrequencyGrid:
        """Create a uniform imaginary frequency grid.

        Args:
            num: Number of points in the grid.
            beta: Inverse temperature.

        Returns:
            Uniform imaginary frequency grid.
        """
        if beta is None:
            beta = cls._beta
        if beta is None:
            beta = 256
        separation = 2.0 * np.pi / beta
        start = 0.5 * separation
        stop = (num - 0.5) * separation
        grid = np.linspace(start, stop, num, endpoint=True).view(cls)
        grid.beta = beta
        return grid

    @classmethod
    def from_legendre(
        cls, num: int, diffuse_factor: float = 1.0, beta: float | None = None
    ) -> ImaginaryFrequencyGrid:
        """Create a Legendre imaginary frequency grid.

        Args:
            num: Number of points in the grid.
            diffuse_factor: Diffuse factor for the grid.
            beta: Inverse temperature.

        Returns:
            Legendre imaginary frequency grid.
        """
        if beta is None:
            beta = cls._beta
        if beta is None:
            beta = 256
        points, weights = scipy.special.roots_legendre(num)
        grid = ((1 - points) / (diffuse_factor * (1 + points))).view(cls)
        grid.weights = weights
        grid.beta = beta
        return grid

    def __array_finalize__(self, obj: Array | None, *args: Any, **kwargs: Any) -> None:
        """Finalize the array.

        Args:
            obj: Array to finalize.
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """
        if obj is None:
            return
        super().__array_finalize__(obj, *args, **kwargs)
        self._weights = getattr(obj, "_weights", None)
        self._beta = getattr(obj, "_beta", ImaginaryFrequencyGrid._beta)


GridIF = ImaginaryFrequencyGrid
