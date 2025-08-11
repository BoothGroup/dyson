"""Frequency grids."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import scipy.special

from dyson import numpy as np
from dyson import util
from dyson.grids.grid import BaseGrid
from dyson.representations.enums import Component, Ordering, Reduction

if TYPE_CHECKING:
    from typing import Any

    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array


class BaseFrequencyGrid(BaseGrid):
    """Base class for frequency grids."""

    def evaluate_lehmann(
        self,
        lehmann: Lehmann,
        reduction: Reduction = Reduction.NONE,
        component: Component = Component.FULL,
        **kwargs: Any,
    ) -> Dynamic[BaseFrequencyGrid]:
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
            reduction: The reduction of the dynamic representation.
            component: The component of the dynamic representation.
            kwargs: Additional keyword arguments for the resolvent.

        Returns:
            Lehmann representation, realised on the grid.
        """
        from dyson.representations.dynamic import Dynamic  # noqa: PLC0415

        left, right = lehmann.unpack_couplings()
        resolvent = self.resolvent(lehmann.energies, lehmann.chempot, **kwargs)
        reduction = Reduction(reduction)
        component = Component(component)

        # Get the input and output indices based on the reduction type
        inp = "qk"
        out = "wpq"
        if reduction == reduction.NONE:
            pass
        elif reduction == reduction.DIAG:
            inp = "pk"
            out = "wp"
        elif reduction == reduction.TRACE:
            inp = "pk"
            out = "w"
        else:
            reduction.raise_invalid_representation()

        # Perform the downfolding operation
        array = util.einsum(f"pk,{inp},wk->{out}", right, left.conj(), resolvent)

        # Get the required component
        # TODO: Save time by not evaluating the full array when not needed
        if component == Component.REAL:
            array = array.real
        elif component == Component.IMAG:
            array = array.imag

        return Dynamic(
            self, array, reduction=reduction, component=component, hermitian=lehmann.hermitian
        )

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

    eta: float = 1e-2

    _options = {"eta"}

    def __init__(  # noqa: D417
        self, points: Array, weights: Array | None = None, **kwargs: Any
    ) -> None:
        """Initialise the grid.

        Args:
            points: Points of the grid.
            weights: Weights of the grid.
            eta: Broadening factor.
        """
        self._points = np.asarray(points)
        self._weights = np.asarray(weights) if weights is not None else None
        self.set_options(**kwargs)

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return True

    @staticmethod
    def _resolvent_signs(energies: Array, ordering: Ordering) -> Array:
        """Get the signs for the resolvent based on the time ordering."""
        ordering = Ordering(ordering)
        signs: Array
        if ordering == ordering.ORDERED:
            signs = np.where(energies >= 0, 1.0, -1.0)
        elif ordering == ordering.ADVANCED:
            signs = -np.ones_like(energies)
        elif ordering == ordering.RETARDED:
            signs = np.ones_like(energies)
        else:
            ordering.raise_invalid_representation()
        return signs

    def resolvent(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
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
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
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
        points = np.linspace(start, stop, num, endpoint=True)
        return cls(points, eta=eta)


GridRF = RealFrequencyGrid


class ImaginaryFrequencyGrid(BaseFrequencyGrid):
    """Imaginary frequency grid."""

    beta: float = 256

    _options = {"beta"}

    def __init__(  # noqa: D417
        self, points: Array, weights: Array | None = None, **kwargs: Any
    ) -> None:
        """Initialise the grid.

        Args:
            points: Points of the grid.
            weights: Weights of the grid.
            beta: Inverse temperature.
        """
        self._points = np.asarray(points)
        self._weights = np.asarray(weights) if weights is not None else None
        self.set_options(**kwargs)

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return False

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
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
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
            beta = cls.beta
        separation = 2.0 * np.pi / beta
        start = 0.5 * separation
        stop = (num - 0.5) * separation
        points = np.linspace(start, stop, num, endpoint=True)
        return cls(points, beta=beta)

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
        points, weights = scipy.special.roots_legendre(num)
        points = (1 - points) / (diffuse_factor * (1 + points))
        return cls(points, weights=weights, beta=beta)


GridIF = ImaginaryFrequencyGrid
