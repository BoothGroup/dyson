"""Second-order Møller--Plesset perturbation theory (MP2) expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson.expressions.expression import BaseExpression

if TYPE_CHECKING:
    pass


class BaseMP2(BaseExpression):
    """Base class for MP2 expressions."""

    hermitian = True
