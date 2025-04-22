"""Second-order Møller--Plesset perturbation theory (MP2) expressions."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from dyson.expressions.base import BaseExpression

if TYPE_CHECKING:
    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import RHF

    from dyson.typing import Array

einsum = functools.partial(np.einsum, optimize=True)  # TODO: Move


class BaseMP2(BaseExpression):
    """Base class for MP2 expressions."""

    hermitian = True
