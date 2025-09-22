r"""Solvers for solving the Dyson equation.

Solvers generally provide some method to solving the Dyson equation

.. math::
    \mathbf{G}(\omega) = \left( \left[ \mathbf{G}^0(\omega) \right]^{-1} -
    \boldsymbol{\Sigma}(\omega) \right)^{-1},

which can also be written recursively as

.. math::
    \mathbf{G}(\omega) = \mathbf{G}^0(\omega) + \mathbf{G}^0(\omega) \boldsymbol{\Sigma}(\omega)
    \mathbf{G}(\omega),

and can be expressed as an eigenvalue problem as

.. math::
    \begin{bmatrix} \boldsymbol{\Sigma}(\omega) & \mathbf{v} \\ \mathbf{v}^\dagger & \mathbf{K} +
    \mathbf{C} \end{bmatrix} \mathbf{u} = \omega \mathbf{u}.

For more details on the equivalence of these representations, see the :mod:`~dyson.representations`
module.

The :class:`~dyson.solvers.solver.BaseSolver` interface provides the constructors
:func:`~dyson.solvers.solver.BaseSolver.from_expression` and
:func:`~dyson.solvers.solver.BaseSolver.from_self_energy` to create a solver of that type from
either an instance of a subclass of :class:`~dyson.expressions.expression.BaseExpression` or a
self-energy in the form of an instance of :class:`~dyson.representations.lehmann.Lehmann` object,
respectively

>>> from dyson import util, quiet, CCSD, Exact
>>> quiet()  # Suppress output
>>> mf = util.get_mean_field("H 0 0 0; H 0 0 1", "6-31g")
>>> ccsd = CCSD.h.from_mf(mf)
>>> solver = Exact.from_expression(ccsd)

Solvers can be run by calling the :meth:`~dyson.solvers.solver.BaseSolver.kernel` method, which
in the case of :mod:`~dyson.solvers.static` solvers sets the attribute and returns :attr:`result`,
an instance of :class:`~dyson.representations.spectral.Spectral`

>>> result = solver.kernel()
>>> type(result)
<class 'dyson.representations.spectral.Spectral'>

The result can then be used to construct Lehmann representations of the Green's function and
self-energy, details of which can be found in the :mod:`~dyson.representations` module. On the other
hand, solvers in :mod:`~dyson.solvers.dynamic` return an instance of
:class:`~dyson.representations.dynamic.Dynamic`, which contains the dynamic Green's function in the
format requested by the solver arguments.

A list of available solvers is provided in the documentation of :mod:`dyson`, along with their
expected inputs.


Submodules
----------

.. autosummary::
    :toctree:

    solver
    static
    dynamic

"""

from dyson.solvers.static.exact import Exact
from dyson.solvers.static.davidson import Davidson
from dyson.solvers.static.downfolded import Downfolded
from dyson.solvers.static.mblse import MBLSE
from dyson.solvers.static.mblgf import MBLGF
from dyson.solvers.static.chempot import AufbauPrinciple, AuxiliaryShift
from dyson.solvers.static.density import DensityRelaxation
from dyson.solvers.dynamic.corrvec import CorrectionVector
from dyson.solvers.dynamic.cpgf import CPGF
