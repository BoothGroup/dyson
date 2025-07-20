r"""Expressions for constructing Green's functions and self-energies.

Subclasses of :class:`~dyson.expressions.expression.BaseExpression` expose various methods which
provide different representations of the self-energy or Green's function for the given level of
theory. The Green's function is related to the resolvent

.. math::
    left[ \omega - \mathbf{H} \right]^{-1}

where :math:`\mathbf{H}` is the Hamiltonian, and in the presence of correlation, takes the form of a
self-energy supermatrix

.. math::
    \mathbf{H} = \begin{bmatrix} \boldsymbol{\Sigma}(\omega) & \mathbf{v} \\ \mathbf{v}^\dagger &
    \mathbf{K} + \mathbf{C} \end{bmatrix}

which possesses its own Lehmann representation. For more details on these representations, see the
:module:`~dyson.representations` module.

The :class:`~dyson.expressions.expression.BaseExpression` interface provides a
:func:`~dyson.expressions.expression.BaseExpression.from_mf` constructor to create an expression of
that level of theory from a mean-field object

>>> from dyson import util, CCSD
>>> mf = util.get_mean_field("H 0 0 0; H 0 0 1", "6-31g")
>>> ccsd = CCSD.h.from_mf(mf)

The :class:`~dyson.expressions.expression.BaseExpression` interface provides methods to compute the
matrix-vector operations and diagonal of the self-energy supermatrix

>>> import numpy as np
>>> ham = ccsd.build_matrix()
>>> np.allclose(np.diag(ham), ccsd.diagonal())
True
>>> vec = np.random.random(ccsd.shape[0])
>>> np.allclose(ccsd.apply_hamiltonian_right(vec), ham @ vec)
True
>>> np.allclose(ccsd.apply_hamiltonian_left(vec), vec @ ham)
True

More precisely, the Green's function requires also the excitation operators to connect to the
ground state

.. math::
    \mathbf{G}(\omega) = \left\langle \boldsymbol{\Psi}_0 \right| \hat{a}_p \left[ \omega -
    \mathbf{H} \right]^{-1} \hat{a}_q^\dagger \left| \boldsymbol{\Psi}_0 \right\rangle,

which may be a simple projection when the ground state is mean-field, or otherwise
in the case of correlated ground states. The interface can provide these vectors

>>> bra = ccsd.get_excitation_bras()
>>> ket = ccsd.get_excitation_kets()

which are vectors with shape `(nphys, nconfig)` where `nphys` is the number of physical states.

These methods can be used to construct the moments of the Green's function

.. math::
    \mathbf{G}_n = \left\langle \boldsymbol{\Psi}_0 \right| \hat{a}_p \mathbf{H}^n
    \hat{a}_q^\dagger \left| \boldsymbol{\Psi}_0 \right\rangle,

which are important for some of the novel approaches implemented in :mod:`dyson`. In the case of
some levels of theory, analytic expressions for the moments of the self-energy are also available.
These moments can be calculated using

>>> gf_moments = ccsd.build_gf_moments(nmom=10)

A list of available expressions is provided in the documentation of :mod:`dyson`. Each expression
is an instance of :class:`~dyson.expressions.expression.ExpressionCollection`, which provides the
subclasses of :class:`~dyson.expressions.expression.BaseExpression` for various sectors such as the
hole and particle.


Submodules
----------

.. autosummary::
    :toctree:

    expression
    hf
    ccsd
    fci
    adc
    gw

"""

from dyson.expressions.hf import HF
from dyson.expressions.ccsd import CCSD
from dyson.expressions.fci import FCI
from dyson.expressions.adc import ADC2, ADC2x
from dyson.expressions.gw import TDAGW
