r"""Representations for Green's functions and self-energies.

Both the Green's function and self-energy can be represented in the frequency domain according to
their Lehmann representation, for the Green's function

.. math::
    G_{pq}(\omega) = \sum_{x} \frac{u_{px} u_{qx}^*}{\omega - \varepsilon_x},

where poles :math:`\varepsilon_x` couple to the physical states of the system according to the
Dyson orbitals :math:`u_{px}`. For the self-energy, the representation is given by

.. math::
    \Sigma_{pq}(\omega) = \sum_{k} \frac{v_{pk} v_{qk}^*}{\omega - \epsilon_k},

where :math:`v_{px}` are the couplings between auxiliary states and the physical states, and
:math:`\epsilon_k` are the auxiliary state energies.

These two Lehmann representations can be relataed to each other via the Dyson equation, which can
be written as an eigenvalue problem in the upfolded configuration space as

.. math::
    \begin{bmatrix} \boldsymbol{\Sigma}(\omega) & \mathbf{v} \\ \mathbf{v}^\dagger &
    \boldsymbol{\epsilon} \mathbf{I} \end{bmatrix} \begin{bmatrix} \mathbf{u} \\ \mathbf{w}
    \end{bmatrix} = \boldsymbol{\varepsilon} \begin{bmatrix} \mathbf{u} \\ \mathbf{w} \end{bmatrix}.

The Lehmann representations of either the Green's function or self-energy are contained in
:class:`~dyson.representations.lehmann.Lehmann` objects, which is a simple container for the
energies and couplings, along with a chemical potential. The
:class:`~dyson.representations.spectral.Spectral` representation provides a container for the full
eigenspectrum (including :math:`\mathbf{w}`), and can provide the Lehmann representation of both the
Green's function and self-energy.

>>> from dyson import util, FCI, Exact
>>> mf = util.get_mean_field("H 0 0 0; H 0 0 1", "6-31g")
>>> fci = FCI.h.from_mf(mf)
>>> solver = Exact.from_expression(fci)
>>> result = solver.kernel()
>>> type(result)
<class 'dyson.representations.spectral.Spectral'>
>>> self_energy = result.get_self_energy()
>>> type(self_energy)
<class 'dyson.representations.lehmann.Lehmann'>
>>> greens_function = result.get_greens_function()
>>> type(greens_function)
<class 'dyson.representations.lehmann.Lehmann'>

Lehmann representations can be realised onto a subclass :class:`~dyson.grids.grid.BaseGrid` to
provide a dynamic representation of the function, which is stored in a
:class:`~dyson.representations.dynamic.Dynamic` object. This dynamic representation has varied
formats, principally depending on the type of grid used, but also according to the so-called
:class:`~dyson.representations.enums.Reduction` and :class:`~dyson.representations.enums.Component`
of the representation. The :class:`~dyson.representations.enums.Reduction` enum encodes the format
of the matrix, i.e. whether it is the full matrix, the diagonal part, or the trace. The
:class:`~dyson.representations.enums.Component` enum encodes the numerical component of the matrix,
i.e. whether it is the real or imaginary part, or the full complex matrix.

>>> from dyson.grids import GridRF
>>> grid = GridRF.from_uniform(-3.0, 3.0, 256, eta=1e-1)
>>> spectrum = grid.evaluate_lehmann(
...     greens_function, ordering="retarded", reduction="trace", component="imag"
... )
>>> type(spectrum)
<class 'dyson.representations.dynamic.Dynamic'>

The various solvers in :mod:`~dyson.solvers` have different representations a their inputs and
outputs.

"""

from dyson.representations.enums import Reduction, Component
from dyson.representations.lehmann import Lehmann
from dyson.representations.spectral import Spectral
from dyson.representations.dynamic import Dynamic
