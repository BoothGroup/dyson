"""
**********************************************************
dyson: Dyson equation solvers for Green's function methods
**********************************************************

Dyson equation solvers in :mod:`dyson` are general solvers that accept a variety of inputs to
represent self-energies or existing Green's functions, and solve the Dyson equation in some fashion
to obtain either

a) a static spectral representation that can be projected into a static Lehmann representation
   of the Green's function or self-energy, or
b) a dynamic Green's function.

The self-energy and Green's function are represented in the following ways:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Representation
     - Description
   * - :class:`~dyson.representations.spectral.Spectral`
     - Eigenvalues and eigenvectors of the static self-energy supermatrix, from which the
       Lehmann representation of the self-energy or Green's function can be constructed.
   * - :class:`~dyson.representations.lehmann.Lehmann`
     - The Lehmann representation of the self-energy or Green's function, consisting of pole
       energies and their couplings to a physical space.
   * - :class:`~dyson.representations.dynamic.Dynamic`
     - The dynamic self-energy or Green's function, represented as a series of arrays at each
       point on a grid of time or frequency points.

The available static solvers are, along with their expected inputs:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Solver
     - Inputs
   * - :class:`~dyson.solvers.static.exact.Exact`
     - Supermatrix of the static and dynamic self-energy.
   * - :class:`~dyson.solvers.static.davidson.Davidson`
     - Matrix-vector operation and diagonal of the supermatrix of the static and dynamic
       self-energy.
   * - :class:`~dyson.solvers.static.downfolded.Downfolded`
     - Static self-energy and function returning the dynamic self-energy at a given frequency.
   * - :class:`~dyson.solvers.static.mblse.MBLSE`
     - Static self-energy and moments of the dynamic self-energy.
   * - :class:`~dyson.solvers.static.mblgf.MBLGF`
     - Moments of the dynamic Green's function.
   * - :class:`~dyson.solvers.static.chempot.AufbauPrinciple`
     - Static self-energy, Lehmann representation of the dynamic self-energy, and the target
       number of electrons.
   * - :class:`~dyson.solvers.static.chempot.AuxiliaryShift`
     - Static self-energy, Lehmann representation of the dynamic self-energy, and the target
       number of electrons.
   * - :class:`~dyson.solvers.static.density.DensityRelaxation`
     - Lehmann representation of the dynamic self-energy, function returning the Fock matrix at a
       given density, and the target number of electrons.

For dynamic solvers, all solvers require the grid parameters, along with:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Solver
     - Inputs
   * - :class:`~dyson.solvers.dynamic.corrvec.CorrectionVector`
     - Matrix-vector operation and diagonal of the supermatrix of the static and dynamic
       self-energy.
   * - :class:`~dyson.solvers.dynamic.cpgf.CPGF`
     - Chebyshev polynomial moments of the dynamic Green's function.

For a full accounting of the inputs and their types, please see the documentation for each solver.

A number of classes are provided to represent the expressions needed to construct these inputs at
different levels of theory. These expressions are all implemented for RHF references, with other
spin symmetries left to the user to implement as needed. The available expressions are:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Expression
      - Description
    * - :data:`~dyson.expressions.hf.HF`
      - Hartree--Fock (mean-field) ground state, exploiting Koopmans' theorem for the excited
        states.
    * - :data:`~dyson.expressions.ccsd.CCSD`
      - Coupled cluster singles and doubles ground state, and the respective equation-of-motion
        method for the excited states.
    * - :data:`~dyson.expressions.fci.FCI`
      - Full configuration interaction (exact diagonalisation) ground and excited states.
    * - :data:`~dyson.expressions.adc.ADC2`
      - Algebraic diagrammatic construction second order excited states, based on a mean-field
        ground state.
    * - :data:`~dyson.expressions.adc.ADC2x`
      - Algebraic diagrammatic construction extended second order excited states, based on a
        mean-field ground state.
    * - :data:`~dyson.expressions.gw.TDAGW`
      - GW theory with the Tamm--Dancoff approximation for the excited states, based on a
        mean-field ground state.


Submodules
----------

.. autosummary::
    :toctree: _autosummary

    dyson.expressions
    dyson.grids
    dyson.representations
    dyson.solvers
    dyson.util

"""

__version__ = "1.0.0"

import numpy
import scipy

from dyson.printing import console, quiet
from dyson.representations import Lehmann, Spectral, Dynamic
from dyson.solvers import (
    Exact,
    Davidson,
    Downfolded,
    MBLSE,
    MBLGF,
    AufbauPrinciple,
    AuxiliaryShift,
    DensityRelaxation,
    CorrectionVector,
    CPGF,
)
from dyson.expressions import HF, CCSD, FCI, ADC2, ADC2x, TDAGW
