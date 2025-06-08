"""
*************************************************************
dyson: Dyson equation solvers for electron propagator methods
*************************************************************

Dyson equation solvers in :mod:`dyson` are general solvers that a variety of inputs to represent
self-energies or existing Green's functions, and solve the Dyson equation in some fashion to
obtain either

    a) a static spectral representation that can be projected into a static representation of the
         Green's function or self-energy, or
    b) a dynamic Green's function.

Below is a table summarising the inputs expected by each solver, first for static solvers:

    +-------------------+--------------------------------------------------------------------------+
    | Solver            | Inputs                                                                   |
    | :---------------- | :----------------------------------------------------------------------- |
    | Exact             | Supermatrix of the static and dynamic self-energy.                       |
    | Davidson          | Matrix-vector operation and diagonal of the supermatrix of the static
                          ad dynamic self-energy.                                                  |
    | Downfolded        | Static self-energy and function returning the dynamic self-energy at a
                          given frequency.                                                         |
    | MBLSE             | Static self-energy and moments of the dynamic self-energy.               |
    | MBLGF             | Moments of the dynamic Green's function.                                 |
    | BlockMBLSE        | Static self-energy and moments of the dynamic self-energies.             |
    | BlockMBLGF        | Moments of the dynamic Green's functions.                                |
    | AufbauPrinciple   | Static self-energy, Lehmann representation of the dynamic self-energy,
                          and the target number of electrons.                                      |
    | AuxiliaryShift    | Static self-energy, Lehmann representation of the dynamic self-energy,
                          and the target number of electrons.                                      |
    | DensityRelaxation | Lehmann representation of the dynamic self-energy, function returning
                          the Fock matrix at a given density, and the target number of electrons.  |
    +-------------------+--------------------------------------------------------------------------+

For dynamic solvers, all solvers require the grid parameters, along with:

    +-------------------+--------------------------------------------------------------------------+
    | Solver            | Inputs                                                                   |
    | :---------------- | :----------------------------------------------------------------------- |
    | CorrectionVector  | Matrix-vector operation and diagonal of the supermatrix of the static
                          and dynamic self-energy.                                                 |
    | CPGF              | Chebyshev polynomial moments of the dynamic Green's function.            |
    +-------------------+--------------------------------------------------------------------------+

For a full accounting of the inputs and their types, please see the documentation for each solver.

"""

__version__ = "0.0.0"

import numpy

from dyson.printing import console, quiet
from dyson.lehmann import Lehmann
from dyson.spectral import Spectral
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
