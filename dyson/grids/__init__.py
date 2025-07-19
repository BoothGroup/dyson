r"""Grids for Green's functions and self-energies.

Grids are arrays of points in either the frequency or time domain.


Submodules
----------

.. autosummary::
    :toctree:

    dyson.grids.grid
    dyson.grids.frequency
"""

from dyson.grids.frequency import RealFrequencyGrid, GridRF
from dyson.grids.frequency import ImaginaryFrequencyGrid, GridIF
