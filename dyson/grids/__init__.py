r"""Grids for Green's functions and self-energies.

Grids are arrays of points in either the frequency or time domain.


Submodules
----------

.. autosummary::
    :toctree:

    grid
    frequency
"""

from dyson.grids.frequency import RealFrequencyGrid, GridRF
from dyson.grids.frequency import ImaginaryFrequencyGrid, GridIF
from dyson.grids.time import RealTimeGrid, GridRT
