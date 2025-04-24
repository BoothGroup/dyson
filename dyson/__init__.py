"""
*************************************************************
dyson: Dyson equation solvers for electron propagator methods
*************************************************************

Dyson equation solvers in :mod:`dyson` are general solvers that a variety of inputs to represent
self-energies or existing Green's functions, and solve the Dyson equation in some fashion to
obtain either

    a) a static Lehmann representation of the self-energy and Green's function, or
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
    | SelfConsistent    | Function returning the Lehmann representation of the dynamic self-energy
                          for a given Lehmann representation of the dynamic Green's function,
                          function returning the Fock matrix at a given density, and the target
                          number of electrons.                                                     |
    +-------------------+--------------------------------------------------------------------------+

For dynamic solvers, all solvers require the grid parameters, along with:

    +-------------------+--------------------------------------------------------------------------+
    | Solver            | Inputs                                                                   |
    | :---------------- | :----------------------------------------------------------------------- |
    | CorrectionVector  | Matrix-vector operation and diagonal of the supermatrix of the static
                          and dynamic self-energy.                                                 |
    | CPGF              | Chebyshev polynomial moments of the dynamic Green's function.            |
    | KPMGF             | Chebyshev polynomial moments of the dynamic Green's function.            |
    +-------------------+--------------------------------------------------------------------------+

For a full accounting of the inputs and their types, please see the documentation for each solver.
"""

__version__ = "0.0.0"

import logging
import os
import subprocess
import sys


# --- NumPy backend:

import numpy


# --- Logging:


def output(self, msg, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, msg, args, **kwargs)


default_log = logging.getLogger(__name__)
default_log.setLevel(logging.INFO)
default_log.addHandler(logging.StreamHandler(sys.stderr))
logging.addLevelName(25, "OUTPUT")
logging.Logger.output = output


class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__("null")

    def _log(self, level, msg, args, **kwargs):
        pass


HEADER = """     _                           
    | |                          
  __| | _   _  ___   ___   _ __  
 / _` || | | |/ __| / _ \ | '_ \ 
| (_| || |_| |\__ \| (_) || | | |
 \__,_| \__, ||___/ \___/ |_| |_|
         __/ |                   
        |___/  %s
"""


def init_logging(log):
    """Initialise the logging with a header."""

    if globals().get("_DYSON_LOG_INITIALISED", False):
        return

    # Print header
    header_size = max([len(line) for line in HEADER.split("\n")])
    log.info(HEADER % (" " * (18 - len(__version__)) + __version__))

    # Print versions of dependencies and ebcc
    def get_git_hash(directory):
        git_directory = os.path.join(directory, ".git")
        cmd = ["git", "--git-dir=%s" % git_directory, "rev-parse", "--short", "HEAD"]
        try:
            git_hash = subprocess.check_output(
                cmd, universal_newlines=True, stderr=subprocess.STDOUT
            ).rstrip()
        except subprocess.CalledProcessError:
            git_hash = "N/A"
        return git_hash

    import numpy
    import pyscf

    log.info("numpy:")
    log.info(" > Version:  %s" % numpy.__version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(numpy.__file__), "..")))

    log.info("pyscf:")
    log.info(" > Version:  %s" % pyscf.__version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(pyscf.__file__), "..")))

    log.info("dyson:")
    log.info(" > Version:  %s" % __version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(__file__), "..")))

    # Environment variables
    log.info("OMP_NUM_THREADS = %s" % os.environ.get("OMP_NUM_THREADS", ""))

    log.info("")

    globals()["_DYSON_LOG_INITIALISED"] = True


# -- Other imports:

from dyson.util import *
from dyson.lehmann import Lehmann
from dyson.solvers import *
from dyson.expressions import *
