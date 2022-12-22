"""
*************************************************************
dyson: Dyson equation solvers for electron propagator methods
*************************************************************
"""

__version__ = "1.0.0a"

import logging
import os
import subprocess
import sys

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


HEADER = """      _                       
     | |                      
   __| |_   _ ___  ___  _ __  
  / _` | | | / __|/ _ \| '_ \ 
 | (_| | |_| \__ \ (_) | | | |
  \__,_|\__, |___/\___/|_| |_|
         __/ |                
        |___/                 
%s"""


def init_logging(log):
    """Initialise the logging with a header."""

    if globals().get("_DYSON_LOG_INITIALISED", False):
        return

    # Print header
    header_size = max([len(line) for line in HEADER.split("\n")])
    log.info(HEADER % (" " * (header_size - len(__version__)) + __version__))

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

    log.info("numpy:")
    log.info(" > Version:  %s" % numpy.__version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(numpy.__file__), "..")))

    # Environment variables
    log.info("OMP_NUM_THREADS = %s" % os.environ.get("OMP_NUM_THREADS", ""))

    log.info("")

    globals()["_DYSON_LOG_INITIALISED"] = True
