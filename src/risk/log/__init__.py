"""
risk/_log
~~~~~~~~~
"""

from .console import log_header, logger, set_global_verbosity
from .parameters import Params

# Initialize the global parameters logger
params = Params()
params.initialize()
