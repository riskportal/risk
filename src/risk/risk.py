"""
risk/risk
~~~~~~~~~
"""

from .annotation import AnnotationAPI
from .cluster import ClusterAPI
from .log import params, set_global_verbosity
from .network import GraphAPI, NetworkAPI, PlotterAPI
from .stats import StatsAPI


class RISK(NetworkAPI, AnnotationAPI, ClusterAPI, StatsAPI, GraphAPI, PlotterAPI):
    """
    RISK: a framework for scalable network annotation and visualization.

    Combines clustering, statistical enrichment, and visualization methods into a unified API
    for analyzing biological and interdisciplinary networks. Supports multiple input formats,
    community detection algorithms, and statistical tests for functional annotation.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the high-level API and propagate verbosity to all submodules.

        Args:
            verbose (bool): Whether to emit debug-level log messages. Defaults to True.
        """
        # Set global verbosity for logging
        set_global_verbosity(verbose)
        # Provide public access to network parameters
        self.params = params
        super().__init__()
