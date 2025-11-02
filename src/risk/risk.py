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
    """High-level API for network loading, annotation, clustering, statistics, and plotting."""

    def __init__(self, verbose: bool = True):
        """
        Create a high-level API instance and propagate verbosity to affiliated modules.

        Args:
            verbose (bool): Whether to emit debug-level log messages. Defaults to True.

        Notes:
            The instance exposes the shared `params` object so downstream analysis can
            inspect the configuration captured during each load or run call.
        """
        # Set global verbosity for logging
        set_global_verbosity(verbose)
        # Provide public access to network parameters
        self.params = params
        super().__init__()
