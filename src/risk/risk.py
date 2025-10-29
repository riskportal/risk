"""
risk/risk
~~~~~~~~~
"""

from .annotation import AnnotationHandler
from .cluster import ClusterAPI
from .log import params, set_global_verbosity
from .network import GraphAPI, NetworkAPI, PlotterAPI
from .stats import StatsAPI


class RISK(NetworkAPI, AnnotationHandler, ClusterAPI, StatsAPI, GraphAPI, PlotterAPI):
    """
    RISK: A class for network analysis and visualization.

    The RISK class integrates functionalities for loading networks, processing annotations,
    performing network-based statistical analysis to quantify cluster relationships,
    and visualizing networks and their properties.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the RISK class with configuration settings.

        Args:
            verbose (bool): If False, suppresses all log messages to the console. Defaults to True.
        """
        # Set global verbosity for logging
        set_global_verbosity(verbose)
        # Provide public access to network parameters
        self.params = params
        super().__init__()
