"""
risk/cluster/api
~~~~~~~~~~~~~~~~
"""

import copy
from typing import List, Tuple, Union

import networkx as nx
import numpy as np

from ..log import log_header, logger, params
from .cluster import get_network_clusters


class ClusterAPI:
    """
    Handles the loading of statistical results and annotation significance for clusters.

    The ClusterAPI class provides methods to load cluster results from statistical tests.
    """

    def load_clusters(
        self,
        network: nx.Graph,
        clustering: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        random_seed: int = 888,
        **kwargs,
    ):
        """
        Load clusters for the network.

        Args:
            network (nx.Graph): The input network graph.
            clustering (Union[str, List, Tuple, np.ndarray], optional): The clustering method(s) to define clusters.
                Can be a single method (e.g., 'louvain', 'leiden') or a collection of methods. Defaults to "louvain".
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (Union[float, List, Tuple, np.ndarray], optional): Fraction of shortest edges to consider for creating subgraphs.
                Can be a single value or a collection of thresholds for flexibility. Defaults to 0.5.
            random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.

        Returns:
            csr_matrix: The cluster matrix.
        """
        log_header("Loading clusters")
        # Log and display cluster settings
        clustering_log = {
            "louvain": f"louvain (resolution={louvain_resolution})",
            "leiden": f"leiden (resolution={leiden_resolution})",
        }
        logger.debug(f"Clustering: '{clustering_log.get(clustering, clustering)}'")
        logger.debug(f"Edge length threshold: {fraction_shortest_edges}")
        logger.debug(f"Random seed: {random_seed}")
        # Log clustering parameters
        params.log_clusters(
            clustering=clustering,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            random_seed=random_seed,
            **kwargs,
        )

        # Make a copy of the network to avoid modifying the original
        network = copy.copy(network)
        # Compute clusters
        clusters = get_network_clusters(
            network,
            clustering,
            fraction_shortest_edges,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            random_seed=random_seed,
        )

        # Return the sparse cluster matrix
        return clusters
