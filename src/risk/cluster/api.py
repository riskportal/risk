"""
risk/cluster/api
~~~~~~~~~~~~~~~~
"""

import copy
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

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
        clustering: str = "louvain",
        fraction_shortest_edges: float = 0.5,
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        random_seed: int = 888,
        **kwargs,
    ) -> csr_matrix:
        """
        Load clusters for the network.

        Args:
            network (nx.Graph): The input network graph.
            clustering (str, optional): Clustering method to use ('greedy', 'labelprop', 'leiden', 'louvain', 'markov', 'spinglass', 'walktrap').
                Defaults to "louvain".
            fraction_shortest_edges (float, optional): Fraction of shortest edges to consider for creating subgraphs. Defaults to 0.5.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.

        Returns:
            csr_matrix: The cluster matrix.
        """
        log_header("Computing clusters")
        # Log and display cluster settings
        clustering_log = {
            "louvain": f"louvain (resolution={louvain_resolution})",
            "leiden": f"leiden (resolution={leiden_resolution})",
        }
        clustering_display = clustering_log.get(clustering, clustering)
        logger.debug(f"Clustering: '{clustering_display}'")
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
