"""
risk/cluster/api
~~~~~~~~~~~~~~~~
"""

import copy
import random

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from ..log import log_header, logger, params
from .cluster import cluster_method
from ._community import (
    calculate_greedy_modularity_clusters,
    calculate_label_propagation_clusters,
    calculate_leiden_clusters,
    calculate_louvain_clusters,
    calculate_markov_clustering_clusters,
    calculate_spinglass_clusters,
    calculate_walktrap_clusters,
)


class ClusterAPI:
    """
    Handles the loading of statistical results and annotation significance for clusters.

    The ClusterAPI class provides methods for explicit clustering algorithms.
    """

    @cluster_method
    def cluster_greedy(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
    ) -> csr_matrix:
        """
        Compute greedy modularity clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="greedy",
            fraction_shortest_edges=fraction_shortest_edges,
        )
        network = copy.copy(network)
        return calculate_greedy_modularity_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
        )

    @cluster_method
    def cluster_labelprop(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
    ) -> csr_matrix:
        """
        Compute label propagation clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="labelprop",
            fraction_shortest_edges=fraction_shortest_edges,
        )
        network = copy.copy(network)
        return calculate_label_propagation_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
        )

    @cluster_method
    def cluster_leiden(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
        resolution: float = 1.0,
        random_seed: int = 888,
    ) -> csr_matrix:
        """
        Compute Leiden clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.
            resolution (float, optional): Resolution parameter for Leiden algorithm. Defaults to 1.0.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="leiden",
            fraction_shortest_edges=fraction_shortest_edges,
            resolution=resolution,
            random_seed=random_seed,
        )
        # Additional logging for specific parameters
        logger.debug(f"Resolution: {resolution}")
        logger.debug(f"Random seed: {random_seed}")
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        network = copy.copy(network)
        return calculate_leiden_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
            resolution=resolution,
            random_seed=random_seed,
        )

    @cluster_method
    def cluster_louvain(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
        resolution: float = 0.1,
        random_seed: int = 888,
    ) -> csr_matrix:
        """
        Compute Louvain clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.
            resolution (float, optional): Resolution parameter for Louvain algorithm. Defaults to 1.0.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="louvain",
            fraction_shortest_edges=fraction_shortest_edges,
            resolution=resolution,
            random_seed=random_seed,
        )
        # Additional logging for specific parameters
        logger.debug(f"Resolution: {resolution}")
        logger.debug(f"Random seed: {random_seed}")
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        network = copy.copy(network)
        return calculate_louvain_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
            resolution=resolution,
            random_seed=random_seed,
        )

    @cluster_method
    def cluster_markov(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
    ) -> csr_matrix:
        """
        Compute Markov clustering clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="markov",
            fraction_shortest_edges=fraction_shortest_edges,
        )
        network = copy.copy(network)
        return calculate_markov_clustering_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
        )

    @cluster_method
    def cluster_spinglass(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
    ) -> csr_matrix:
        """
        Compute spinglass clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="spinglass",
            fraction_shortest_edges=fraction_shortest_edges,
        )
        network = copy.copy(network)
        return calculate_spinglass_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
        )

    @cluster_method
    def cluster_walktrap(
        self,
        network: nx.Graph,
        fraction_shortest_edges: float = 0.5,
    ) -> csr_matrix:
        """
        Compute walktrap clusters for the given network.

        Args:
            network (nx.Graph): The network graph to cluster.
            fraction_shortest_edges (float, optional): Rank-based fraction (0, 1] of the shortest edges
                retained when building the clustering subgraph. Defaults to 0.5.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        self._log_clustering_params(
            clustering="walktrap",
            fraction_shortest_edges=fraction_shortest_edges,
        )
        network = copy.copy(network)
        return calculate_walktrap_clusters(
            network,
            fraction_shortest_edges=fraction_shortest_edges,
        )

    def _log_clustering_params(
        self,
        clustering: str,
        fraction_shortest_edges: float,
        **kwargs,
    ) -> None:
        """
        Log clustering parameters for debugging and reproducibility.

        Args:
            clustering (str): The display name of the clustering method.
            fraction_shortest_edges (float): Rank-based fraction (0, 1] of the shortest edges used for clustering.
            **kwargs: Additional clustering parameters to log.
        """
        log_header("Computing clusters")
        # Log and display cluster settings
        logger.debug(f"Clustering: '{clustering}'")
        logger.debug(f"Edge length threshold: {fraction_shortest_edges}")
        # Log clustering parameters
        params.log_clusters(
            clustering=clustering,
            fraction_shortest_edges=fraction_shortest_edges,
            **kwargs,
        )
