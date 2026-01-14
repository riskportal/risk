"""
risk/cluster/cluster
~~~~~~~~~~~~~~~~~~~~
"""

import random
import warnings
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import DataConversionWarning

from ..log import logger
from ._community import (
    calculate_greedy_modularity_clusters,
    calculate_label_propagation_clusters,
    calculate_leiden_clusters,
    calculate_louvain_clusters,
    calculate_markov_clustering_clusters,
    calculate_spinglass_clusters,
    calculate_walktrap_clusters,
)

# Suppress DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def cluster_method(func):
    """
    Decorator for clustering functions to ensure deterministic, reproducible results.
    Sets random seeds, copies the network, and ensures output is normalized.

    Args:
        func (callable): The clustering function to be decorated.

    Returns:
        callable: The wrapped clustering function with added functionality.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function to set random seeds and normalize output.

        Args:
            *args: Positional arguments for the clustering function.
            **kwargs: Keyword arguments for the clustering function.

        Returns:
            csr_matrix: Sparse matrix representing cluster assignments.
        """
        clusters = func(*args, **kwargs)
        return _set_max_row_value_to_one_sparse(clusters)

    return wrapper


def get_network_clusters(
    network: nx.Graph,
    clustering: str = "louvain",
    fraction_shortest_edges: float = 0.5,
    louvain_resolution: float = 0.1,
    leiden_resolution: float = 1.0,
    random_seed: int = 888,
) -> csr_matrix:
    """
    Calculate clusters for the network using a single method.

    Args:
        network (nx.Graph): The network graph.
        clustering (str, optional): The clustering method ('greedy', 'labelprop', 'leiden', 'louvain', 'markov', 'spinglass', 'walktrap').
        fraction_shortest_edges (float, optional): Fraction of shortest edges to consider for creating subgraphs. Defaults to 0.5.
        louvain_resolution (float, optional): Resolution for Louvain.
        leiden_resolution (float, optional): Resolution for Leiden.
        random_seed (int, optional): Random seed.

    Returns:
        csr_matrix: Sparse cluster matrix.

    Raises:
        ValueError: If invalid clustering method is provided.
    """
    # Set random seed for cluster reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    clusters = None
    # Determine clustering method and compute clusters
    if clustering == "greedy":
        clusters = calculate_greedy_modularity_clusters(
            network, fraction_shortest_edges=fraction_shortest_edges
        )
    elif clustering == "labelprop":
        clusters = calculate_label_propagation_clusters(
            network, fraction_shortest_edges=fraction_shortest_edges
        )
    elif clustering == "leiden":
        clusters = calculate_leiden_clusters(
            network,
            resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            random_seed=random_seed,
        )
    elif clustering == "louvain":
        clusters = calculate_louvain_clusters(
            network,
            resolution=louvain_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            random_seed=random_seed,
        )
    elif clustering == "markov":
        clusters = calculate_markov_clustering_clusters(
            network, fraction_shortest_edges=fraction_shortest_edges
        )
    elif clustering == "spinglass":
        clusters = calculate_spinglass_clusters(
            network, fraction_shortest_edges=fraction_shortest_edges
        )
    elif clustering == "walktrap":
        clusters = calculate_walktrap_clusters(
            network, fraction_shortest_edges=fraction_shortest_edges
        )
    else:
        raise ValueError(
            "Invalid clustering method. Choose from: 'greedy', 'labelprop', 'leiden', 'louvain', 'markov', 'spinglass', 'walktrap'."
        )

    # Ensure maximum per row set to 1
    clusters = _set_max_row_value_to_one_sparse(clusters)

    return clusters


def _set_max_row_value_to_one_sparse(matrix: csr_matrix) -> csr_matrix:
    """
    Set the maximum value in each row of a sparse matrix to 1.

    Args:
        matrix (csr_matrix): The input sparse matrix.

    Returns:
        csr_matrix: The modified sparse matrix where only the maximum value in each row is set to 1.
    """
    # Iterate over each row and set the maximum value to 1
    for i in range(matrix.shape[0]):
        row_data = matrix[i].data
        if len(row_data) > 0:
            row_data[:] = (row_data == max(row_data)).astype(int)

    return matrix


def process_significant_clusters(
    network: nx.Graph,
    significant_clusters: Dict[str, Any],
    display_prune_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Process clusters based on the display pruning settings.

    Args:
        network (nx.Graph): The network data structure used for pruning neighbors.
        significant_clusters (Dict[str, Any]): Dictionary containing 'significance_matrix', 'significant_binary_significance_matrix', and 'significant_significance_matrix'.
        display_prune_threshold (float, optional): Display-only pruning based on spatial layout
            distance that suppresses spatially diffuse or isolated regions in plots. Runs after
            enrichment and clustering on plotting matrices only, does not use enrichment strength
            or statistical significance or affect clustering, enrichment, or statistical testing,
            and defaults to 0.0.

    Returns:
        Dict[str, Any]: Processed clusters data, including the updated matrices and significance counts.
    """
    significance_matrix = significant_clusters["significance_matrix"]
    significant_binary_significance_matrix = significant_clusters[
        "significant_binary_significance_matrix"
    ]
    significant_significance_matrix = significant_clusters["significant_significance_matrix"]

    logger.debug(f"Display prune threshold: {display_prune_threshold}")
    if display_prune_threshold:
        (
            significance_matrix,
            significant_binary_significance_matrix,
            significant_significance_matrix,
        ) = _prune_neighbors(
            network,
            significance_matrix,
            significant_binary_significance_matrix,
            distance_threshold=display_prune_threshold,
        )

    cluster_significance_counts = np.sum(significant_binary_significance_matrix, axis=0)
    node_significance_sums = np.sum(significance_matrix, axis=1)
    return {
        "significance_matrix": significance_matrix,
        "significant_binary_significance_matrix": significant_binary_significance_matrix,
        "significant_significance_matrix": significant_significance_matrix,
        "cluster_significance_counts": cluster_significance_counts,
        "node_significance_sums": node_significance_sums,
    }


def _prune_neighbors(
    network: nx.Graph,
    significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    distance_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove outliers based on their rank for edge lengths.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        significance_matrix (np.ndarray): The significance matrix.
        significant_binary_significance_matrix (np.ndarray): The alpha threshold matrix.
        distance_threshold (float): Rank threshold (0 to 1) to determine outliers.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - np.ndarray: The updated significance matrix with outliers set to zero.
            - np.ndarray: The updated alpha threshold matrix with outliers set to zero.
            - np.ndarray: The significant significance matrix, where non-significant entries are set to zero.
    """
    # Identify indices with non-zero rows in the binary significance matrix
    non_zero_indices = np.where(significant_binary_significance_matrix.sum(axis=1) != 0)[0]
    median_distances = []
    distance_lookup = {}
    isolated_nodes = []  # Track nodes with no significant neighbors
    for node in non_zero_indices:
        dist = _median_distance_to_significant_neighbors(
            node, network, significant_binary_significance_matrix
        )
        if dist is not None:
            median_distances.append(dist)
            distance_lookup[node] = dist
        else:
            isolated_nodes.append(node)  # Node has no significant neighbors

    if not median_distances:
        logger.warning("No significant neighbors found for pruning.")
        significant_significance_matrix = np.where(
            significant_binary_significance_matrix == 1, significance_matrix, 0
        )
        return (
            significance_matrix,
            significant_binary_significance_matrix,
            significant_significance_matrix,
        )

    # Calculate the distance threshold value based on rank
    distance_threshold_value = _calculate_threshold(median_distances, 1 - distance_threshold)
    # Prune nodes that are outliers based on the distance threshold
    for node, dist in distance_lookup.items():
        if dist >= distance_threshold_value:
            significance_matrix[node] = 0
            significant_binary_significance_matrix[node] = 0

    # Prune isolated nodes (no significant neighbors)
    for node in isolated_nodes:
        significance_matrix[node] = 0
        significant_binary_significance_matrix[node] = 0

    # Create a matrix where non-significant entries are set to zero
    significant_significance_matrix = np.where(
        significant_binary_significance_matrix == 1, significance_matrix, 0
    )

    return (
        significance_matrix,
        significant_binary_significance_matrix,
        significant_significance_matrix,
    )


def _median_distance_to_significant_neighbors(
    node, network, significance_mask
) -> Union[float, Any, None]:
    """
    Calculate the median distance from a node to its significant neighbors.

    Args:
        node (Any): The node for which the median distance is being calculated.
        network (nx.Graph): The network graph containing the nodes.
        significance_mask (np.ndarray): Binary matrix indicating significant nodes.

    Returns:
        Union[float, None]: The median distance to significant neighbors, or None if no significant neighbors exist.
    """
    # Get all neighbors at once
    neighbors = list(network.neighbors(node))
    if not neighbors:
        return None

    # Vectorized check for significant neighbors
    neighbors = np.array(neighbors)
    significant_mask = significance_mask[neighbors].sum(axis=1) != 0
    significant_neighbors = neighbors[significant_mask]
    if len(significant_neighbors) == 0:
        return None

    # Vectorized distance calculation
    node_pos = _get_node_position(network, node)
    neighbor_positions = np.array([_get_node_position(network, n) for n in significant_neighbors])
    distances = np.linalg.norm(neighbor_positions - node_pos, axis=1)

    return np.median(distances)


def _get_euclidean_distance(node1: Any, node2: Any, network: nx.Graph) -> float:
    """
    Calculate the Euclidean distance between two nodes in the network.

    Args:
        node1 (Any): The first node.
        node2 (Any): The second node.
        network (nx.Graph): The network graph containing the nodes.

    Returns:
        float: The Euclidean distance between the two nodes.
    """
    pos1 = _get_node_position(network, node1)
    pos2 = _get_node_position(network, node2)
    return np.linalg.norm(pos1 - pos2)


def _get_node_position(network: nx.Graph, node: Any) -> np.ndarray:
    """
    Retrieve the position of a node in the network as a numpy array.

    Args:
        network (nx.Graph): The network graph containing node positions.
        node (Any): The node for which the position is being retrieved.

    Returns:
        np.ndarray: A numpy array representing the position of the node in the format [x, y, z].
    """
    return np.array(
        [
            network.nodes[node].get(coord, 0)
            for coord in ["x", "y", "z"]
            if coord in network.nodes[node]
        ]
    )


def _calculate_threshold(median_distances: List, distance_threshold: float) -> float:
    """
    Calculate the distance threshold based on the given median distances and a percentile threshold.

    Args:
        median_distances (List): An array of median distances.
        distance_threshold (float): A percentile threshold (0 to 1) used to determine the distance cutoff.

    Returns:
        float: The calculated distance threshold value.

    Raises:
        ValueError: If no significant annotation is found in the median distances.
    """
    # Sort the median distances
    sorted_distances = np.sort(median_distances)
    # Compute the rank fractions for the sorted distances
    rank_percentiles = np.linspace(0, 1, len(sorted_distances))
    # Interpolating the ranks to 1000 evenly spaced percentiles
    interpolated_percentiles = np.linspace(0, 1, 1000)
    try:
        smoothed_distances = np.interp(interpolated_percentiles, rank_percentiles, sorted_distances)
    except ValueError as e:
        raise ValueError("No significant annotation found.") from e

    # Determine the index corresponding to the distance threshold
    threshold_index = int(np.ceil(distance_threshold * len(smoothed_distances))) - 1
    # Return the smoothed distance at the calculated index
    return smoothed_distances[threshold_index]
