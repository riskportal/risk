"""
risk/stats/_stats/tests
~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import binom, chi2, hypergeom


def compute_binom_test(
    clusters: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute Binomial test for enrichment and depletion in clusters with selectable null distribution.

    Args:
        clusters (csr_matrix): Sparse binary matrix representing clusters.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Get the total number of nodes in the network
    total_nodes = clusters.shape[1]

    # Compute sums (remain sparse here)
    cluster_sizes = clusters.sum(axis=1)  # Row sums
    annotation_totals = annotation.sum(axis=0)  # Column sums
    # Compute probabilities (convert to dense)
    if null_distribution == "network":
        p_values = (annotation_totals / total_nodes).A.flatten()  # Dense 1D array
    elif null_distribution == "annotation":
        p_values = (annotation_totals / annotation.sum()).A.flatten()  # Dense 1D array
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Observed counts (sparse matrix multiplication)
    annotated_counts = clusters @ annotation  # Sparse result
    annotated_counts_dense = annotated_counts.toarray()  # Convert for dense operations

    # Compute enrichment and depletion p-values
    enrichment_pvals = 1 - binom.cdf(annotated_counts_dense - 1, cluster_sizes.A, p_values)
    depletion_pvals = binom.cdf(annotated_counts_dense, cluster_sizes.A, p_values)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}


def compute_chi2_test(
    clusters: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute chi-squared test for enrichment and depletion in clusters with selectable null distribution.

    Args:
        clusters (csr_matrix): Sparse binary matrix representing clusters.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Total number of nodes in the network
    total_node_count = clusters.shape[0]

    if null_distribution == "network":
        # Case 1: Use all nodes as the background
        background_population = total_node_count
        cluster_sums = clusters.sum(axis=0)  # Column sums of clusters
        annotation_sums = annotation.sum(axis=0)  # Column sums of annotations
    elif null_distribution == "annotation":
        # Case 2: Only consider nodes with at least one annotation
        annotated_nodes = (
            np.ravel(annotation.sum(axis=1)) > 0
        )  # Row-wise sum to filter nodes with annotations
        background_population = annotated_nodes.sum()  # Total number of annotated nodes
        cluster_sums = clusters[annotated_nodes].sum(axis=0)  # Cluster sums for annotated nodes
        annotation_sums = annotation[annotated_nodes].sum(
            axis=0
        )  # Annotation sums for annotated nodes
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Convert to dense arrays for downstream computations
    cluster_sums = np.asarray(cluster_sums).reshape(-1, 1)  # Ensure column vector shape
    annotation_sums = np.asarray(annotation_sums).reshape(1, -1)  # Ensure row vector shape

    # Observed values: number of annotated nodes in each cluster
    observed = clusters.T @ annotation  # Shape: (clusters, annotation)
    # Expected values under the null
    expected = (cluster_sums @ annotation_sums) / background_population
    # Chi-squared statistic: sum((observed - expected)^2 / expected)
    with np.errstate(divide="ignore", invalid="ignore"):  # Handle divide-by-zero
        chi2_stat = np.where(expected > 0, np.power(observed - expected, 2) / expected, 0)

    # Compute p-values for enrichment (upper tail) and depletion (lower tail)
    enrichment_pvals = chi2.sf(chi2_stat, df=1)  # Survival function for upper tail
    depletion_pvals = chi2.cdf(chi2_stat, df=1)  # Cumulative distribution for lower tail

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}


def compute_hypergeom_test(
    clusters: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute hypergeometric test for enrichment and depletion in clusters with selectable null distribution.

    Args:
        clusters (csr_matrix): Sparse binary matrix representing clusters.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Get the total number of nodes in the network
    total_nodes = clusters.shape[1]

    # Compute sums
    cluster_sums = clusters.sum(axis=0).A.flatten()  # Convert to dense array
    annotation_sums = annotation.sum(axis=0).A.flatten()  # Convert to dense array

    if null_distribution == "network":
        background_population = total_nodes
    elif null_distribution == "annotation":
        annotated_nodes = annotation.sum(axis=1).A.flatten() > 0  # Boolean mask
        background_population = annotated_nodes.sum()
        cluster_sums = clusters[annotated_nodes].sum(axis=0).A.flatten()
        annotation_sums = annotation[annotated_nodes].sum(axis=0).A.flatten()
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Observed counts
    annotated_in_cluster = clusters.T @ annotation  # Sparse result
    annotated_in_cluster = annotated_in_cluster.toarray()  # Convert to dense
    # Align shapes for broadcasting
    cluster_sums = cluster_sums.reshape(-1, 1)
    annotation_sums = annotation_sums.reshape(1, -1)
    background_population = np.array(background_population).reshape(1, 1)

    # Compute hypergeometric p-values
    depletion_pvals = hypergeom.cdf(
        annotated_in_cluster, background_population, annotation_sums, cluster_sums
    )
    enrichment_pvals = hypergeom.sf(
        annotated_in_cluster - 1, background_population, annotation_sums, cluster_sums
    )

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
