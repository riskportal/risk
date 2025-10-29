"""
risk/stats/_stats/permutation/_test_functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
from scipy.sparse import csr_matrix

# NOTE: Cython optimizations provided minimal performance benefits.
# The final version with Cython is archived in the `cython_permutation` branch.

# DISPATCH_TEST_FUNCTIONS can be found at the end of the file.


def compute_cluster_score_by_sum(
    clusters_matrix: csr_matrix, annotation_matrix: csr_matrix
) -> np.ndarray:
    """
    Compute the sum of attribute values for each cluster using sparse matrices.

    Args:
        clusters_matrix (csr_matrix): Sparse binary matrix representing clusters.
        annotation_matrix (csr_matrix): Sparse matrix representing annotation values.

    Returns:
        np.ndarray: Dense array of summed attribute values for each cluster.
    """
    # Calculate the cluster score as the dot product of clusters and annotation
    cluster_score = clusters_matrix @ annotation_matrix  # Sparse matrix multiplication
    # Convert the result to a dense array for downstream calculations
    cluster_score_dense = cluster_score.toarray()
    return cluster_score_dense


def compute_cluster_score_by_stdev(
    clusters_matrix: csr_matrix, annotation_matrix: csr_matrix
) -> np.ndarray:
    """
    Compute the standard deviation of cluster scores for sparse matrices.

    Args:
        clusters_matrix (csr_matrix): Sparse binary matrix representing clusters.
        annotation_matrix (csr_matrix): Sparse matrix representing annotation values.

    Returns:
        np.ndarray: Standard deviation of the cluster scores.
    """
    # Calculate the cluster score as the dot product of clusters and annotation
    cluster_score = clusters_matrix @ annotation_matrix  # Sparse matrix multiplication
    # Calculate the number of elements in each cluster (sum of rows)
    N = clusters_matrix.sum(axis=1).A.flatten()  # Convert to 1D array
    # Avoid division by zero by replacing zeros in N with np.nan temporarily
    N[N == 0] = np.nan
    # Compute the mean of the cluster scores
    M = cluster_score.multiply(1 / N[:, None]).toarray()  # Sparse element-wise division
    # Compute the mean of squares (EXX) directly using squared annotation matrix
    annotation_squared = annotation_matrix.multiply(annotation_matrix)  # Element-wise squaring
    EXX = (clusters_matrix @ annotation_squared).multiply(1 / N[:, None]).toarray()
    # Calculate variance as EXX - M^2
    variance = EXX - np.power(M, 2)
    # Compute the standard deviation as the square root of the variance
    cluster_stdev = np.sqrt(variance)
    # Replace np.nan back with zeros in case N was 0 (no elements in the cluster)
    cluster_stdev[np.isnan(cluster_stdev)] = 0
    return cluster_stdev


# Dictionary to dispatch statistical test functions based on the score metric
DISPATCH_TEST_FUNCTIONS = {
    "sum": compute_cluster_score_by_sum,
    "stdev": compute_cluster_score_by_stdev,
}
