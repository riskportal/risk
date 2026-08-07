"""
risk/cluster/label
~~~~~~~~~~~~~~~~~~
"""

from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from risk.annotation import get_weighted_description

from ..log import logger

# Keep candidates in deterministic order so auto-optimization is stable across processes.
# fmt: off
LINKAGE_METHODS = tuple(sorted({
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
}))
LINKAGE_METRICS = tuple(sorted({
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
    "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis", "matching", "minkowski",
    "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule",
}))
# fmt: on


def define_domains(
    top_annotation: pd.DataFrame,
    significant_clusters_significance: np.ndarray,
    linkage_criterion: str,
    linkage_method: str,
    linkage_metric: str,
    linkage_threshold: Union[float, str],
) -> Tuple[pd.DataFrame, Dict[int, Dict[int, List[Tuple[int, str, float]]]]]:
    """
    Define domains and assign nodes to these domains based on their significance scores and clustering,
    handling errors by assigning unique domains when clustering fails.

    Args:
        top_annotation (pd.DataFrame): DataFrame of top annotations data for the network nodes.
        significant_clusters_significance (np.ndarray): The binary significance matrix below alpha.
        linkage_criterion (str): The clustering criterion for defining groups. Choose "off" to disable clustering.
        linkage_method (str): The linkage method for clustering. Choose "auto" to optimize.
        linkage_metric (str): The linkage metric for clustering. Choose "auto" to optimize.
        linkage_threshold (float, str): The threshold for clustering. Choose "auto" to optimize.

    Returns:
        Tuple[pd.DataFrame, Dict[int, Dict[int, List[Tuple[int, str, float]]]]]:
            - DataFrame with the primary domain for each node.
            - Nonzero contributing terms per node and domain, sorted descending by absolute significance.

    Raises:
        ValueError: If any clustering argument is invalid.
    """
    # Validate args first; let user mistakes raise immediately
    clustering_off = _validate_clustering_args(
        linkage_criterion, linkage_method, linkage_metric, linkage_threshold
    )

    # If clustering is turned off, assign unique domains and skip
    if clustering_off:
        n_rows = len(top_annotation)
        logger.warning("Clustering is turned off. Skipping clustering.")
        top_annotation["domain"] = range(1, n_rows + 1)
    else:
        # Transpose the matrix to cluster annotations
        significant_mask = top_annotation["significant_annotation"]
        if not significant_mask.any():
            raise ValueError(
                "Domain clustering aborted: no annotations remained significant after enrichment filtering. "
                "RISK did not detect any terms passing the significance thresholds, so domains cannot be defined. "
                "To proceed without domain clustering, set `linkage_criterion='off'`, which disables clustering "
                "and assigns domains directly from raw enrichment values. "
                "Alternatively, consider relaxing `pval_cutoff`/`fdr_cutoff` or reviewing annotation coverage."
            )
        significant_annotation_ids = top_annotation.index[significant_mask].tolist()
        m = significant_clusters_significance[:, significant_mask].T
        # Clean matrix values and keep a row mask to preserve annotation-domain alignment.
        m, kept_rows_mask = _safeguard_matrix(m)
        if len(significant_annotation_ids) != len(kept_rows_mask):
            raise ValueError(
                "Annotation row mapping mismatch: significant annotation ids and keep mask length differ."
            )
        kept_annotation_ids = [
            annotation_id
            for annotation_id, keep in zip(significant_annotation_ids, kept_rows_mask)
            if keep
        ]
        dropped_annotation_ids = [
            annotation_id
            for annotation_id, keep in zip(significant_annotation_ids, kept_rows_mask)
            if not keep
        ]

        top_annotation["domain"] = 0
        next_domain_id = 1
        if kept_annotation_ids:
            try:
                # Optimize silhouette score across different linkage methods and metrics
                (
                    best_linkage,
                    best_metric,
                    best_threshold,
                ) = _optimize_silhouette_across_linkage_and_metrics(
                    m, linkage_criterion, linkage_method, linkage_metric, linkage_threshold
                )
                # Perform hierarchical clustering
                Z = linkage(m, method=best_linkage, metric=best_metric)
                logger.warning(
                    f"Linkage criterion: '{linkage_criterion}'\nLinkage method: '{best_linkage}'\nLinkage metric: '{best_metric}'\nLinkage threshold: {round(best_threshold, 3)}"
                )
                # Calculate the optimal threshold for clustering
                cut_param = np.max(Z[:, 2]) * best_threshold
                domains = fcluster(Z, cut_param, criterion=linkage_criterion).astype(int)
                top_annotation.loc[kept_annotation_ids, "domain"] = domains
                next_domain_id = int(np.max(domains)) + 1
            except (LinAlgError, ValueError):
                # Numerical errors or degenerate input are handled gracefully (not user error)
                logger.error(
                    "Clustering failed due to numerical or data degeneracy. Assigning unique domains to significant annotations."
                )
                n_kept = len(kept_annotation_ids)
                top_annotation.loc[kept_annotation_ids, "domain"] = np.arange(
                    next_domain_id,
                    next_domain_id + n_kept,
                    dtype=int,
                )
                next_domain_id += n_kept

        if dropped_annotation_ids:
            logger.warning(
                f"{len(dropped_annotation_ids)} significant annotations were excluded from linkage due to invalid or zero-variance vectors and were assigned unique domains."
            )
            n_dropped = len(dropped_annotation_ids)
            top_annotation.loc[dropped_annotation_ids, "domain"] = np.arange(
                next_domain_id,
                next_domain_id + n_dropped,
                dtype=int,
            )
            next_domain_id += n_dropped

    # Create DataFrames to store domain information
    node_to_significance = pd.DataFrame(
        data=significant_clusters_significance,
        columns=pd.MultiIndex.from_arrays(
            [top_annotation.index.values, top_annotation["domain"]],
            names=["annotation", "domain"],
        ),
    )
    # Capture nonzero contributing terms before domain-level summation collapses term identity
    contributing_terms = _rank_contributing_terms(node_to_significance, top_annotation)
    node_to_domain = node_to_significance.T.groupby(level="domain").sum().T

    # Find the dominant domain per node using absolute significance:
    # right-tail scores are positive while left-tail scores are negative.
    domain_signal = node_to_domain.loc[:, 1:]
    t_abs_max = domain_signal.abs().max(axis=1)
    t_idxmax = domain_signal.abs().idxmax(axis=1)
    t_idxmax[t_abs_max == 0] = 0

    # Assign all domains where the node has any non-zero significance, regardless of sign.
    node_to_domain["all_domains"] = domain_signal.apply(
        lambda row: list(row[row.abs() > 0].index), axis=1
    )
    # Assign primary domain
    node_to_domain["primary_domain"] = t_idxmax

    return node_to_domain, contributing_terms


def trim_domains(
    domains: pd.DataFrame,
    top_annotation: pd.DataFrame,
    min_cluster_size: int = 5,
    max_cluster_size: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trim domains that do not meet size criteria and find outliers.

    Args:
        domains (pd.DataFrame): DataFrame of domain data for the network nodes.
        top_annotation (pd.DataFrame): DataFrame of top annotations data for the network nodes.
        min_cluster_size (int, optional): Minimum size of a cluster to be retained. Defaults to 5.
        max_cluster_size (int, optional): Maximum size of a cluster to be retained. Defaults to 1000.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Trimmed domains (pd.DataFrame)
            - A DataFrame with domain labels (pd.DataFrame)
    """
    # Identify domains to remove based on size criteria
    domain_counts = domains["primary_domain"].value_counts()
    to_remove = set(
        domain_counts[(domain_counts < min_cluster_size) | (domain_counts > max_cluster_size)].index
    )

    # Add invalid domain IDs
    invalid_domain_id = 888888
    invalid_domain_ids = {0, invalid_domain_id}
    # Mark domains to be removed
    top_annotation["domain"] = top_annotation["domain"].replace(to_remove, invalid_domain_id)
    domains.loc[domains["primary_domain"].isin(to_remove), ["primary_domain"]] = invalid_domain_id

    # Normalize "num significant clusters" by percentile for each domain and scale to 0-10
    top_annotation["normalized_value"] = top_annotation.groupby("domain")[
        "significant_cluster_significance_sums"
    ].transform(lambda x: (x.rank(pct=True) * 10).apply(np.ceil).astype(int))
    # Modify the lambda function to pass both full_terms and significant_significance_score
    top_annotation["combined_terms"] = top_annotation.apply(
        lambda row: " ".join([str(row["full_terms"])] * row["normalized_value"]), axis=1
    )

    # Perform the groupby operation while retaining the other columns and adding the weighting with significance scores
    domain_labels = (
        top_annotation.groupby("domain")
        .agg(
            full_terms=("full_terms", lambda x: list(x)),
            significance_scores=("significant_significance_score", lambda x: list(x)),
        )
        .reset_index()
    )
    domain_labels["combined_terms"] = domain_labels.apply(
        lambda row: get_weighted_description(
            pd.Series(row["full_terms"]), pd.Series(row["significance_scores"])
        ),
        axis=1,
    )

    # Rename the columns as necessary
    trimmed_domains_matrix = domain_labels.rename(
        columns={
            "domain": "id",
            "combined_terms": "normalized_description",
            "full_terms": "full_descriptions",
            "significance_scores": "significance_scores",
        }
    ).set_index("id")

    # Remove invalid domains
    valid_domains = domains[~domains["primary_domain"].isin(invalid_domain_ids)]
    valid_trimmed_domains_matrix = trimmed_domains_matrix[
        ~trimmed_domains_matrix.index.isin(invalid_domain_ids)
    ]
    return valid_domains, valid_trimmed_domains_matrix


def _rank_contributing_terms(
    node_to_significance: pd.DataFrame,
    top_annotation: pd.DataFrame,
) -> Dict[int, Dict[int, List[Tuple[int, str, float]]]]:
    """
    Rank each node's nonzero contributing annotation terms within each domain by masked significance.

    Args:
        node_to_significance (pd.DataFrame): Node-by-term significance matrix with a MultiIndex
            (annotation, domain) column structure, prior to domain-level summation.
        top_annotation (pd.DataFrame): Top annotation data, used to resolve annotation term
            indices to their full term strings.

    Returns:
        Dict[int, Dict[int, List[Tuple[int, str, float]]]]: Mapping of node ID to domain ID to a
            list of (term index, term string, masked significance) tuples with nonzero
            significance, sorted descending by absolute significance.
    """
    contributing_terms: Dict[int, Dict[int, List[Tuple[int, str, float]]]] = {}
    for domain_id, domain_group in node_to_significance.T.groupby(level="domain"):
        term_indices = domain_group.index.get_level_values("annotation")
        term_strings = top_annotation.loc[term_indices, "full_terms"].tolist()
        for node_id, node_values in domain_group.items():
            # Only nonzero terms actually contribute to this node's domain association
            nonzero_terms = [
                (term_idx, term_str, value)
                for term_idx, term_str, value in zip(term_indices, term_strings, node_values)
                if value != 0
            ]
            if not nonzero_terms:
                continue
            # Rank by magnitude so depletion (negative) signal sorts consistently with enrichment
            nonzero_terms.sort(key=lambda term: abs(term[2]), reverse=True)
            contributing_terms.setdefault(node_id, {})[int(domain_id)] = nonzero_terms

    return contributing_terms


def _validate_clustering_args(
    linkage_criterion: str,
    linkage_method: str,
    linkage_metric: str,
    linkage_threshold: Union[float, str],
) -> bool:
    """
    Validate user-provided clustering arguments.

    Returns:
        bool: True if clustering is turned off (criterion == 'off'); False otherwise.

    Raises:
        ValueError: If any argument is invalid (user error).
    """
    allowed_criteria = {"distance", "off"}
    if linkage_criterion not in allowed_criteria:
        raise ValueError(
            f"Invalid linkage_criterion '{linkage_criterion}'. Allowed values are 'distance' or 'off'."
        )
    # Allow opting out of clustering without raising
    if linkage_criterion == "off":
        return True
    # Validate linkage method (allow "auto")
    if linkage_method != "auto" and linkage_method not in LINKAGE_METHODS:
        raise ValueError(
            f"Invalid linkage_method '{linkage_method}'. Allowed values are 'auto' or one of: {sorted(LINKAGE_METHODS)}"
        )
    # Validate linkage metric (allow "auto")
    if linkage_metric != "auto" and linkage_metric not in LINKAGE_METRICS:
        raise ValueError(
            f"Invalid linkage_metric '{linkage_metric}'. Allowed values are 'auto' or one of: {sorted(LINKAGE_METRICS)}"
        )
    # Validate linkage threshold (allow "auto"; otherwise must be float in (0, 1])
    if linkage_threshold != "auto":
        try:
            lt = float(linkage_threshold)
        except (TypeError, ValueError):
            raise ValueError("linkage_threshold must be 'auto' or a float in the interval (0, 1].")
        if not (0.0 < lt <= 1.0):
            raise ValueError(f"linkage_threshold must be within (0, 1]. Received: {lt}")

    return False


def _safeguard_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safeguard the matrix by replacing NaN/Inf values and dropping zero-variance rows.

    Args:
        matrix (np.ndarray): Data matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Safeguarded matrix with only non-zero-variance rows.
            - Boolean keep mask aligned to the original row order.
    """
    n_rows = matrix.shape[0]
    # Safety guard: handle empty or invalid matrices
    if matrix.size == 0 or not np.isfinite(matrix).any():
        logger.warning(
            "Input matrix is empty or contains no finite values. Returning an empty matrix."
        )
        return np.empty((0, matrix.shape[1]), dtype=float), np.zeros(n_rows, dtype=bool)
    # Replace NaN with column mean
    nan_replacement = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isnan(matrix), nan_replacement, matrix)
    # Replace Inf/-Inf with maximum/minimum finite values
    finite_max = np.nanmax(matrix[np.isfinite(matrix)])
    finite_min = np.nanmin(matrix[np.isfinite(matrix)])
    matrix = np.where(np.isposinf(matrix), finite_max, matrix)
    matrix = np.where(np.isneginf(matrix), finite_min, matrix)
    # Keep only rows that can contribute to distance-based linkage.
    kept_rows_mask = np.var(matrix, axis=1) > 0
    matrix = matrix[kept_rows_mask]
    return matrix, kept_rows_mask


def _optimize_silhouette_across_linkage_and_metrics(
    m: np.ndarray,
    linkage_criterion: str,
    linkage_method: str,
    linkage_metric: str,
    linkage_threshold: Union[str, float],
) -> Tuple[str, str, float]:
    """
    Optimize silhouette score across different linkage methods and metrics.

    Args:
        m (np.ndarray): Data matrix.
        linkage_criterion (str): Clustering criterion.
        linkage_method (str): Linkage method for clustering. Choose "auto" to optimize.
        linkage_metric (str): Linkage metric for clustering. Choose "auto" to optimize.
        linkage_threshold (Union[str, float]): Threshold for clustering. Choose "auto" to optimize.

    Returns:
        Tuple[str, str, float]:
            - Best linkage method (str)
            - Best linkage metric (str)
            - Best threshold (float)
    """
    # Initialize best overall values
    best_overall_method = linkage_method
    best_overall_metric = linkage_metric
    best_overall_threshold = 0.0
    best_overall_score = -np.inf

    # Set linkage methods and metrics to all combinations if "auto" is selected
    linkage_methods = LINKAGE_METHODS if linkage_method == "auto" else [linkage_method]
    linkage_metrics = LINKAGE_METRICS if linkage_metric == "auto" else [linkage_metric]
    total_combinations = len(linkage_methods) * len(linkage_metrics)

    # Evaluating optimal linkage method and metric
    for method, metric in tqdm(
        product(linkage_methods, linkage_metrics),
        desc="Evaluating linkage methods and metrics",
        total=total_combinations,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        # Some linkage methods and metrics may not work with certain data
        try:
            Z = linkage(m, method=method, metric=metric)
            if linkage_threshold == "auto":
                try:
                    threshold, score = _find_best_silhouette_score(Z, m, metric, linkage_criterion)
                except (ValueError, LinAlgError):
                    continue  # Skip to the next combination
                current_threshold = threshold
            else:
                cut_param = linkage_threshold * np.max(Z[:, 2])
                score = silhouette_score(
                    m,
                    fcluster(Z, cut_param, criterion=linkage_criterion),
                    metric=metric,
                )
                current_threshold = linkage_threshold
        except (ValueError, LinAlgError):
            continue  # Skip to the next combination

        is_better_score = score > best_overall_score
        is_tied_score = (
            np.isfinite(score)
            and np.isfinite(best_overall_score)
            and np.isclose(score, best_overall_score, rtol=0.0, atol=1e-12)
        )
        is_better_tie_break = (method, metric, float(current_threshold)) < (
            best_overall_method,
            best_overall_metric,
            float(best_overall_threshold),
        )
        if is_better_score or (is_tied_score and is_better_tie_break):
            best_overall_score = score
            best_overall_threshold = float(current_threshold)  # Ensure it's a float
            best_overall_method = method
            best_overall_metric = metric

    # Ensure that we always return a valid tuple:
    if best_overall_score == -np.inf:
        # No valid linkage was found; return default values.
        best_overall_threshold = float(linkage_threshold) if linkage_threshold != "auto" else 0.0
        best_overall_method = linkage_method
        best_overall_metric = linkage_metric

    return best_overall_method, best_overall_metric, best_overall_threshold


def _find_best_silhouette_score(
    Z: np.ndarray,
    m: np.ndarray,
    linkage_metric: str,
    linkage_criterion: str,
) -> Tuple[float, float]:
    """
    Find the best silhouette score via discrete enumeration over linkage merge heights.

    Args:
        Z (np.ndarray): Linkage matrix.
        m (np.ndarray): Data matrix.
        linkage_metric (str): Linkage metric for silhouette score calculation.
        linkage_criterion (str): Clustering criterion.

    Returns:
        Tuple[float, float]:
            - Best threshold (float): Normalized fraction in (0, 1] of the merge height that
              yields the best silhouette score.
            - Best silhouette score (float): The highest silhouette score achieved.

    Raises:
        ValueError: If no candidate merge height yields a scoreable partition (requires
            2 to N-1 clusters).
    """
    # fcluster(..., criterion="distance") only changes partition at actual merge heights, so
    # the silhouette-vs-threshold objective is a stepwise, often multimodal function of the
    # cut height. Binary search assumes unimodality between two probed endpoints and can
    # converge on a suboptimal cut; enumerating the true candidate heights cannot miss the
    # best partition. This is slower at large N (O(N) candidates vs. a fixed probe budget),
    # but searches the actual space the objective lives on.
    raw_max = np.max(Z[:, 2])
    n = m.shape[0]
    dist_matrix = squareform(pdist(m, metric=linkage_metric))

    best_score = -np.inf
    best_height = None
    for height in np.unique(Z[:, 2]):
        labels = fcluster(Z, height, criterion=linkage_criterion)
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= n:
            continue
        try:
            score = silhouette_score(dist_matrix, labels, metric="precomputed")
        except ValueError:
            continue

        is_better = score > best_score
        is_tied = (
            np.isfinite(score)
            and np.isfinite(best_score)
            and np.isclose(score, best_score, rtol=0.0, atol=1e-12)
        )
        if is_better or (is_tied and height < best_height):
            best_score = score
            best_height = height

    if best_height is None:
        raise ValueError(
            "No candidate linkage merge height yielded a scoreable silhouette partition "
            "(requires 2 to N-1 clusters)."
        )

    return float(best_height) / float(raw_max), float(best_score)
