"""
risk/network/graph/api
~~~~~~~~~~~~~~~~~~~~~~
"""

import copy
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from ...annotation import define_top_annotation
from ...log import log_header, logger, params
from ...cluster import (
    define_domains,
    process_significant_clusters,
    trim_domains,
)
from .graph import Graph
from ._stats import calculate_significance_matrices


class GraphAPI:
    """
    Handles the loading of network graphs and associated data.

    The GraphAPI class provides methods to load and process network graphs, annotations, and
    cluster results.
    """

    def load_graph(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        stats_results: Dict[str, Any],
        tail: str = "right",
        pval_cutoff: float = 0.01,
        fdr_cutoff: float = 0.9999,
        display_prune_threshold: float = 0.0,
        linkage_criterion: str = "distance",
        linkage_method: str = "average",
        linkage_metric: str = "yule",
        linkage_threshold: Union[float, str] = 0.2,
        min_cluster_size: int = 5,
        max_cluster_size: int = 1000,
    ) -> Graph:
        """
        Load and process the network graph, defining top annotations and domains.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            stats_results (Dict[str, Any]): Cluster significance data.
            tail (str, optional): Type of significance tail ("right", "left", "both"). Defaults to "right".
            pval_cutoff (float, optional): p-value cutoff for significance. Defaults to 0.01.
            fdr_cutoff (float, optional): FDR cutoff for significance. Defaults to 0.9999.
            display_prune_threshold (float, optional): Display-only pruning based on spatial layout
                distance that suppresses spatially diffuse or isolated regions in plots. Runs
                after enrichment and clustering on plotting matrices only, does not use
                enrichment strength or statistical significance or affect clustering, enrichment,
                or statistical testing, and defaults to 0.0.
            linkage_criterion (str, optional): Clustering criterion for defining domains. Defaults to "distance".
            linkage_method (str, optional): Clustering method to use. Choose "auto" to optimize. Defaults to "average".
            linkage_metric (str, optional): Metric to use for calculating distances. Choose "auto" to optimize.
                Defaults to "yule".
            linkage_threshold (float, str, optional): Threshold for clustering. Choose "auto" to optimize.
                Defaults to 0.2.
            min_cluster_size (int, optional): Minimum size for significant clusters. Defaults to 5.
            max_cluster_size (int, optional): Maximum size for significant clusters. Defaults to 1000.

        Returns:
            Graph: A fully initialized and processed Graph object.
        """
        # Log the parameters and display headers
        log_header("Finding significant clusters")
        params.log_graph(
            tail=tail,
            pval_cutoff=pval_cutoff,
            fdr_cutoff=fdr_cutoff,
            display_prune_threshold=display_prune_threshold,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
            linkage_threshold=linkage_threshold,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Make a copy of the network to avoid modifying the original
        network = copy.deepcopy(network)

        logger.debug(f"p-value cutoff: {pval_cutoff}")
        logger.debug(f"FDR BH cutoff: {fdr_cutoff}")
        logger.debug(
            f"Significance tail: '{tail}' ({'enrichment' if tail == 'right' else 'depletion' if tail == 'left' else 'both'})"
        )
        # Calculate significant clusters based on the provided parameters
        significant_clusters = calculate_significance_matrices(
            stats_results["depletion_pvals"],
            stats_results["enrichment_pvals"],
            tail=tail,
            pval_cutoff=pval_cutoff,
            fdr_cutoff=fdr_cutoff,
        )

        log_header("Processing significant clusters")
        # Process significant clusters for layout-based display pruning
        processed_clusters = process_significant_clusters(
            network=network,
            significant_clusters=significant_clusters,
            display_prune_threshold=display_prune_threshold,
        )

        log_header("Finding top annotations")
        logger.debug(f"Min cluster size: {min_cluster_size}")
        logger.debug(f"Max cluster size: {max_cluster_size}")
        # Define top annotations based on processed significant clusters
        top_annotation = self._define_top_annotation(
            network=network,
            annotation=annotation,
            processed_clusters=processed_clusters,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        log_header("Grouping clusters into domains")
        # Extract the significant significance matrix from the processed_clusters data
        significant_clusters_significance = processed_clusters["significant_significance_matrix"]
        # Define domains in the network using the specified clustering settings
        domains, contributing_terms = define_domains(
            top_annotation=top_annotation,
            significant_clusters_significance=significant_clusters_significance,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
            linkage_threshold=linkage_threshold,
        )
        # Trim domains and top annotations based on cluster size constraints
        domains, trimmed_domains = trim_domains(
            domains=domains,
            top_annotation=top_annotation,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Resolve each node-domain association's contributing terms and paired raw statistics
        node_domain_terms, node_domain_pvals, node_domain_fdrs = (
            self._resolve_node_domain_provenance(
                contributing_terms=contributing_terms,
                enrichment_pvals=stats_results["enrichment_pvals"],
                depletion_pvals=stats_results["depletion_pvals"],
                enrichment_qvals=significant_clusters["enrichment_qvals"],
                depletion_qvals=significant_clusters["depletion_qvals"],
                enrichment_selection_matrix=significant_clusters["enrichment_selection_matrix"],
            )
        )

        # Prepare node mapping and significance sums for the final Graph object
        ordered_nodes = annotation["ordered_nodes"]
        node_label_to_id = dict(zip(ordered_nodes, range(len(ordered_nodes))))
        node_significance_sums = processed_clusters["node_significance_sums"]

        # Return the fully initialized Graph object
        return Graph(
            network=network,
            annotation=annotation,
            stats_results=stats_results,
            domains=domains,
            trimmed_domains=trimmed_domains,
            node_label_to_node_id_map=node_label_to_id,
            node_significance_sums=node_significance_sums,
            node_domain_terms=node_domain_terms,
            node_domain_pvals=node_domain_pvals,
            node_domain_fdrs=node_domain_fdrs,
        )

    def _define_top_annotation(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        processed_clusters: Dict[str, Any],
        min_cluster_size: int = 5,
        max_cluster_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Define top annotations for the network.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): Annotation data for the network.
            processed_clusters (Dict[str, Any]): Processed cluster significance data.
            min_cluster_size (int, optional): Minimum size for clusters. Defaults to 5.
            max_cluster_size (int, optional): Maximum size for clusters. Defaults to 1000.

        Returns:
            pd.DataFrame: Top annotations identified within the network.
        """
        # Extract necessary data from annotation and processed_clusters
        ordered_annotation = annotation["ordered_annotation"]
        cluster_significance_sums = processed_clusters["cluster_significance_counts"]
        significant_significance_matrix = processed_clusters["significant_significance_matrix"]
        significant_binary_significance_matrix = processed_clusters[
            "significant_binary_significance_matrix"
        ]
        # Call external function to define top annotations
        return define_top_annotation(
            network=network,
            ordered_annotation_labels=ordered_annotation,
            cluster_significance_sums=cluster_significance_sums,
            significant_significance_matrix=significant_significance_matrix,
            significant_binary_significance_matrix=significant_binary_significance_matrix,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

    def _resolve_node_domain_provenance(
        self,
        contributing_terms: Dict[int, Dict[int, List[Tuple[int, str, float]]]],
        enrichment_pvals: np.ndarray,
        depletion_pvals: np.ndarray,
        enrichment_qvals: Union[np.ndarray, None],
        depletion_qvals: Union[np.ndarray, None],
        enrichment_selection_matrix: np.ndarray,
    ) -> Tuple[
        Dict[int, Dict[int, List[str]]],
        Dict[int, Dict[int, float]],
        Dict[int, Dict[int, Union[float, None]]],
    ]:
        """
        Resolve each node-domain association's contributing terms and pair the strongest term
        with its raw p-value and FDR.

        Args:
            contributing_terms (Dict[int, Dict[int, List[Tuple[int, str, float]]]]): Node ID to
                domain ID to a list of (term index, term string, masked significance) tuples,
                sorted descending by absolute significance.
            enrichment_pvals (np.ndarray): Raw enrichment p-value matrix [node, term].
            depletion_pvals (np.ndarray): Raw depletion p-value matrix [node, term].
            enrichment_qvals (np.ndarray, optional): FDR-corrected enrichment q-value matrix
                [node, term]. None when FDR correction was not applied.
            depletion_qvals (np.ndarray, optional): FDR-corrected depletion q-value matrix
                [node, term]. None when FDR correction was not applied.
            enrichment_selection_matrix (np.ndarray): Boolean matrix [node, term] indicating
                whether the enrichment (True) or depletion (False) tail was selected for that cell.

        Returns:
            Tuple[Dict[int, Dict[int, List[str]]], Dict[int, Dict[int, float]], Dict[int, Dict[int, Union[float, None]]]]:
                - Node ID to domain ID to the list of contributing term strings.
                - Node ID to domain ID to the raw p-value paired to the strongest contributing term.
                - Node ID to domain ID to the raw FDR paired to that same term, or None when FDR
                  correction was not applied.
        """
        node_domain_terms: Dict[int, Dict[int, List[str]]] = {}
        node_domain_pvals: Dict[int, Dict[int, float]] = {}
        node_domain_fdrs: Dict[int, Dict[int, Union[float, None]]] = {}
        for node_id, domain_terms in contributing_terms.items():
            for domain_id, terms in domain_terms.items():
                # terms is already sorted descending, so the strongest term is always first
                strongest_term_idx, _, _ = terms[0]
                is_enrichment = bool(enrichment_selection_matrix[node_id, strongest_term_idx])
                pvals = enrichment_pvals if is_enrichment else depletion_pvals
                qvals = enrichment_qvals if is_enrichment else depletion_qvals

                node_domain_terms.setdefault(node_id, {})[domain_id] = [
                    term_str for _, term_str, _ in terms
                ]
                node_domain_pvals.setdefault(node_id, {})[domain_id] = float(
                    pvals[node_id, strongest_term_idx]
                )
                node_domain_fdrs.setdefault(node_id, {})[domain_id] = (
                    float(qvals[node_id, strongest_term_idx]) if qvals is not None else None
                )

        return node_domain_terms, node_domain_pvals, node_domain_fdrs
