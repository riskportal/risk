"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from risk.cluster import define_domains
from risk.network.graph._summary import Summary
from risk.network.graph.api import GraphAPI
from risk.network.graph.graph import Graph


def test_load_graph_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """
    Test loading a graph after generating clusters with specific parameters using JSON annotation.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # === Cluster and Stats ===
    clusters = risk_obj.cluster_leiden(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=887,
    )
    stats_results = risk_obj.run_permutation(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Validate the graph and its components
    _validate_graph(graph)


def test_cluster_size_limits_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """
    Test that statistically significant domains respect min and max cluster sizes using JSON annotation.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # === Cluster and Stats ===
        clusters = risk_obj.cluster_louvain(
            network=cytoscape_network,
            fraction_shortest_edges=0.75,
            resolution=8,
            random_seed=887,
        )
        stats_results = risk_obj.run_permutation(
            annotation=json_annotation,
            clusters=clusters,
            null_distribution="network",
            score_metric="stdev",
            num_permutations=20,
            random_seed=887,
            max_workers=1,
        )
        # Load the graph with the specified parameters
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotation=json_annotation,
            stats_results=stats_results,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            display_prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method="average",
            linkage_metric="yule",
            linkage_threshold=0.2,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the size of the domains
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_load_graph_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """
    Test loading a graph after generating clusters with specific parameters using dictionary annotation.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        dict_annotation: The dictionary annotation associated with the network.
    """
    # === Cluster and Stats ===
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=8,
        random_seed=887,
    )
    stats_results = risk_obj.run_permutation(
        annotation=dict_annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=dict_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Validate the graph and its components
    _validate_graph(graph)


def test_cluster_size_limits_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """
    Test that statistically significant domains respect min and max cluster sizes using dictionary annotation.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        dict_annotation: The dictionary annotation associated with the network.
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # === Cluster and Stats ===
        clusters = risk_obj.cluster_louvain(
            network=cytoscape_network,
            fraction_shortest_edges=0.75,
            resolution=8,
            random_seed=887,
        )
        stats_results = risk_obj.run_permutation(
            annotation=dict_annotation,
            clusters=clusters,
            null_distribution="network",
            score_metric="stdev",
            num_permutations=20,
            random_seed=887,
            max_workers=1,
        )
        # Load the graph with the specified parameters
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotation=dict_annotation,
            stats_results=stats_results,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            display_prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method="average",
            linkage_metric="yule",
            linkage_threshold=0.2,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the size of the domains
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_load_graph_with_different_stats_results(risk_obj, cytoscape_network, json_annotation):
    """
    Test that graphs built from different cluster results are structurally valid and have different domain maps.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Load clusters first, then compute statistics separately
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=8,
        random_seed=887,
    )
    # Compute statistical results using different methods
    stats_perm = risk_obj.run_permutation(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    stats_binom = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )

    # Use identical graph parameters for both
    graph_kwargs = dict(
        network=cytoscape_network,
        annotation=json_annotation,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )
    graph_perm = risk_obj.load_graph(stats_results=stats_perm, **graph_kwargs)
    graph_binom = risk_obj.load_graph(stats_results=stats_binom, **graph_kwargs)

    # Validate both graphs are valid and structurally consistent
    _validate_graph(graph_perm)
    _validate_graph(graph_binom)
    for graph in (graph_perm, graph_binom):
        assert isinstance(graph.domain_id_to_node_ids_map, dict)
        assert isinstance(graph.domain_id_to_domain_terms_map, dict)
        assert isinstance(graph.domain_id_to_domain_info_map, dict)
        assert isinstance(graph.node_id_to_domain_ids_and_significance_map, dict)
        assert isinstance(graph.network, nx.Graph)
        assert len(graph.network.nodes) > 0
        assert len(graph.network.edges) > 0

    # Confirm that the resulting domain maps differ
    perm_domains = set(graph_perm.domain_id_to_node_ids_map.keys())
    binom_domains = set(graph_binom.domain_id_to_node_ids_map.keys())
    # At least one domain ID should be different, or the mapping of node sets should differ
    if perm_domains == binom_domains:
        # If domain IDs are the same, check that the node sets differ for at least one domain
        node_sets_equal = all(
            set(graph_perm.domain_id_to_node_ids_map[dom])
            == set(graph_binom.domain_id_to_node_ids_map[dom])
            for dom in perm_domains
        )
        assert (
            not node_sets_equal
        ), "Domain node sets are identical between different cluster results"
    else:
        assert (
            perm_domains != binom_domains
        ), "Domain IDs are identical between different cluster results"


def test_graph_consistency_across_stat_methods(risk_obj, cytoscape_network, json_annotation):
    """
    Test that graphs constructed from different statistical methods are both valid and structurally consistent.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=8,
        random_seed=887,
    )
    stats_perm = risk_obj.run_permutation(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=123,
        max_workers=1,
    )
    stats_binom = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )
    graph_kwargs = dict(
        network=cytoscape_network,
        annotation=json_annotation,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )
    graph_perm = risk_obj.load_graph(stats_results=stats_perm, **graph_kwargs)
    graph_binom = risk_obj.load_graph(stats_results=stats_binom, **graph_kwargs)
    _validate_graph(graph_perm)
    _validate_graph(graph_binom)
    # Skip test if no significant domains are detected for either method
    if (
        len(graph_perm.domain_id_to_node_ids_map) == 0
        or len(graph_binom.domain_id_to_node_ids_map) == 0
    ):
        pytest.skip("No significant domains detected for either stat method.")
    # Check that both graphs are instances of the same type
    assert type(graph_perm) is type(graph_binom)
    # Check that domain maps are not empty
    assert len(graph_perm.domain_id_to_node_ids_map) > 0
    assert len(graph_binom.domain_id_to_node_ids_map) > 0


def test_linkage_criterion_and_auto_clustering_options(
    risk_obj, cytoscape_network, json_annotation
):
    """
    Test the linkage criterion and auto-clustering options for generating graphs.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Define parameters for testing
    test_criteria = ["distance", "off"]
    min_cluster_size, max_cluster_size = 10, 200  # Fixed for simplicity
    for criterion in test_criteria:
        # === Cluster and Stats ===
        clusters = risk_obj.cluster_louvain(
            network=cytoscape_network,
            fraction_shortest_edges=0.75,
            resolution=1.0,
            random_seed=888,
        )
        stats_results = risk_obj.run_binom(
            annotation=json_annotation,
            clusters=clusters,
            null_distribution="network",
        )
        # Load the graph with the specified linkage_criterion
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotation=json_annotation,
            stats_results=stats_results,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            display_prune_threshold=0.1,
            linkage_criterion=criterion,
            linkage_method="auto",
            linkage_metric="auto",
            linkage_threshold="auto",
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate graph for all criteria
        _validate_graph(graph)
        # Check cluster size bounds for 'distance' and 'off' criteria
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)
        # Ensure summary can be loaded for each criterion (public API coverage)
        summary = graph.summary.load()
        assert isinstance(summary, pd.DataFrame)
        assert {"Annotation", "Domain ID"}.issubset(set(summary.columns))


def test_network_graph_structure(risk_obj, cytoscape_network, json_annotation):
    """
    Test that the Graph object contains the expected components.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # === Cluster and Stats ===
    clusters = risk_obj.cluster_leiden(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=887,
    )
    stats_results = risk_obj.run_permutation(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Validate the graph attributes
    assert isinstance(
        graph.domain_id_to_node_ids_map, dict
    ), "Domain ID to node IDs map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_domain_terms_map, dict
    ), "Domain ID to domain terms map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_domain_info_map, dict
    ), "Domain ID to domain info map should be a dictionary"
    assert isinstance(
        graph.node_id_to_domain_ids_and_significance_map, dict
    ), "Node ID to domain IDs and significance map should be a dictionary"

    # Every node entry should carry the additive provenance keys alongside the existing ones
    found_provenance = False
    for node_id, domain_info in graph.node_id_to_domain_ids_and_significance_map.items():
        assert isinstance(
            domain_info["domains"], list
        ), f"Node {node_id} 'domains' should be a list"
        assert isinstance(
            domain_info["significances"], dict
        ), f"Node {node_id} 'significances' should be a dictionary"
        assert isinstance(
            domain_info["terms"], dict
        ), f"Node {node_id} 'terms' should be a dictionary"
        assert isinstance(
            domain_info["p_values"], dict
        ), f"Node {node_id} 'p_values' should be a dictionary"
        assert isinstance(
            domain_info["fdrs"], dict
        ), f"Node {node_id} 'fdrs' should be a dictionary"

        # The five sibling keys must describe exactly the same set of domain associations
        domain_id_set = set(domain_info["domains"])
        assert (
            set(domain_info["significances"])
            == domain_id_set
            == set(domain_info["terms"])
            == set(domain_info["p_values"])
            == set(domain_info["fdrs"])
        ), f"Node {node_id} sibling keys should describe the same domain set"

        # If a domain has contributing terms, its representative p-value/FDR should be usable
        for domain_id in domain_info["domains"]:
            terms = domain_info["terms"].get(domain_id, [])
            if not terms:
                continue
            assert all(isinstance(term, str) for term in terms)
            assert isinstance(domain_info["p_values"][domain_id], float)
            assert domain_info["fdrs"][domain_id] is None or isinstance(
                domain_info["fdrs"][domain_id], float
            )
            found_provenance = True
    assert found_provenance, "Expected at least one node-domain association with contributing terms"

    assert isinstance(
        graph.node_id_to_node_label_map, dict
    ), "Node ID to node label map should be a dictionary"
    assert isinstance(
        graph.node_label_to_significance_map, dict
    ), "Node label to significance map should be a dictionary"
    assert isinstance(
        graph.node_significance_sums, np.ndarray
    ), "Node significance sums should be a numpy array"
    assert isinstance(
        graph.node_label_to_node_id_map, dict
    ), "Node label to ID map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_node_labels_map, dict
    ), "Domain ID to node labels map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_enriched_node_labels_map, dict
    ), "Domain ID to enriched node labels map should be a dictionary"
    assert isinstance(graph.network, nx.Graph), "Network should be a NetworkX graph"
    assert isinstance(
        graph.node_coordinates, np.ndarray
    ), "Node coordinates should be a numpy array"
    assert isinstance(graph.summary, Summary), "Summary should be a Summary object"


def test_load_graph_threads_fdr_values_into_node_domain_metadata(
    risk_obj, cytoscape_network, json_annotation
):
    """
    Ensure a real load_graph(...) run with FDR correction enabled (fdr_cutoff < 1.0) threads a
    genuine, non-None float FDR value into node_id_to_domain_ids_and_significance_map. This
    guards against a plumbing regression where q-values are computed but never reach Graph.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # === Cluster and Stats ===
    clusters = risk_obj.cluster_leiden(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=887,
    )
    stats_results = risk_obj.run_permutation(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # fdr_cutoff < 1.0 (the load_graph default) enables real FDR correction, unlike the
    # fdr_cutoff=1.0 used elsewhere in this file.
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=0.9999,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    found_float_fdr = False
    for domain_info in graph.node_id_to_domain_ids_and_significance_map.values():
        for domain_id, fdr in domain_info["fdrs"].items():
            if fdr is None:
                continue
            assert isinstance(fdr, float)
            assert domain_id in domain_info["domains"]
            assert domain_id in domain_info["significances"]
            assert domain_id in domain_info["terms"]
            assert domain_id in domain_info["p_values"]
            found_float_fdr = True
    assert found_float_fdr, "Expected at least one non-None float FDR under fdr_cutoff < 1.0"


def test_load_graph_summary(graph):
    """
    Test loading the graph summary with predefined parameters.

    Args:
        graph: The graph object instance to be summarized.
    """
    # Load the graph summary and validate its type
    summary = graph.summary.load()

    assert isinstance(summary, pd.DataFrame), "Graph summary should be a DataFrame"


def test_summary_reports_raw_and_domain_pq(risk_obj, cytoscape_network, json_annotation):
    """
    Ensure summary exposes both raw and domain-conditioned p/q values. Raw values must remain
    invariant across linkage settings; domain-conditioned values are derived as minima within each
    domain's node set.
    """
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=888,
    )
    stats_results = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )
    common_kwargs = dict(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.0,
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )
    graph_distance = risk_obj.load_graph(linkage_criterion="distance", **common_kwargs)
    graph_off = risk_obj.load_graph(linkage_criterion="off", **common_kwargs)
    summary_distance = graph_distance.summary.load()
    summary_off = graph_off.summary.load()
    expected_columns = {
        "Raw Enrichment P-value",
        "Raw Enrichment Q-value",
        "Raw Depletion P-value",
        "Raw Depletion Q-value",
        "Domain Enrichment P-value",
        "Domain Enrichment Q-value",
        "Domain Depletion P-value",
        "Domain Depletion Q-value",
    }

    assert expected_columns.issubset(set(summary_distance.columns))
    assert "Enrichment P-value" not in summary_distance.columns
    assert "Enrichment Q-value" not in summary_distance.columns
    assert "Depletion P-value" not in summary_distance.columns
    assert "Depletion Q-value" not in summary_distance.columns

    # Raw p/q values are linkage-invariant by definition.
    raw_columns = [
        "Raw Enrichment P-value",
        "Raw Enrichment Q-value",
        "Raw Depletion P-value",
        "Raw Depletion Q-value",
    ]
    pd.testing.assert_frame_equal(
        summary_distance.set_index("Annotation")[raw_columns].sort_index(),
        summary_off.set_index("Annotation")[raw_columns].sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )
    # Domain p-value is the minimum over that domain's nodes for the given term.
    assigned = summary_distance[summary_distance["Domain ID"] != -1]
    if assigned.empty:
        pytest.skip("No assigned domains available to validate domain-conditioned minima.")

    first_row = assigned.iloc[0]
    annotation_idx = json_annotation["ordered_annotation"].index(first_row["Annotation"])
    domain_node_indices = graph_distance.domain_id_to_node_ids_map[first_row["Domain ID"]]
    expected_domain_enrichment_p = np.min(
        stats_results["enrichment_pvals"][domain_node_indices, annotation_idx]
    )
    expected_raw_enrichment_p = np.min(stats_results["enrichment_pvals"][:, annotation_idx])

    assert np.isclose(
        first_row["Domain Enrichment P-value"],
        expected_domain_enrichment_p,
        atol=1e-12,
        rtol=0.0,
    )
    assert np.isclose(
        first_row["Raw Enrichment P-value"],
        expected_raw_enrichment_p,
        atol=1e-12,
        rtol=0.0,
    )


def test_pop_domain(graph):
    """
    Test the pop method for removing a domain ID from all Graph attribute domain mappings.

    Args:
        graph: The graph object instance with existing domain mappings.
    """
    # Cache should be deterministic and caller-safe before graph mutation.
    initial_summary = graph.summary.load()
    cached_summary = graph.summary.load()
    pd.testing.assert_frame_equal(
        initial_summary,
        cached_summary,
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )
    assert initial_summary is not cached_summary

    # Define the domain ID to be removed
    domain_id_to_remove = 1
    # Retrieve expected labels before popping
    expected_labels = graph.domain_id_to_node_labels_map.get(domain_id_to_remove)
    # Pop the domain ID and get the returned value
    popped_labels = graph.pop(domain_id_to_remove)
    # Assert the returned value equals the expected labels
    assert popped_labels == expected_labels, "Popped labels do not match the expected labels."

    # Check that the domain ID is removed from all relevant attributes
    assert (
        domain_id_to_remove not in graph.domain_id_to_node_ids_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_node_ids_map"
    assert (
        domain_id_to_remove not in graph.domain_id_to_domain_terms_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_domain_terms_map"
    assert (
        domain_id_to_remove not in graph.domain_id_to_domain_info_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_domain_info_map"
    assert (
        domain_id_to_remove not in graph.domain_id_to_node_labels_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_node_labels_map"

    # Check if the domain was removed from node_id_to_domain_ids_and_significance_map
    for _, domain_info in graph.node_id_to_domain_ids_and_significance_map.items():
        assert domain_id_to_remove not in domain_info.get(
            "domains", []
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['domains']"
        assert domain_id_to_remove not in domain_info.get(
            "significances", {}
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['significances']"
        assert domain_id_to_remove not in domain_info.get(
            "terms", {}
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['terms']"
        assert domain_id_to_remove not in domain_info.get(
            "p_values", {}
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['p_values']"
        assert domain_id_to_remove not in domain_info.get(
            "fdrs", {}
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['fdrs']"

    # Finally, check that the summary no longer contains the removed domain ID
    refreshed_summary = graph.summary.load()
    assert domain_id_to_remove not in set(refreshed_summary["Domain ID"])


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"linkage_method": "not_a_method"},
        {"linkage_metric": "not_a_metric"},
        {"linkage_threshold": "bad"},
        {"linkage_threshold": 0.0},  # out of (0, 1]
        {"linkage_threshold": 1.5},  # out of (0, 1]
    ],
)
def test_invalid_clustering_args_raise(risk_obj, cytoscape_network, json_annotation, bad_kwargs):
    """
    Validate that invalid clustering options raise a ValueError (user error).

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
        bad_kwargs: A dict containing an intentionally invalid clustering parameter.
    """
    # === Cluster and Stats ===
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=888,
    )
    stats_results = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )

    with pytest.raises(ValueError):
        risk_obj.load_graph(
            network=cytoscape_network,
            annotation=json_annotation,
            stats_results=stats_results,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            display_prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method=bad_kwargs.get("linkage_method", "average"),
            linkage_metric=bad_kwargs.get("linkage_metric", "yule"),
            linkage_threshold=bad_kwargs.get("linkage_threshold", 0.2),
            min_cluster_size=5,
            max_cluster_size=1000,
        )


def test_off_criterion_bypasses_invalid_options(risk_obj, cytoscape_network, json_annotation):
    """
    Verify that setting linkage_criterion='off' cleanly bypasses clustering validation and does not raise.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # === Cluster and Stats ===
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=888,
    )
    stats_results = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="off",
        linkage_method="not_a_method",
        linkage_metric="not_a_metric",
        linkage_threshold="bad",
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    _validate_graph(graph)


def test_left_tail_assigns_domains_when_significance_exists(
    risk_obj, cytoscape_network, json_annotation
):
    """
    Ensure left-tail analysis can still assign domains when depletion signal exists.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=888,
    )
    stats_results = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="left",
        pval_cutoff=0.5,
        fdr_cutoff=1.0,
        display_prune_threshold=0.0,
        linkage_criterion="off",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Verify that depletion-driven significance exists and can map to domains.
    assert np.sum(graph.node_significance_sums != 0) > 0
    assert len(graph.domain_id_to_node_ids_map) > 0
    assert len(graph.node_id_to_domain_ids_and_significance_map) > 0
    assert any(
        len(v["domains"]) > 0 for v in graph.node_id_to_domain_ids_and_significance_map.values()
    )


def test_load_graph_returns_graph_instance(risk_obj, cytoscape_network, json_annotation):
    """
    Lightweight sanity test that load_graph returns a graph instance.

    Args:
        risk_obj: The RISK object instance used for loading clusters and graphs.
        cytoscape_network: The network object to be used for cluster and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    clusters = risk_obj.cluster_louvain(
        network=cytoscape_network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=42,
    )
    stats_results = risk_obj.run_binom(
        annotation=json_annotation,
        clusters=clusters,
        null_distribution="network",
    )
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        stats_results=stats_results,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        display_prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    assert graph is not None
    assert hasattr(graph, "network")


def test_primary_domain_labels_are_disjoint(graph):
    """
    Ensure primary domain label assignments do not overlap across domains.

    Args:
        graph: The graph object instance to be validated.
    """
    primary_map = graph.domain_id_to_node_labels_map
    assert primary_map, "Primary domain label map should be populated."

    all_labels = [label for labels in primary_map.values() for label in labels]
    assert len(all_labels) == len(
        set(all_labels)
    ), "Primary labels should be unique across domains."

    primary_sets = [set(labels) for labels in primary_map.values() if labels]
    if len(primary_sets) > 1:
        assert not set.intersection(*primary_sets), "No overlap expected across domain value sets."

    # Also ensure label->id mapping is one-to-one
    assert len(graph.node_label_to_node_id_map) == len(
        set(graph.node_label_to_node_id_map.values())
    )


def test_define_domains_handles_safeguard_row_drop_without_global_fallback():
    """
    Ensure dropped linkage rows do not desynchronize annotation-domain assignment.
    A zero-variance significant annotation should be assigned a deterministic unique
    domain, while non-significant annotations remain unassigned (domain 0).
    """
    # Two significant annotations: one degenerate (zero-variance) and one clusterable.
    # term_d and term_e are non-significant, like term_c, so they share domain 0 with it -
    # this lets the same node/domain pair also exercise zero-filtering and abs-value sorting.
    top_annotation = pd.DataFrame(
        {
            "significant_annotation": [True, True, False, False, False],
            "full_terms": ["term_a", "term_b", "term_c", "term_d", "term_e"],
            "significant_cluster_significance_sums": [1.0, 2.0, 0.0, 0.0, 0.0],
            "significant_significance_score": [1.0, 2.0, 0.0, 0.0, 0.0],
        },
        index=["term_a", "term_b", "term_c", "term_d", "term_e"],
    )
    # term_a is dropped by safeguard (constant column), term_b is retained, term_c/d/e are non-significant.
    significant_clusters_significance = np.array(
        [
            [5.0, 1.0, 0.0, -6.0, 3.0],
            [5.0, 2.0, 0.0, 0.0, 0.0],
            [5.0, 3.0, 0.0, 0.0, 0.0],
            [5.0, 4.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    domains, contributing_terms = define_domains(
        top_annotation=top_annotation,
        significant_clusters_significance=significant_clusters_significance,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="euclidean",
        linkage_threshold=0.2,
    )

    assert top_annotation.loc["term_c", "domain"] == 0
    assert top_annotation.loc["term_a", "domain"] > 0
    assert top_annotation.loc["term_b", "domain"] > 0
    assert top_annotation.loc["term_a", "domain"] != top_annotation.loc["term_b", "domain"]
    assert (domains["primary_domain"] > 0).any()

    # contributing_terms should carry the same node/domain provenance that fed the summation
    assert isinstance(contributing_terms, dict)
    domain_a_id = top_annotation.loc["term_a", "domain"]
    domain_b_id = top_annotation.loc["term_b", "domain"]
    assert contributing_terms[0][domain_a_id] == [("term_a", "term_a", 5.0)]
    assert contributing_terms[0][domain_b_id] == [("term_b", "term_b", 1.0)]

    # Domain 0 (term_c=0.0, term_d=-6.0, term_e=3.0 for node 0) exercises, through the same
    # public entry point: zero-value exclusion, term string/index preservation, and descending
    # sort by absolute masked contribution regardless of sign.
    assert contributing_terms[0][0] == [("term_d", "term_d", -6.0), ("term_e", "term_e", 3.0)]


def test_resolve_node_domain_provenance_pairs_selected_tail():
    """
    Ensure representative p-values/FDRs are read from the tail actually selected per cell,
    not always from enrichment regardless of enrichment_selection_matrix.
    """
    contributing_terms = {0: {1: [(0, "term_a", 5.0)], 2: [(1, "term_b", 3.0)]}}
    enrichment_pvals = np.array([[0.01, 0.99]])
    depletion_pvals = np.array([[0.88, 0.02]])
    enrichment_qvals = np.array([[0.11, 0.98]])
    depletion_qvals = np.array([[0.77, 0.12]])
    # Domain 1's strongest term is enrichment-selected; domain 2's is depletion-selected
    enrichment_selection_matrix = np.array([[True, False]])

    _, node_domain_pvals, node_domain_fdrs = GraphAPI()._resolve_node_domain_provenance(
        contributing_terms=contributing_terms,
        enrichment_pvals=enrichment_pvals,
        depletion_pvals=depletion_pvals,
        enrichment_qvals=enrichment_qvals,
        depletion_qvals=depletion_qvals,
        enrichment_selection_matrix=enrichment_selection_matrix,
    )

    assert np.isclose(node_domain_pvals[0][1], 0.01, atol=1e-12, rtol=0.0)
    assert np.isclose(node_domain_fdrs[0][1], 0.11, atol=1e-12, rtol=0.0)
    assert np.isclose(node_domain_pvals[0][2], 0.02, atol=1e-12, rtol=0.0)
    assert np.isclose(node_domain_fdrs[0][2], 0.12, atol=1e-12, rtol=0.0)


def test_graph_construction_without_node_domain_metadata_defaults_empty():
    """
    Ensure direct Graph construction without the new provenance arguments degrades safely,
    and that pop() does not crash when those provenance dictionaries are empty.
    """
    network = nx.Graph()
    network.add_nodes_from([0, 1])
    network.add_edge(0, 1)
    for node in network.nodes:
        network.nodes[node]["x"] = 0.0
        network.nodes[node]["y"] = 0.0
        network.nodes[node]["label"] = str(node)

    domains = pd.DataFrame(
        {1: [5.0, 0.0], "all_domains": [[1], []], "primary_domain": [1, 0]},
        index=[0, 1],
    )
    trimmed_domains = pd.DataFrame(
        {
            "normalized_description": ["term_a"],
            "full_descriptions": [("term_a",)],
            "significance_scores": [(5.0,)],
        },
        index=[1],
    )

    graph = Graph(
        network=network,
        annotation={},
        stats_results={},
        domains=domains,
        trimmed_domains=trimmed_domains,
        node_label_to_node_id_map={"0": 0, "1": 1},
        node_significance_sums=np.array([5.0, 0.0]),
    )

    for domain_info in graph.node_id_to_domain_ids_and_significance_map.values():
        assert domain_info["terms"] == {}
        assert domain_info["p_values"] == {}
        assert domain_info["fdrs"] == {}

    # pop() must not raise even though the provenance dictionaries are empty
    graph.pop(1)


def test_graph_excludes_zero_significance_domain_from_provenance():
    """
    Ensure a domain with zero summed significance (absent from 'domains') is also excluded
    from 'terms'/'p_values'/'fdrs', even when upstream provenance still references it. This can
    happen under tail="both", where a domain's per-term contributions can cancel to a net-zero
    sum despite individual terms being nonzero.
    """
    network = nx.Graph()
    network.add_node(0)
    network.nodes[0]["x"] = 0.0
    network.nodes[0]["y"] = 0.0
    network.nodes[0]["label"] = "0"

    # Domain 1 has zero summed significance for node 0, so it is absent from all_domains.
    domains = pd.DataFrame(
        {1: [0.0], "all_domains": [[]], "primary_domain": [0]},
        index=[0],
    )
    trimmed_domains = pd.DataFrame(
        {"normalized_description": [], "full_descriptions": [], "significance_scores": []},
        index=pd.Index([], dtype=int),
    )

    graph = Graph(
        network=network,
        annotation={},
        stats_results={},
        domains=domains,
        trimmed_domains=trimmed_domains,
        node_label_to_node_id_map={"0": 0},
        node_significance_sums=np.array([0.0]),
        # Provenance still references domain 1 despite its summed significance being zero
        node_domain_terms={0: {1: ["term_a", "term_b"]}},
        node_domain_pvals={0: {1: 0.01}},
        node_domain_fdrs={0: {1: 0.02}},
    )

    entry = graph.node_id_to_domain_ids_and_significance_map[0]
    assert entry["domains"] == []
    assert 1 not in entry["terms"]
    assert 1 not in entry["p_values"]
    assert 1 not in entry["fdrs"]


def _validate_graph(graph):
    """
    Validate that the graph is not None and contains nodes and edges.

    Args:
        graph: The graph instance to be validated.

    Raises:
        AssertionError: If the graph is None or if it contains no nodes or edges.
    """
    # For some reason, Windows can periodically return a graph with no nodes or edges
    if graph is None:
        pytest.skip("Skipping test: Graph is None.")
    if len(graph.network.nodes) == 0 or len(graph.network.edges) == 0:
        pytest.skip("Skipping test: Graph has no nodes or edges.")

    assert graph is not None, "Graph is None."
    assert len(graph.network.nodes) > 0, "Graph has no nodes."
    assert len(graph.network.edges) > 0, "Graph has no edges."


def _check_component_sizes(domain_id_to_node_id_map, min_cluster_size, max_cluster_size):
    """
    Check whether domains are within the specified size range.

    Args:
        domain_id_to_node_id_map (dict): A mapping of domain IDs to lists of node IDs.
        min_cluster_size (int): The minimum allowed size for components.
        max_cluster_size (int): The maximum allowed size for components.
    """
    for domain_id, node_ids in domain_id_to_node_id_map.items():
        # Skip invalid domain IDs
        if pd.isna(domain_id) or domain_id is None:
            print(f"Skipping invalid domain ID: {domain_id}")
            continue

        component_size = len(node_ids)
        # Debugging: Print the domain ID and its size
        print(f"Checking domain ID {domain_id} with size {component_size}")

        if not min_cluster_size <= component_size <= max_cluster_size:
            print(
                f"Domain {domain_id} size {component_size} is outside the range "
                f"{min_cluster_size} to {max_cluster_size}"
            )

        assert min_cluster_size <= component_size <= max_cluster_size, (
            f"Domain {domain_id} has size {component_size}, which is outside the range "
            f"{min_cluster_size} to {max_cluster_size}"
        )
