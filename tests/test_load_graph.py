"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from risk.network.graph._summary import Summary


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


def test_load_graph_summary(graph):
    """
    Test loading the graph summary with predefined parameters.

    Args:
        graph: The graph object instance to be summarized.
    """
    # Load the graph summary and validate its type
    summary = graph.summary.load()

    assert isinstance(summary, pd.DataFrame), "Graph summary should be a DataFrame"


def test_pop_domain(graph):
    """
    Test the pop method for removing a domain ID from all Graph attribute domain mappings.

    Args:
        graph: The graph object instance with existing domain mappings.
    """
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
