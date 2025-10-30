"""
tests/test_load_clusters
~~~~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import pytest
import numpy as np


CLUSTER_METHOD_DEFAULTS = {
    "louvain": {"resolution": 0.1, "random_seed": 887},
    "leiden": {"resolution": 1.0, "random_seed": 887},
}


def _run_cluster_method(
    risk_obj,
    clustering: str,
    network: nx.Graph,
    fraction_shortest_edges: float,
    **overrides,
):
    """
    Helper to invoke an explicit clustering method on the RISK API.
    """
    kwargs = CLUSTER_METHOD_DEFAULTS.get(clustering, {}).copy()
    kwargs.update(overrides)
    cluster_fn = getattr(risk_obj, f"cluster_{clustering}")
    return cluster_fn(
        network=network,
        fraction_shortest_edges=fraction_shortest_edges,
        **kwargs,
    )


@pytest.mark.parametrize(
    "clustering, fraction_shortest_edges",
    [
        ("greedy", 0.75),
        ("louvain", 0.80),
        ("leiden", 0.85),
        ("labelprop", 0.70),
        ("markov", 0.65),
        ("walktrap", 0.85),
        ("spinglass", 0.90),
    ],
)
def test_basic_cluster_loading(risk_obj, cytoscape_network, clustering, fraction_shortest_edges):
    """
    Test basic cluster loading functionality using various clustering methods and corresponding
    fraction_shortest_edges values.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        clustering: The clustering method(s) to be used.
        fraction_shortest_edges: The fraction(s) of shortest edges to consider.
    """
    clusters = _run_cluster_method(
        risk_obj,
        clustering,
        cytoscape_network,
        fraction_shortest_edges,
    )

    assert clusters is not None
    assert clusters.shape[0] == clusters.shape[1]  # Ensure square matrix
    assert clusters.getnnz() > 0  # Ensure clusters contain some entries


def test_load_clusters_empty_network(risk_obj):
    """
    Test loading clusters with an empty network to ensure proper error handling.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
    """
    empty_network = nx.Graph()

    with pytest.raises(
        ValueError,
        match="No edge lengths found in the graph. Ensure edges have 'length' attributes.",
    ):
        _run_cluster_method(
            risk_obj,
            "louvain",
            empty_network,
            0.75,
        )


@pytest.mark.parametrize(
    "clustering, fraction_shortest_edges",
    [
        ("greedy", 0.75),
        ("louvain", 0.80),
        ("leiden", 0.85),
        ("labelprop", 0.70),
        ("markov", 0.65),
        ("walktrap", 0.85),
        ("spinglass", 0.90),
    ],
)
def test_load_clusters_output_dimensions(
    risk_obj, cytoscape_network, clustering, fraction_shortest_edges
):
    """
    Test that the output cluster matrix dimensions correspond to the number of nodes in the network.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        clustering: The clustering method to be used.
        fraction_shortest_edges: The fraction of shortest edges to consider.
    """
    clusters = _run_cluster_method(
        risk_obj,
        clustering,
        cytoscape_network,
        fraction_shortest_edges,
    )

    num_nodes = len(cytoscape_network.nodes)
    assert clusters.shape[0] == num_nodes
    assert clusters.shape[1] == num_nodes


def test_load_clusters_deterministic_output(risk_obj, cytoscape_network):
    """
    Test that loading clusters with the same random seed produces consistent clustering results.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
    """
    clusters_1 = _run_cluster_method(
        risk_obj,
        "louvain",
        cytoscape_network,
        0.75,
        random_seed=887,
    )
    clusters_2 = _run_cluster_method(
        risk_obj,
        "louvain",
        cytoscape_network,
        0.75,
        random_seed=887,
    )

    # Validate that the cluster assignments are identical
    np.testing.assert_array_equal(clusters_1.toarray(), clusters_2.toarray())


@pytest.mark.parametrize(
    "clustering, fraction_shortest_edges, resolutions",
    [
        ("greedy", 0.25, (None,)),
        ("greedy", 1.0, (None,)),
        ("louvain", 0.5, (0.05, 1.0)),
        ("louvain", 0.9, (0.5, 2.5)),
        ("leiden", 0.25, (0.1, 0.25)),
        ("leiden", 0.75, (0.5, 1.5)),
        ("labelprop", 0.5, (None,)),
        ("markov", 0.3, (None,)),
        ("walktrap", 0.6, (None,)),
        ("spinglass", 0.7, (None,)),
    ],
)
def test_cluster_param_space(
    risk_obj,
    cytoscape_network,
    clustering,
    fraction_shortest_edges,
    resolutions,
):
    """
    Test cluster loading across a wide parameter space of clustering algorithms, edge fractions,
    and resolution parameters.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        clustering: The clustering method to be used.
        fraction_shortest_edges: The fraction of shortest edges to consider.
        resolutions: The resolution values to exercise for the clustering method (use (None,) if not applicable).
    """
    for resolution in resolutions:
        overrides = {}
        if resolution is not None:
            overrides["resolution"] = resolution
        clusters = _run_cluster_method(
            risk_obj,
            clustering,
            cytoscape_network,
            fraction_shortest_edges,
            **overrides,
        )

        assert clusters is not None
        assert clusters.shape[0] == clusters.shape[1]  # Ensure square matrix
        assert clusters.getnnz() > 0  # Ensure clusters contain some entries


# Additional tests for cluster loading


def test_disconnected_graph_clustering(risk_obj):
    """
    Test that clustering handles disconnected graphs correctly.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
    """
    # Create a disconnected graph: two components
    G = nx.Graph()
    G.add_edge("A", "B", length=1.0)
    G.add_edge("C", "D", length=1.0)
    # Expect that clusters are assigned per component (no cross-component clusters)
    clusters = _run_cluster_method(
        risk_obj,
        "louvain",
        G,
        1.0,
        random_seed=887,
    )
    arr = clusters.toarray()
    # Nodes in different components should not be clustered together
    idx = {n: i for i, n in enumerate(G.nodes())}
    assert arr[idx["A"], idx["C"]] == 0
    assert arr[idx["B"], idx["D"]] == 0
    # Nodes in same component should be clustered together
    assert arr[idx["A"], idx["B"]] == 1
    assert arr[idx["C"], idx["D"]] == 1


def test_different_methods_produce_different_results(risk_obj, cytoscape_network):
    """
    Test that different clustering methods produce different results.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
    """
    clusters_louvain = _run_cluster_method(
        risk_obj,
        "louvain",
        cytoscape_network,
        0.75,
        random_seed=887,
    )
    clusters_leiden = _run_cluster_method(
        risk_obj,
        "leiden",
        cytoscape_network,
        0.75,
        random_seed=887,
    )
    # They should not be exactly equal (though not impossible)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(clusters_louvain.toarray(), clusters_leiden.toarray())


def test_cluster_matrix_symmetry(risk_obj, cytoscape_network):
    """
    Test that the cluster matrix is symmetric.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
    """
    clusters = _run_cluster_method(
        risk_obj,
        "louvain",
        cytoscape_network,
        0.75,
        random_seed=887,
    )
    arr = clusters.toarray()
    assert np.allclose(arr, arr.T)


def test_fraction_shortest_edges_sensitivity(risk_obj, cytoscape_network):
    """
    Test that different fraction_shortest_edges values produce different cluster matrices.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
    """
    clusters_low = _run_cluster_method(
        risk_obj,
        "louvain",
        cytoscape_network,
        0.5,
        random_seed=887,
    )
    clusters_high = _run_cluster_method(
        risk_obj,
        "louvain",
        cytoscape_network,
        0.95,
        random_seed=887,
    )
    # They should not be exactly equal (though not impossible)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(clusters_low.toarray(), clusters_high.toarray())
