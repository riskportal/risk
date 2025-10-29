"""
tests/test_load_clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np
import pytest


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_binom(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """
    Test loading clusters using the binomial test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the binomial test (either 'network' or 'annotation').
    """
    clusters = risk_obj.load_clusters_binom(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_chi2(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """
    Test loading clusters using the chi-squared test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the chi-squared test (either 'network' or 'annotation').
    """
    clusters = risk_obj.load_clusters_chi2(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_hypergeom(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """
    Test loading clusters using the hypergeometric test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the hypergeometric test (either 'network' or 'annotation').
    """
    clusters = risk_obj.load_clusters_hypergeom(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_permutation_single_process(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test loading clusters using a single process with the permutation test with multiple
    null distributions.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the permutation test (either 'network' or 'annotation').
    """
    # Load clusters with 1 process
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="leiden",
        louvain_resolution=0.01,
        leiden_resolution=1.0,
        fraction_shortest_edges=0.25,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,  # Set to 10 permutations as requested
        random_seed=887,
        max_workers=1,  # Single process
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


def test_load_clusters_permutation_multi_process(risk_obj, cytoscape_network, json_annotation):
    """
    Test loading clusters using multiple processes with the permutation test.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
    """
    # Load clusters with 4 processes
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,  # Set to 10 permutations as requested
        random_seed=887,
        max_workers=4,  # Four processes
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


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
        (["louvain"], [0.75]),
        (["louvain", "labelprop"], [0.75, 0.70]),
        (["louvain", "markov"], [0.75, 0.65]),
        (["labelprop", "walktrap", "spinglass"], [0.70, 0.85, 0.90]),
        (
            [
                "louvain",
                "labelprop",
                "markov",
                "walktrap",
                "spinglass",
                "leiden",
            ],
            [0.75, 0.70, 0.65, 0.85, 0.90, 0.50],
        ),
        (
            [
                "louvain",
                "leiden",
                "labelprop",
                "markov",
                "walktrap",
                "spinglass",
                "greedy",
            ],
            [0.75, 0.70, 0.65, 0.85, 0.90, 0.80, 0.90],
        ),
    ],
)
def test_load_clusters_with_various_clustering(
    risk_obj, cytoscape_network, json_annotation, clustering, fraction_shortest_edges
):
    """
    Test loading clusters using various clustering methods with matching edge length thresholds.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        clustering: The specific clustering method(s) to be used for generating clusters.
        fraction_shortest_edges: The edge length threshold(s) corresponding to each clustering method.
    """
    # Load clusters with the current clustering method(s) and matching edge length threshold(s)
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering=clustering,
        louvain_resolution=8,
        fraction_shortest_edges=fraction_shortest_edges,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


@pytest.mark.parametrize("score_metric", ["sum", "stdev"])
def test_load_clusters_with_various_score_metrics(
    risk_obj, cytoscape_network, json_annotation, score_metric
):
    """
    Test loading clusters using various score metrics.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        score_metric: The specific score metric to be used for generating clusters.
    """
    # Load clusters with the specified score metric
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",  # Using louvain as the clustering method
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric=score_metric,
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_with_various_null_distributions(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test loading clusters using various null distributions.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        null_distribution: The specific null distribution to be used for generating clusters.
    """
    # Load clusters with the specified null distribution
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",  # Using louvain as the clustering method
        fraction_shortest_edges=0.75,
        score_metric="stdev",  # Using stdev as the score metric
        null_distribution=null_distribution,  # Parametrized null distribution
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    assert clusters is not None
    assert len(clusters) > 0  # Ensure clusters are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_structure(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """Test the structure of the clusters object."""
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,
        random_seed=887,
        max_workers=1,
    )

    # Validate that the clusters object has the expected keys
    assert "depletion_pvals" in clusters, "Clusters should contain a 'depletion_pvals' key"
    assert "enrichment_pvals" in clusters, "Clusters should contain an 'enrichment_pvals' key"
    assert isinstance(
        clusters["depletion_pvals"], np.ndarray
    ), "'depletion_pvals' should be a numpy array"
    assert isinstance(
        clusters["enrichment_pvals"], np.ndarray
    ), "'enrichment_pvals' should be a numpy array"


def test_load_clusters_empty_network(risk_obj, json_annotation):
    """Test loading clusters with an empty network."""
    # Create an empty network
    empty_network = nx.Graph()

    # Expect a ValueError due to missing edge lengths
    with pytest.raises(
        ValueError,
        match="No edge lengths found in the graph. Ensure edges have 'length' attributes.",
    ):
        risk_obj.load_clusters_permutation(
            network=empty_network,
            annotation=json_annotation,
            clustering="louvain",
            louvain_resolution=8,
            fraction_shortest_edges=0.75,
            score_metric="stdev",
            null_distribution="network",
            num_permutations=10,
            random_seed=887,
            max_workers=1,
        )


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_clusters_output_dimensions(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test that the output dimensions of clusters match expectations.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
        null_distribution: The specific null distribution to be used for generating clusters.
    """
    clusters = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,
        random_seed=887,
        max_workers=1,
    )

    # Validate dimensions of p-value matrices
    num_nodes = len(cytoscape_network.nodes)
    num_annotation = len(json_annotation["ordered_annotation"])
    assert clusters["depletion_pvals"].shape == (
        num_nodes,
        num_annotation,
    ), "Depletion p-values matrix dimensions do not match the expected size"
    assert clusters["enrichment_pvals"].shape == (
        num_nodes,
        num_annotation,
    ), "Enrichment p-values matrix dimensions do not match the expected size"


def test_load_clusters_deterministic_output(risk_obj, cytoscape_network, json_annotation):
    """
    Test that loading clusters with the same random seed produces consistent results.

    Args:
        risk_obj: The RISK object instance used for loading clusters.
        cytoscape_network: The network object to be used for cluster generation.
        json_annotation: The annotation associated with the network.
    """
    clusters_1 = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,
        random_seed=887,
        max_workers=1,
    )
    clusters_2 = risk_obj.load_clusters_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        clustering="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,
        random_seed=887,  # Same seed
        max_workers=1,
    )

    # Validate that the outputs are identical
    assert np.array_equal(
        clusters_1["depletion_pvals"], clusters_2["depletion_pvals"]
    ), "Depletion p-values should be identical for the same random seed"
    assert np.array_equal(
        clusters_1["enrichment_pvals"], clusters_2["enrichment_pvals"]
    ), "Enrichment p-values should be identical for the same random seed"
