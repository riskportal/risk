"""
tests/test_load_io_combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "network_fixture",
    [
        "cytoscape_network",
        "cytoscape_json_network",
        "gpickle_network",
        "networkx_network",
    ],
)
@pytest.mark.parametrize(
    "annotation_fixture",
    [
        "json_annotation",
        "csv_annotation",
        "tsv_annotation",
        "excel_annotation",
    ],
)
def test_load_graphs(request, risk_obj, network_fixture, annotation_fixture):
    """
    Test loading all possible combinations of networks and annotation followed by graph loading.

    Args:
        request: The pytest request object to access fixture values.
        risk_obj: The RISK object instance used for loading the network and annotation.
        network_fixture: The name of the fixture to load the network.
        annotation_fixture: The name of the fixture to load the annotation.
    """
    # Load the network using the specified fixture
    network = request.getfixturevalue(network_fixture)
    # Load the annotation using the specified fixture
    annotation = request.getfixturevalue(annotation_fixture)
    # Cluster the network using the Leiden algorithm
    clusters = risk_obj.cluster_leiden(
        network=network,
        fraction_shortest_edges=0.75,
        resolution=1.0,
        random_seed=887,
    )
    stats_results = risk_obj.run_permutation(
        annotation=annotation,
        clusters=clusters,
        null_distribution="network",
        score_metric="stdev",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    # Validate clusters structure
    assert "depletion_pvals" in stats_results, "Clusters should contain a 'depletion_pvals' key"
    assert "enrichment_pvals" in stats_results, "Clusters should contain an 'enrichment_pvals' key"
    assert isinstance(
        stats_results["depletion_pvals"], np.ndarray
    ), "'depletion_pvals' should be a numpy array"
    assert isinstance(
        stats_results["enrichment_pvals"], np.ndarray
    ), "'enrichment_pvals' should be a numpy array"
    assert (
        stats_results["depletion_pvals"].shape == stats_results["enrichment_pvals"].shape
    ), "'depletion_pvals' and 'enrichment_pvals' should have the same shape"
    # Ensure that the p-value matrices are not empty
    assert stats_results["depletion_pvals"].size > 0, "'depletion_pvals' array is empty"
    assert stats_results["enrichment_pvals"].size > 0, "'enrichment_pvals' array is empty"

    graph = risk_obj.load_graph(
        network=network,
        annotation=annotation,
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

    # Validate the graph
    assert graph is not None, "Graph should not be None"
    assert len(graph.network.nodes) > 0, "Graph should have nodes"
    assert len(graph.network.edges) > 0, "Graph should have edges"
