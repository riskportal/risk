"""
risk/stats/api
~~~~~~~~~~~~~~
"""

from typing import Any, Dict

from scipy.sparse import csr_matrix

from ..log import log_header, logger, params
from ._stats import (
    compute_binom_test,
    compute_chi2_test,
    compute_hypergeom_test,
    compute_permutation_test,
)


class StatsAPI:
    """
    Handles the loading of statistical results and annotation significance for clusters.

    The StatsAPI class provides methods to load cluster results from statistical tests.
    """

    def run_binom(
        self,
        annotation: Dict[str, Any],
        clusters: csr_matrix,
        null_distribution: str = "network",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute cluster significance using the binomial test.

        Args:
            annotation (Dict[str, Any]): The annotation associated with the network.
            clusters (csr_matrix): The cluster assignments for the network.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation').

        Returns:
            Dict[str, Any]: The computed significance of clusters based on the specified statistical test.
        """
        log_header("Running binomial test")
        # Compute cluster significance using the binomial test
        return self._run_statistical_test(
            annotation=annotation,
            clusters=clusters,
            null_distribution=null_distribution,
            statistical_test_key="binom",
            statistical_test_function=compute_binom_test,
            **kwargs,
        )

    def run_chi2(
        self,
        annotation: Dict[str, Any],
        clusters: csr_matrix,
        null_distribution: str = "network",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute cluster significance using the chi-squared test.

        Args:
            annotation (Dict[str, Any]): The annotation associated with the network.
            clusters (csr_matrix): The cluster assignments for the network.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

        Returns:
            Dict[str, Any]: The computed significance of clusters based on the specified statistical test.
        """
        log_header("Running chi-squared test")
        # Compute cluster significance using the chi-squared test
        return self._run_statistical_test(
            annotation=annotation,
            clusters=clusters,
            null_distribution=null_distribution,
            statistical_test_key="chi2",
            statistical_test_function=compute_chi2_test,
            **kwargs,
        )

    def run_hypergeom(
        self,
        annotation: Dict[str, Any],
        clusters: csr_matrix,
        null_distribution: str = "network",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute cluster significance using the hypergeometric test.

        Args:
            annotation (Dict[str, Any]): The annotation associated with the network.
            clusters (csr_matrix): The cluster matrix to use.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

        Returns:
            Dict[str, Any]: The computed significance of clusters based on the specified statistical test.
        """
        log_header("Running hypergeometric test")
        # Compute cluster significance using the hypergeometric test
        return self._run_statistical_test(
            annotation=annotation,
            clusters=clusters,
            null_distribution=null_distribution,
            statistical_test_key="hypergeom",
            statistical_test_function=compute_hypergeom_test,
            **kwargs,
        )

    def run_permutation(
        self,
        annotation: Dict[str, Any],
        clusters: csr_matrix,
        score_metric: str = "sum",
        null_distribution: str = "network",
        num_permutations: int = 1000,
        random_seed: int = 888,
        max_workers: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute cluster significance using the permutation test.

        Args:
            annotation (Dict[str, Any]): Annotation data, typically the output of
                `AnnotationAPI.load_annotation_*`.
            clusters (csr_matrix): Sparse cluster-by-node matrix created by a `ClusterAPI` method.
            score_metric (str, optional): Scoring metric for cluster significance. Defaults to "sum".
            null_distribution (str, optional): Null distribution ('network' or 'annotation').
                Defaults to "network".
            num_permutations (int, optional): Number of permutations for significance testing.
                Defaults to 1000.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            max_workers (int, optional): Maximum number of workers for parallel computation.
                Defaults to 1.

        Returns:
            Dict[str, Any]: Enrichment and depletion p-values for each cluster-term pair.

        Raises:
            ValueError: If `null_distribution` is not "network" or "annotation".
        """
        log_header("Running permutation test")
        # Log and display permutation test settings, which is unique to this test
        logger.debug(f"Cluster scoring metric: '{score_metric}'")
        logger.debug(f"Number of permutations: {num_permutations}")
        logger.debug(f"Maximum workers: {max_workers}")
        # Compute cluster significance using the permutation test
        return self._run_statistical_test(
            annotation=annotation,
            clusters=clusters,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="permutation",
            statistical_test_function=compute_permutation_test,
            score_metric=score_metric,
            num_permutations=num_permutations,
            max_workers=max_workers,
            **kwargs,
        )

    def _run_statistical_test(
        self,
        annotation: Dict[str, Any],
        clusters: csr_matrix,
        null_distribution: str = "network",
        statistical_test_key: str = "hypergeom",
        statistical_test_function: Any = compute_hypergeom_test,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the specified statistical test to compute cluster significance.

        Args:
            annotation (Dict[str, Any]): Annotation data associated with the network.
            clusters (csr_matrix): The cluster matrix to analyze.
            null_distribution (str, optional): The type of null distribution to use ('network' or 'annotation').
                Defaults to "network".
            statistical_test_key (str, optional): Key or name of the statistical test to be applied (e.g., "hypergeom", "binom").
                Used for logging and debugging. Defaults to "hypergeom".
            statistical_test_function (Any, optional): The function implementing the statistical test.
                It should accept clusters, annotation, null distribution, and additional kwargs.
                Defaults to `compute_hypergeom_test`.
            **kwargs: Additional parameters to be passed to the statistical test function.

        Returns:
            Dict[str, Any]: A dictionary containing the computed significance values for clusters.

        Raises:
            ValueError: If `null_distribution` is not recognised by the statistical test function.
        """
        # Log null distribution type
        logger.debug(f"Null distribution: '{null_distribution}'")
        # Log cluster analysis parameters
        params.log_stats(
            statistical_test_function=statistical_test_key,
            null_distribution=null_distribution,
            **kwargs,
        )
        # Apply statistical test function to compute cluster significance
        cluster_significance = statistical_test_function(
            clusters=clusters,
            annotation=annotation["matrix"],
            null_distribution=null_distribution,
            **kwargs,
        )
        # Return the computed cluster significance
        return cluster_significance
