"""
risk/network/graph/_summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

from ...log import log_header, logger


class Summary:
    """
    Handles the processing, storage, and export of network analysis results.

    The Summary class provides methods to process significance and depletion data,
    compute FDR-corrected q-values, and structure information on domains and
    annotations into a DataFrame. It also offers functionality to export the
    processed data in CSV, JSON, and text formats for analysis and reporting.
    """

    _ANNOTATION_COLUMN = "Annotation"
    _DOMAIN_VALUE_COLUMNS = [
        "Domain Enrichment P-value",
        "Domain Enrichment Q-value",
        "Domain Depletion P-value",
        "Domain Depletion Q-value",
    ]
    _RAW_VALUE_COLUMNS = [
        "Raw Enrichment P-value",
        "Raw Enrichment Q-value",
        "Raw Depletion P-value",
        "Raw Depletion Q-value",
    ]
    _DOMAIN_RESULT_COLUMNS = [
        "Domain ID",
        _ANNOTATION_COLUMN,
        "Matched Members",
        "Matched Count",
        *_DOMAIN_VALUE_COLUMNS,
    ]
    _DOMAIN_FILL_DEFAULTS = {
        "Domain ID": -1,
        "Matched Members": "",
        "Matched Count": 0,
        "Domain Enrichment P-value": 1.0,
        "Domain Enrichment Q-value": 1.0,
        "Domain Depletion P-value": 1.0,
        "Domain Depletion Q-value": 1.0,
    }
    _FINAL_COLUMN_ORDER = [
        "Domain ID",
        _ANNOTATION_COLUMN,
        "Matched Members",
        "Matched Count",
        *_RAW_VALUE_COLUMNS,
        *_DOMAIN_VALUE_COLUMNS,
    ]

    def __init__(
        self,
        annotation: Dict[str, Any],
        stats_results: Dict[str, Any],
        graph,  # Avoid type hinting Graph to prevent circular imports
    ):
        """
        Initialize the Summary object with analysis components.

        Args:
            annotation (Dict[str, Any]): Annotation data, including ordered annotations and matrix of associations.
            stats_results (Dict[str, Any]): Cluster data containing p-values for significance and depletion analysis.
            graph (Graph): Graph object representing domain-to-node and node-to-label mappings.
        """
        self.annotation = annotation
        self.stats_results = stats_results
        self.graph = graph

    def to_csv(self, filepath: str) -> None:
        """
        Export significance results to a CSV file.

        Args:
            filepath (str): The path where the CSV file will be saved.
        """
        # Load results and export directly to CSV
        results = self.load()
        results.to_csv(filepath, index=False)
        logger.info(f"Analysis summary exported to CSV file: {filepath}")

    def to_json(self, filepath: str) -> None:
        """
        Export significance results to a JSON file.

        Args:
            filepath (str): The path where the JSON file will be saved.
        """
        # Load results and export directly to JSON
        results = self.load()
        results.to_json(filepath, orient="records", indent=4)
        logger.info(f"Analysis summary exported to JSON file: {filepath}")

    def to_txt(self, filepath: str) -> None:
        """
        Export significance results to a text file.

        Args:
            filepath (str): The path where the text file will be saved.
        """
        # Load results and export directly to text file
        results = self.load()
        with open(filepath, "w", encoding="utf-8") as txt_file:
            txt_file.write(results.to_string(index=False))

        logger.info(f"Analysis summary exported to text file: {filepath}")

    def load(self) -> pd.DataFrame:
        """
        Load and process domain and annotation data into a DataFrame with significance metrics.

        Returns:
            pd.DataFrame: Processed DataFrame containing:
                - Domain grouping metadata (Domain ID, matched members/counts)
                - Raw per-annotation p/q values (from full matrices, independent of domain grouping)
                - Domain-conditioned p/q values (minimum p/q within nodes belonging to each domain)
        """
        log_header("Loading analysis summary")
        (
            enrichment_pvals,
            depletion_pvals,
            enrichment_qvals,
            depletion_qvals,
        ) = self._compute_significance_matrices()
        ordered_annotation = pd.DataFrame(
            {self._ANNOTATION_COLUMN: self.annotation["ordered_annotation"]}
        )
        raw_results = self._build_raw_results(
            enrichment_pvals=enrichment_pvals,
            depletion_pvals=depletion_pvals,
            enrichment_qvals=enrichment_qvals,
            depletion_qvals=depletion_qvals,
        )
        domain_results = self._build_domain_results(
            enrichment_pvals=enrichment_pvals,
            depletion_pvals=depletion_pvals,
            enrichment_qvals=enrichment_qvals,
            depletion_qvals=depletion_qvals,
        )
        return self._merge_and_finalize_results(
            ordered_annotation=ordered_annotation,
            raw_results=raw_results,
            domain_results=domain_results,
        )

    def _compute_significance_matrices(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract p-value matrices and compute their row-wise BH-corrected q-values.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Enrichment p-values, depletion p-values, enrichment q-values,
                depletion q-values.
        """
        enrichment_pvals = self.stats_results["enrichment_pvals"]
        depletion_pvals = self.stats_results["depletion_pvals"]
        enrichment_qvals = self._calculate_qvalues(enrichment_pvals)
        depletion_qvals = self._calculate_qvalues(depletion_pvals)
        return enrichment_pvals, depletion_pvals, enrichment_qvals, depletion_qvals

    def _build_raw_results(
        self,
        enrichment_pvals: np.ndarray,
        depletion_pvals: np.ndarray,
        enrichment_qvals: np.ndarray,
        depletion_qvals: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build raw (ungrouped) per-annotation statistics from full matrices.

        Raw values are invariant to linkage/domain grouping and represent per-term
        minima across all nodes in the network.

        Args:
            enrichment_pvals (np.ndarray): Enrichment p-value matrix [node, annotation].
            depletion_pvals (np.ndarray): Depletion p-value matrix [node, annotation].
            enrichment_qvals (np.ndarray): Enrichment q-value matrix [node, annotation].
            depletion_qvals (np.ndarray): Depletion q-value matrix [node, annotation].

        Returns:
            pd.DataFrame: One row per annotation containing raw p/q summary columns.
        """
        return pd.DataFrame(
            {
                self._ANNOTATION_COLUMN: self.annotation["ordered_annotation"],
                "Raw Enrichment P-value": self._min_per_annotation(enrichment_pvals),
                "Raw Enrichment Q-value": self._min_per_annotation(enrichment_qvals),
                "Raw Depletion P-value": self._min_per_annotation(depletion_pvals),
                "Raw Depletion Q-value": self._min_per_annotation(depletion_qvals),
            }
        )

    def _build_domain_results(
        self,
        enrichment_pvals: np.ndarray,
        depletion_pvals: np.ndarray,
        enrichment_qvals: np.ndarray,
        depletion_qvals: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build domain-conditioned rows used in the summary table.

        For each annotation assigned to a domain label, domain-level p/q values are
        computed as minima over nodes that belong to that domain.

        Args:
            enrichment_pvals (np.ndarray): Enrichment p-value matrix [node, annotation].
            depletion_pvals (np.ndarray): Depletion p-value matrix [node, annotation].
            enrichment_qvals (np.ndarray): Enrichment q-value matrix [node, annotation].
            depletion_qvals (np.ndarray): Depletion q-value matrix [node, annotation].

        Returns:
            pd.DataFrame: Domain-level rows with annotation labels, matched members,
                counts, and domain-conditioned p/q values.
        """
        domain_score_column = "Summed Significance Score"
        domain_rows: List[Dict[str, Union[int, str, float]]] = [
            {
                "Domain ID": domain_id,
                self._ANNOTATION_COLUMN: description,
                domain_score_column: score,
            }
            for domain_id, info in self.graph.domain_id_to_domain_info_map.items()
            for description, score in zip(info["full_descriptions"], info["significance_scores"])
        ]
        domain_results = pd.DataFrame(
            domain_rows,
            columns=["Domain ID", self._ANNOTATION_COLUMN, domain_score_column],
        )
        if domain_results.empty:
            return pd.DataFrame(columns=self._DOMAIN_RESULT_COLUMNS)

        domain_results = domain_results.sort_values(
            by=["Domain ID", domain_score_column],
            ascending=[True, False],
        ).reset_index(drop=True)
        domain_results[self._DOMAIN_VALUE_COLUMNS] = domain_results.apply(
            lambda row: self._get_domain_significance_values(
                row["Domain ID"],
                row[self._ANNOTATION_COLUMN],
                enrichment_pvals,
                depletion_pvals,
                enrichment_qvals,
                depletion_qvals,
            ),
            axis=1,
            result_type="expand",
        )
        domain_results["Matched Members"] = domain_results[self._ANNOTATION_COLUMN].apply(
            self._get_annotation_members
        )
        domain_results["Matched Count"] = domain_results["Matched Members"].apply(
            lambda members: len(members.split(";")) if members else 0
        )
        domain_results = domain_results.drop(columns=[domain_score_column])
        return (
            domain_results[self._DOMAIN_RESULT_COLUMNS]
            .dropna(subset=["Domain Enrichment P-value", "Domain Depletion P-value"])
            .reset_index(drop=True)
        )

    def _merge_and_finalize_results(
        self,
        ordered_annotation: pd.DataFrame,
        raw_results: pd.DataFrame,
        domain_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge raw/domain tables, apply defaults, and enforce the final output schema.

        Args:
            ordered_annotation (pd.DataFrame): Canonical annotation order DataFrame.
            raw_results (pd.DataFrame): Raw per-annotation p/q results.
            domain_results (pd.DataFrame): Domain-conditioned annotation results.

        Returns:
            pd.DataFrame: Final user-facing summary table.
        """
        results = pd.merge(
            ordered_annotation,
            domain_results,
            on=self._ANNOTATION_COLUMN,
            how="left",
        ).fillna(self._DOMAIN_FILL_DEFAULTS)
        results = pd.merge(results, raw_results, on=self._ANNOTATION_COLUMN, how="left")

        results["Domain ID"] = results["Domain ID"].astype(int)
        results["Matched Count"] = results["Matched Count"].astype(int)
        return results[self._FINAL_COLUMN_ORDER]

    def _calculate_qvalues(self, pvals: np.ndarray) -> np.ndarray:
        """
        Calculate q-values (FDR) for each row of a p-value matrix.

        Args:
            pvals (np.ndarray): 2D array of p-values.

        Returns:
            np.ndarray: 2D array of q-values, with FDR correction applied row-wise.
        """
        if pvals.size == 0:
            return np.empty_like(pvals, dtype=float)
        return np.apply_along_axis(lambda row: fdrcorrection(row)[1], 1, pvals)

    def _min_per_annotation(self, values: np.ndarray) -> np.ndarray:
        """
        Compute per-annotation minimum values across nodes.

        Args:
            values (np.ndarray): 2D matrix indexed as [node, annotation].

        Returns:
            np.ndarray: 1D vector of per-annotation minima. If no rows exist, returns all ones.
        """
        if values.ndim != 2:
            return np.array([], dtype=float)
        if values.shape[0] == 0:
            return np.ones(values.shape[1], dtype=float)
        return np.min(values, axis=0)

    def _get_domain_significance_values(
        self,
        domain_id: int,
        description: str,
        enrichment_pvals: np.ndarray,
        depletion_pvals: np.ndarray,
        enrichment_qvals: np.ndarray,
        depletion_qvals: np.ndarray,
    ) -> Tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]:
        """
        Retrieve domain-conditioned p-values and q-values for a given annotation.

        Args:
            domain_id (int): The domain ID associated with the annotation.
            description (str): The annotation description.
            enrichment_pvals (np.ndarray): Matrix of significance p-values.
            depletion_pvals (np.ndarray): Matrix of depletion p-values.
            enrichment_qvals (np.ndarray): Matrix of significance q-values.
            depletion_qvals (np.ndarray): Matrix of depletion q-values.

        Returns:
            Tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]:
                Minimum enrichment p, enrichment q, depletion p, and depletion q within domain nodes.
        """
        try:
            annotation_idx = self.annotation["ordered_annotation"].index(description)
        except ValueError:
            return None, None, None, None  # Description not found

        node_indices = self.graph.domain_id_to_node_ids_map.get(domain_id, [])
        if not node_indices:
            return None, None, None, None  # No associated nodes

        sig_p = enrichment_pvals[node_indices, annotation_idx]
        dep_p = depletion_pvals[node_indices, annotation_idx]
        sig_q = enrichment_qvals[node_indices, annotation_idx]
        dep_q = depletion_qvals[node_indices, annotation_idx]

        return (
            np.min(sig_p) if sig_p.size > 0 else None,
            np.min(sig_q) if sig_q.size > 0 else None,
            np.min(dep_p) if dep_p.size > 0 else None,
            np.min(dep_q) if dep_q.size > 0 else None,
        )

    def _get_annotation_members(self, description: str) -> str:
        """
        Retrieve node labels associated with a given annotation description.

        Args:
            description (str): The annotation description.

        Returns:
            str: ';'-separated string of node labels that are associated with the annotation.
        """
        try:
            annotation_idx = self.annotation["ordered_annotation"].index(description)
        except ValueError:
            return ""  # Description not found

        # Get the column (safely) from the sparse matrix
        column = self.annotation["matrix"][:, annotation_idx]
        # Convert the column to a dense array if needed
        column = column.toarray().ravel()  # Convert to a 1D dense array
        # Get nodes present for the annotation and sort by node label - use np.where on the dense array
        nodes_present = np.where(column == 1)[0]
        node_labels = sorted(
            self.graph.node_id_to_node_label_map[node_id]
            for node_id in nodes_present
            if node_id in self.graph.node_id_to_node_label_map
        )
        return ";".join(node_labels)
