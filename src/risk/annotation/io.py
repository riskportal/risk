"""
risk/annotation/io
~~~~~~~~~~~~~~~~~~
"""

import json
from typing import Any, Dict

import networkx as nx
import pandas as pd

from ..log import log_header, logger, params
from .annotation import load_annotation


class AnnotationAPI:
    """
    Handles the loading and exporting of annotation in various file formats.

    The AnnotationAPI class provides methods to load annotation from different file types (JSON, CSV, Excel, etc.)
    and to export parameter data to various formats like JSON, CSV, and text files.
    """

    def load_annotation_json(
        self,
        network: nx.Graph,
        filepath: str,
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a JSON file and return the standardized annotation mapping.

        Args:
            network (nx.Graph): Graph whose node labels define valid annotation members.
            filepath (str): Path to the JSON annotation file.
            min_nodes_per_term (int, optional): Minimum number of network nodes required for each
                annotation term. Defaults to 1.
            max_nodes_per_term (int, optional): Maximum number of network nodes allowed for each
                annotation term. Defaults to 10_000.

        Returns:
            Dict[str, Any]: Mapping with `ordered_nodes`, `ordered_annotation`, and a sparse matrix.

        Notes:
            Annotation members must match the `label` attribute assigned to each node in `network`.
            If no matching label is found, members will be searched in the node IDs.
        """
        filetype = "JSON"
        # Log the loading of the JSON file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            annotation_input = json.load(file)

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_excel(
        self,
        network: nx.Graph,
        filepath: str,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        sheet_name: str = "Sheet1",
        nodes_delimiter: str = ";",
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from an Excel file and return the standardized annotation mapping.

        Args:
            network (nx.Graph): The NetworkX graph to which the annotation is related.
            filepath (str): Path to the Excel annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            sheet_name (str, optional): The name of the Excel sheet to load (default is 'Sheet1').
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary with keys 'ordered_nodes', 'ordered_annotation', and 'matrix'.
        """
        filetype = "Excel"
        # Log the loading of the Excel file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        annotation = pd.read_excel(filepath, sheet_name=sheet_name)
        # Normalise delimited node strings (e.g. "gene1;gene2") before matrix construction.
        annotation[nodes_colname] = annotation[nodes_colname].apply(
            lambda x: x.split(nodes_delimiter)
        )
        annotation_input = annotation.set_index(label_colname)[nodes_colname].to_dict()

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_csv(
        self,
        network: nx.Graph,
        filepath: str,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        nodes_delimiter: str = ";",
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a CSV file and return the standardized annotation mapping.

        Args:
            network (nx.Graph): The NetworkX graph to which the annotation is related.
            filepath (str): Path to the CSV annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary with keys 'ordered_nodes', 'ordered_annotation', and 'matrix'.
        """
        filetype = "CSV"
        # Log the loading of the CSV file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        annotation_input = self._load_matrix_file(
            filepath, label_colname, nodes_colname, delimiter=",", nodes_delimiter=nodes_delimiter
        )

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_tsv(
        self,
        network: nx.Graph,
        filepath: str,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        nodes_delimiter: str = ";",
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a TSV file and return the standardized annotation mapping.

        Args:
            network (nx.Graph): The NetworkX graph to which the annotation is related.
            filepath (str): Path to the TSV annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary with keys 'ordered_nodes', 'ordered_annotation', and 'matrix'.
        """
        filetype = "TSV"
        # Log the loading of the TSV file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        annotation_input = self._load_matrix_file(
            filepath, label_colname, nodes_colname, delimiter="\t", nodes_delimiter=nodes_delimiter
        )

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_dict(
        self,
        network: nx.Graph,
        content: Dict[str, Any],
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from an in-memory dictionary and return the standardized annotation mapping.

        Args:
            network (NetworkX graph): The network to which the annotation is related.
            content (Dict[str, Any]): The annotation dictionary to load.
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary with keys 'ordered_nodes', 'ordered_annotation', and 'matrix'.

        Raises:
            TypeError: If the content is not a dictionary.
        """
        if not isinstance(content, dict):
            raise TypeError(
                f"Expected 'content' to be a dictionary, but got {type(content).__name__} instead."
            )

        filetype = "Dictionary"
        # Capture that the annotation originated in-memory for reproducible params dumps.
        params.log_annotation(
            filepath="In-memory dictionary",
            filetype=filetype,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, "In-memory dictionary")

        return load_annotation(network, content, min_nodes_per_term, max_nodes_per_term)

    def _load_matrix_file(
        self,
        filepath: str,
        label_colname: str,
        nodes_colname: str,
        delimiter: str = ",",
        nodes_delimiter: str = ";",
    ) -> Dict[str, Any]:
        """
        Load annotation from a CSV or TSV file and convert them to a dictionary.

        Args:
            filepath (str): Path to the annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            delimiter (str, optional): Delimiter used to separate columns in the file (default is ',').
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').

        Returns:
            Dict[str, Any]: A dictionary where each label is paired with its respective list of nodes.
        """
        annotation = pd.read_csv(filepath, delimiter=delimiter)
        # Split multi-valued cells into lists so sparse matrix conversion sees iterable members.
        annotation[nodes_colname] = annotation[nodes_colname].apply(
            lambda x: x.split(nodes_delimiter)
        )
        label_node_dict = annotation.set_index(label_colname)[nodes_colname].to_dict()
        return label_node_dict

    def _log_loading_annotation(self, filetype: str, filepath: str = "") -> None:
        """
        Log the loading of annotation files.

        Args:
            filetype (str): The type of the file being loaded (e.g., 'Cytoscape').
            filepath (str, optional): The path to the file being loaded.
        """
        log_header("Loading annotation")
        logger.debug(f"Filetype: {filetype}")
        if filepath:
            logger.debug(f"Filepath: {filepath}")
