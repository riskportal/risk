"""
risk/annotation/annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import re
from collections import Counter
from itertools import compress
from typing import Any, Dict, List, Set, Union

import networkx as nx
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.sparse import coo_matrix

from ..log import logger
from ._nltk_setup import setup_nltk_resources


# Annotation boilerplate that carries no meaning in a domain label
BOILERPLATE_WORDS = {"via", "involved", "process"}
# Number of words in a description of specific words
SPECIFIC_LABEL_WORDS = 6


def initialize_nltk():
    """Initialize all required NLTK components."""
    setup_nltk_resources()

    # After resources are available, initialize the components
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    global STOP_WORDS, LEMMATIZER
    STOP_WORDS = set(stopwords.words("english")) | BOILERPLATE_WORDS
    LEMMATIZER = WordNetLemmatizer()


# Initialize NLTK components
initialize_nltk()


def load_annotation(
    network: nx.Graph,
    annotation_input: Dict[str, Any],
    min_nodes_per_term: int = 1,
    max_nodes_per_term: int = 10_000,
) -> Dict[str, Any]:
    """
    Convert annotation input to a sparse matrix and reindex based on the network's node labels.

    Args:
        network (nx.Graph): The network graph.
        annotation_input (Dict[str, Any]): An annotation dictionary.
        min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
            term to be included. Defaults to 1.
        max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
            term. Defaults to 10_000.

    Returns:
        Dict[str, Any]: A dictionary containing ordered nodes, ordered annotations, and the sparse binary annotations
            matrix.

    Raises:
        ValueError: If no annotation is found for the nodes in the network.
    """
    # Step 1: Map nodes and annotations to indices
    node_label_order = [attr["label"] for _, attr in network.nodes(data=True) if "label" in attr]
    node_to_idx = {node: i for i, node in enumerate(node_label_order)}
    annotation_to_idx = {annotation: i for i, annotation in enumerate(annotation_input)}
    # Step 2: Construct a sparse binary matrix directly
    row = []
    col = []
    data = []
    for annotation, nodes in annotation_input.items():
        for node in nodes:
            if node in node_to_idx and annotation in annotation_to_idx:
                row.append(node_to_idx[node])
                col.append(annotation_to_idx[annotation])
                data.append(1)

    # Create a sparse binary matrix
    num_nodes = len(node_to_idx)
    num_annotation = len(annotation_to_idx)
    # Convert to a sparse matrix and set the data type to uint8 for binary representation
    annotation_pivot = (
        coo_matrix((data, (row, col)), shape=(num_nodes, num_annotation)).tocsr().astype(np.uint8)
    )
    # Step 3: Filter out annotations with too few or too many nodes
    valid_annotation = np.array(
        [
            annotation_pivot[:, i].sum() >= min_nodes_per_term
            and annotation_pivot[:, i].sum() <= max_nodes_per_term
            for i in range(num_annotation)
        ]
    )
    annotation_pivot = annotation_pivot[:, valid_annotation]
    # Step 4: Raise errors for empty matrices
    if annotation_pivot.nnz == 0:
        raise ValueError("No terms found in the annotation file for the nodes in the network.")

    num_remaining_annotation = annotation_pivot.shape[1]
    if num_remaining_annotation == 0:
        raise ValueError(
            f"No annotation terms found with at least {min_nodes_per_term} nodes and at most {max_nodes_per_term} nodes."
        )

    # Step 5: Extract ordered nodes and annotations
    ordered_nodes = tuple(node_label_order)
    ordered_annotation = tuple(
        annotation for annotation, is_valid in zip(annotation_to_idx, valid_annotation) if is_valid
    )

    # Log the filtering details
    logger.info(f"Minimum number of nodes per annotation term: {min_nodes_per_term}")
    logger.info(f"Maximum number of nodes per annotation term: {max_nodes_per_term}")
    logger.info(f"Number of input annotation terms: {num_annotation}")
    logger.info(f"Number of kept annotation terms: {num_remaining_annotation}")

    return {
        "ordered_nodes": ordered_nodes,
        "ordered_annotation": ordered_annotation,
        "matrix": annotation_pivot,
    }


def define_top_annotation(
    network: nx.Graph,
    ordered_annotation_labels: List[str],
    cluster_significance_sums: List[int],
    significant_significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    min_cluster_size: int = 5,
    max_cluster_size: int = 1000,
) -> pd.DataFrame:
    """
    Define top annotations based on cluster significance sums and binary significance matrix.

    Args:
        network (NetworkX graph): The network graph.
        ordered_annotation_labels (list of str): List of ordered annotation labels.
        cluster_significance_sums (list of int): List of cluster significance sums.
        significant_significance_matrix (np.ndarray): Enrichment matrix below alpha threshold.
        significant_binary_significance_matrix (np.ndarray): Binary significance matrix below alpha threshold.
        min_cluster_size (int, optional): Minimum cluster size. Defaults to 5.
        max_cluster_size (int, optional): Maximum cluster size. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame with top annotations and their properties.
    """
    # Sum the columns of the significant significance matrix (positive floating point values)
    significant_significance_scores = significant_significance_matrix.sum(axis=0)
    # Create DataFrame to store annotations, their cluster significance sums, and significance scores
    annotation_significance_matrix = pd.DataFrame(
        {
            "id": range(len(ordered_annotation_labels)),
            "full_terms": ordered_annotation_labels,
            "significant_cluster_significance_sums": cluster_significance_sums,
            "significant_significance_score": significant_significance_scores,
        }
    )
    annotation_significance_matrix["significant_annotation"] = False
    # Apply size constraints to identify potential significant annotations
    annotation_significance_matrix.loc[
        (
            annotation_significance_matrix["significant_cluster_significance_sums"]
            >= min_cluster_size
        )
        & (
            annotation_significance_matrix["significant_cluster_significance_sums"]
            <= max_cluster_size
        ),
        "significant_annotation",
    ] = True
    # Initialize columns for connected components analysis
    annotation_significance_matrix["num_connected_components"] = 0
    annotation_significance_matrix["size_connected_components"] = None
    annotation_significance_matrix["size_connected_components"] = annotation_significance_matrix[
        "size_connected_components"
    ].astype(object)
    annotation_significance_matrix["num_large_connected_components"] = 0

    for attribute in annotation_significance_matrix.index.values[
        annotation_significance_matrix["significant_annotation"]
    ]:
        # Identify significant clusters based on the binary significance matrix
        significant_clusters = list(
            compress(list(network), significant_binary_significance_matrix[:, attribute])
        )
        significant_network = nx.subgraph(network, significant_clusters)
        # Analyze connected components within the significant subnetwork
        connected_components = sorted(
            nx.connected_components(significant_network), key=len, reverse=True
        )
        size_connected_components = np.array([len(c) for c in connected_components])

        # Filter the size of connected components by min_cluster_size and max_cluster_size
        filtered_size_connected_components = size_connected_components[
            (size_connected_components >= min_cluster_size)
            & (size_connected_components <= max_cluster_size)
        ]
        # Calculate the number of connected components and large connected components
        num_connected_components = len(connected_components)
        num_large_connected_components = len(filtered_size_connected_components)

        # Assign the number of connected components
        annotation_significance_matrix.loc[attribute, "num_connected_components"] = (
            num_connected_components
        )
        # Filter out attributes with more than one connected component
        annotation_significance_matrix.loc[
            annotation_significance_matrix["num_connected_components"] > 1,
            "significant_annotation",
        ] = False
        # Assign the number of large connected components
        annotation_significance_matrix.loc[attribute, "num_large_connected_components"] = (
            num_large_connected_components
        )
        # Assign the size of connected components, ensuring it is always a list
        annotation_significance_matrix.at[attribute, "size_connected_components"] = (
            filtered_size_connected_components.tolist()
        )

    return annotation_significance_matrix


def get_weighted_description(
    words_column: pd.Series,
    scores_column: pd.Series,
    word_rarity: Union[Dict[str, float], None] = None,
) -> str:
    """
    Generate a weighted description from words and their corresponding scores,
    using improved weighting logic with normalization, lemmatization, and aggregation.

    Args:
        words_column (pd.Series): A pandas Series containing strings (phrases) to process.
        scores_column (pd.Series): A pandas Series containing significance scores to weigh the terms.
        word_rarity (Dict[str, float], optional): Rarity of each word across the annotation terms,
            from `get_word_rarity`. When provided, words shared by many terms weigh less than words
            specific to a few. Defaults to None (words are not weighed by rarity).

    Returns:
        str: A coherent description formed from the most frequent and significant words.
    """
    weighted_counts = _get_weighted_counts(words_column, scores_column)
    # Describe the phrases by their specific words, if any word is specific
    if word_rarity:
        specific_description = _get_specific_description(
            words_column, scores_column, weighted_counts, word_rarity
        )
        if specific_description:
            return specific_description

    # Reconstruct a weighted token list by repeating each token by its aggregated count.
    weighted_words = []
    for token, count in weighted_counts.items():
        weighted_words.extend([token] * count)

    # Combine tokens that match number-word patterns (e.g. "4-alpha"), but do not remove numeric tokens.
    # All tokens are included in the final list.
    combined_tokens = []
    for token in weighted_words:
        combined_tokens.append(token)

    # Simplify the token list to remove near-duplicates based on the Jaccard index.
    simplified_words = _simplify_word_list(combined_tokens)
    # Generate a coherent description from the simplified words.
    description = _generate_coherent_description(simplified_words)

    return description


def get_word_rarity(terms: List[str], min_terms: int = 30) -> Dict[str, float]:
    """
    Score how rare each word is across an annotation's terms, so that words shared by many terms
    (e.g., "regulation") weigh less than words specific to a few.

    Args:
        terms (List[str]): The annotation terms, after the node-count filters.
        min_terms (int, optional): Minimum number of terms needed for rarity to be meaningful.
            Defaults to 30.

    Returns:
        Dict[str, float]: A dictionary mapping each word to its rarity, ln(terms / (1 + terms with
            the word)) squared, floored at zero. Empty if there are fewer than `min_terms` terms.
    """
    num_terms = len(terms)
    if num_terms < min_terms:
        return {}

    # Count the terms that contain each word
    term_counts = Counter(word for term in terms for word in set(_tokenize_term(term)))
    return {
        word: max(0.0, float(np.log(num_terms / (1 + term_count)))) ** 2
        for word, term_count in term_counts.items()
    }


def _get_specific_description(
    words_column: pd.Series,
    scores_column: pd.Series,
    weighted_counts: Dict[str, int],
    word_rarity: Dict[str, float],
) -> str:
    """
    Generate a description from the words that are specific to the given phrases, ordered as they
    occur in the phrase that contains the most of them.

    Args:
        words_column (pd.Series): A pandas Series containing strings (phrases) to process.
        scores_column (pd.Series): A pandas Series containing significance scores to weigh the terms.
        weighted_counts (Dict[str, int]): The aggregated weight of each token across the phrases.
        word_rarity (Dict[str, float]): Rarity of each word, from `get_word_rarity`.

    Returns:
        str: The most specific words, or an empty string if no word is specific.
    """
    # Weigh each word by its rarity, dropping words that are shared by most terms
    specific_counts = {
        word: count * word_rarity.get(word, 0.0)
        for word, count in weighted_counts.items()
        if count * word_rarity.get(word, 0.0) > 0
    }
    if not specific_counts:
        return ""

    # Keep the heaviest words, one per group of near-duplicates, listed by weight
    specific_words = _select_representative_words(specific_counts)[:SPECIFIC_LABEL_WORDS]
    # Find the phrase that contains the most of them, preferring the higher score, then the earlier phrase
    phrase_tokens = [_tokenize_term(phrase) for phrase in words_column]
    scores = scores_column.fillna(float("-inf")).tolist()
    best_index = max(
        range(len(phrase_tokens)),
        key=lambda i: (len(set(specific_words) & set(phrase_tokens[i])), scores[i], -i),
    )
    # Write the words as they occur in that phrase, then the rest by weight
    ordered_words = [
        word for word in dict.fromkeys(phrase_tokens[best_index]) if word in specific_words
    ]
    ordered_words += [word for word in specific_words if word not in ordered_words]

    return " ".join(ordered_words)


def _tokenize_term(term: str) -> List[str]:
    """
    Split an annotation term into its cleaned, lemmatized tokens, without stopwords.

    Args:
        term (str): The annotation term to tokenize.

    Returns:
        List[str]: The tokens in the order they occur in the term.
    """
    tokens = []
    for token in word_tokenize(str(term)):
        # Clean token: remove extraneous punctuation (but preserve intra-word hyphens)
        token_clean = re.sub(r"[^\w\-]", "", token).strip()
        if not token_clean:
            continue
        # Skip stopwords, matching case-insensitively so capitalized terms are filtered too
        if token_clean.lower() in STOP_WORDS:
            continue
        # Lemmatize the token to merge similar forms
        tokens.append(LEMMATIZER.lemmatize(token_clean))

    return tokens


def _get_weighted_counts(words_column: pd.Series, scores_column: pd.Series) -> Dict[str, int]:
    """
    Accumulate the significance-weighted count of each token across the phrases.

    Args:
        words_column (pd.Series): A pandas Series containing strings (phrases) to process.
        scores_column (pd.Series): A pandas Series containing significance scores to weigh the terms.

    Returns:
        Dict[str, int]: A dictionary mapping each token to its aggregated weight.
    """
    # Normalize significance scores to [0,1]. If all scores are identical, use 1.
    if scores_column.max() == scores_column.min():
        normalized_scores = pd.Series([1] * len(scores_column), index=scores_column.index)
    else:
        normalized_scores = (scores_column - scores_column.min()) / (
            scores_column.max() - scores_column.min()
        )

    # Accumulate weighted counts for each token (after cleaning and lemmatization)
    weighted_counts = {}
    for phrase, score in zip(words_column, normalized_scores):
        # Determine the weight (scale factor; here multiplying normalized score by 10)
        weight = max(1, int((0 if pd.isna(score) else score) * 10))
        for token_norm in _tokenize_term(phrase):
            # Apply weighting boost for biologically structured number-word hybrids
            if re.match(r"^\d+[\-\w]+", token_norm):
                actual_weight = int(weight * 1.5)
            else:
                actual_weight = weight
            weighted_counts[token_norm] = weighted_counts.get(token_norm, 0) + actual_weight

    return weighted_counts


def _select_representative_words(
    word_counts: Dict[str, float], threshold: float = 0.80
) -> List[str]:
    """
    Filter out words that are too similar based on the Jaccard index,
    keeping the word with the higher aggregated count.

    Args:
        word_counts (Dict[str, float]): A dictionary mapping each word to its aggregated count.
        threshold (float, optional): The similarity threshold for the Jaccard index. Defaults to 0.80.

    Returns:
        List[str]: The kept words, ordered by descending count.
    """
    filtered_words = []
    used_words = set()

    # Iterate through words sorted by descending weighted frequency
    for word in sorted(word_counts, key=lambda w: word_counts[w], reverse=True):
        if word in used_words:
            continue

        word_set = set(word)
        # Find similar words (including the current word) based on the Jaccard index
        similar_words = [
            other_word
            for other_word in word_counts
            if _calculate_jaccard_index(word_set, set(other_word)) >= threshold
        ]
        # Choose the word with the highest weighted count among the similar group
        similar_words.sort(key=lambda w: word_counts[w], reverse=True)
        best_word = similar_words[0]
        filtered_words.append(best_word)
        used_words.update(similar_words)

    return filtered_words


def _simplify_word_list(words: List[str], threshold: float = 0.80) -> List[str]:
    """
    Filter out words that are too similar based on the Jaccard index,
    keeping the word with the higher aggregated count.

    Args:
        words (List[str]): The list of tokens to be filtered.
        threshold (float, optional): The similarity threshold for the Jaccard index. Defaults to 0.80.

    Returns:
        List[str]: A list of filtered words, where similar words are reduced to the most frequent one.
    """
    # Count the occurrences (which reflect the weighted importance)
    filtered_words = _select_representative_words(Counter(words), threshold)
    # Preserve the original order (by frequency) from the filtered set
    final_words = [word for word in words if word in filtered_words]

    return final_words


def _calculate_jaccard_index(set1: Set[Any], set2: Set[Any]) -> float:
    """
    Calculate the Jaccard index between two sets.

    Args:
        set1 (Set[Any]): The first set.
        set2 (Set[Any]): The second set.

    Returns:
        float: The Jaccard index (intersection over union). Returns 0 if the union is empty.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0


def _generate_coherent_description(words: List[str]) -> str:
    """
    Generate a coherent description from a list of words. If there is only one unique entry,
    return it directly; otherwise, order the words by frequency and join them into a single
    string.

    Args:
        words (List[str]): A list of tokens.

    Returns:
        str: A coherent, space-separated description.
    """
    if not words:
        return "N/A"

    # If there is only one unique word, return it directly
    unique_words = set(words)
    if len(unique_words) == 1:
        return list(unique_words)[0]

    # Count weighted occurrences and sort in descending order.
    word_counts = Counter(words)
    most_common_words = [word for word, _ in word_counts.most_common()]
    description = " ".join(most_common_words)

    return description
