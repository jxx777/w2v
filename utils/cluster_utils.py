from typing import Dict, List, Tuple, Optional
import logging

from unidecode import unidecode

logger = logging.getLogger("cluster_utils")
logger.setLevel("INFO")


def is_morph_variant(seed: str, candidate: str) -> bool:
    """
    Check if the candidate is a morphological variant of the seed.
    """
    seed_norm = unidecode(seed.lower())
    candidate_norm = unidecode(candidate.lower())
    return seed_norm in candidate_norm or candidate_norm in seed_norm


def generate_topic_clusters(
        seed_words: List[str],
        model,
        topn: int = 10,
        threshold: float = 0.6,
        merge: bool = False
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Generate topic clusters for a list of seed words by retrieving their most similar words.

    Parameters:
        seed_words (List[str]): The list of seed words to start the cluster generation.
        model: The trained Word2Vec model.
        topn (int): Number of similar words to retrieve.
        threshold (float): Minimum similarity score for inclusion.
        merge (bool): If True, merge clusters with overlapping similar words.

    Returns:
        Dict[str, List[Tuple[str, float]]]: A dictionary mapping a seed word (or merged seed key)
            to its cluster of similar words and their similarity scores.
    """
    clusters = {}
    for word in seed_words:
        if word not in model.wv:
            logger.warning(f"Seed word '{word}' not found in vocabulary.")
            clusters[word] = [("not in vocab", 0.0)]
            continue

        similar_words = []
        try:
            for candidate, score in model.wv.most_similar(word, topn=topn):
                if score >= threshold and not is_morph_variant(word, candidate):
                    similar_words.append((candidate, score))
        except KeyError as e:
            logger.error(f"Error retrieving similar words for '{word}': {e}")
            similar_words.append(("error", 0.0))
        clusters[word] = similar_words

    if merge:
        clusters = merge_overlapping_clusters(clusters)

    return clusters


def merge_overlapping_clusters(
        clusters: Dict[str, List[Tuple[str, float]]],
        overlap_threshold: int = 1
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Merge clusters that share overlapping words into single clusters.

    Parameters:
        clusters (Dict[str, List[Tuple[str, float]]]): Original clusters mapping seed words to similar words.
        overlap_threshold (int): Minimum number of common words needed to merge clusters.

    Returns:
        Dict[str, List[Tuple[str, float]]]: Merged cluster mapping.
    """
    merged = {}
    cluster_keys = list(clusters.keys())
    visited = set()

    for i, key_i in enumerate(cluster_keys):
        if key_i in visited:
            continue
        # Start a new merged cluster with the current key
        merged_key = {key_i}
        merged_words = {word: score for word, score in clusters[key_i] if word != "not in vocab"}

        for j in range(i + 1, len(cluster_keys)):
            key_j = cluster_keys[j]
            if key_j in visited:
                continue
            # Count the overlap in words (ignoring scores) between clusters
            words_j = {word for word, _ in clusters[key_j] if word != "not in vocab"}
            common = set(merged_words.keys()).intersection(words_j)
            if len(common) >= overlap_threshold:
                # Merge clusters: update the seed names and combine word lists.
                merged_key.add(key_j)
                for word, score in clusters[key_j]:
                    if word == "not in vocab":
                        continue
                    # If word appears in both, you could average the score; here we take the max.
                    merged_words[word] = max(merged_words.get(word, 0.0), score)
                visited.add(key_j)

        visited.add(key_i)
        # Create a merged key (e.g., joined seed words sorted alphabetically).
        merged_cluster_name = " / ".join(sorted(merged_key))
        # Sort words by score descending for better readability.
        sorted_words = sorted(merged_words.items(), key=lambda x: x[1], reverse=True)
        merged[merged_cluster_name] = sorted_words

    return merged


def print_topic_clusters(
        clusters: Dict[str, List[Tuple[str, float]]],
        title: Optional[str] = None,
        max_per_row: int = 6
) -> None:
    """
    Print the topic clusters in a formatted table-like output.

    Parameters:
        clusters (Dict[str, List[Tuple[str, float]]]): Clusters to print.
        title (Optional[str]): Optional title for the cluster display.
        max_per_row (int): Maximum number of cluster items per row.
    """
    if title:
        header = f"\n{title}\n" + "=" * (len(title) + 4)
        print(header)

    for seed, similar in clusters.items():
        if not similar or (len(similar) == 1 and similar[0][0] in {"not in vocab", "error"}):
            print(f"{seed:<20} → [No valid words found]")
            continue

        # Format the output with seed name and chunks of similar words
        output_line = f"{seed:<20} → "
        formatted_words = [f"{word} ({score:.2f})" for word, score in similar]
        for i in range(0, len(formatted_words), max_per_row):
            chunk = ", ".join(formatted_words[i:i + max_per_row])
            if i == 0:
                print(output_line + chunk)
            else:
                print(" " * 24 + chunk)
