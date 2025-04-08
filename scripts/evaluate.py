import logging
from typing import Dict, List, Tuple
import numpy as np
from itertools import combinations
from gensim.models import Word2Vec

# Import your cluster functions from cluster_utils
from utils.cluster_utils import generate_topic_clusters, print_topic_clusters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger("full_evaluation")

def generate_dynamic_theme_keywords(
        themes: List[str],
        model: Word2Vec,
        topn: int = 50,
        similarity_threshold: float = 0.6
) -> Dict[str, List[str]]:
    """
    Dynamically generate candidate keywords for each theme by querying the model
    using the theme name. Only words exceeding the similarity threshold are returned.
    
    Parameters:
        themes: List of theme names (e.g., ["food", "geography"]).
        model: A trained Word2Vec model.
        topn: Number of similar words to retrieve.
        similarity_threshold: Minimum similarity required for inclusion.
    
    Returns:
        A dictionary mapping each theme to its dynamically generated keyword list.
    """
    dynamic_seeds = {}
    for theme in themes:
        if theme not in model.wv:
            logger.warning(f"Theme '{theme}' not in the model vocabulary. Skipping.")
            dynamic_seeds[theme] = []
            continue

        similar = model.wv.most_similar(theme, topn=topn)
        filtered = [word for word, sim in similar if sim >= similarity_threshold]
        # Optionally add the theme itself as the first keyword.
        dynamic_seeds[theme] = [theme] + filtered
        logger.info(f"Theme '{theme}' generated {len(dynamic_seeds[theme])} keywords.")
    return dynamic_seeds

def evaluate_intra_theme_similarity(
        dynamic_seeds: Dict[str, List[str]],
        model: Word2Vec,
        threshold: float = 0.65
) -> Dict[str, Dict[str, float]]:
    """
    Compute intra-theme similarity metrics (mean, standard deviation, vocabulary coverage)
    for the dynamically generated seed words.
    """
    theme_metrics = {}
    for theme, seeds in dynamic_seeds.items():
        # Only consider seeds that are in the model's vocabulary.
        in_vocab = [word for word in seeds if word in model.wv]
        pair_sims = []
        if len(in_vocab) < 2:
            logger.warning(f"Theme '{theme}' has insufficient in-vocab words. Coverage: {len(in_vocab)}/{len(seeds)}")
            theme_metrics[theme] = {
                "mean_similarity": 0.0,
                "std_similarity": 0.0,
                "coverage": round(len(in_vocab) / len(seeds) if seeds else 0.0, 4)
            }
            continue

        # Compute pairwise cosine similarities.
        for w1, w2 in combinations(in_vocab, 2):
            sim = model.wv.similarity(w1, w2)
            if sim >= threshold:
                pair_sims.append(sim)

        mean_sim = np.mean(pair_sims) if pair_sims else 0.0
        std_sim = np.std(pair_sims) if pair_sims else 0.0
        coverage = round(len(in_vocab) / len(seeds), 4)

        theme_metrics[theme] = {
            "mean_similarity": round(mean_sim, 4),
            "std_similarity": round(std_sim, 4),
            "coverage": coverage
        }
    return theme_metrics

def evaluate_similarity_density_extended(
        dynamic_seeds: Dict[str, List[str]],
        model: Word2Vec,
        topn: int = 10,
        threshold: float = 0.65
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """
    Compute a similarity density metric for each theme based on the related words
    retrieved from the model for its dynamic seeds.
    
    Returns:
        - A dictionary of average similarity density per theme.
        - A dictionary of detailed metrics, e.g., vector count.
    """
    theme_scores = {}
    detailed_metrics = {}

    for theme, seeds in dynamic_seeds.items():
        related_words = []
        for word in seeds:
            if word not in model.wv:
                logger.warning(f"Word '{word}' not in vocabulary for theme '{theme}'.")
                continue
            try:
                similars = model.wv.most_similar(word, topn=topn)
                filtered = [candidate for candidate, score in similars if score >= threshold]
                related_words.extend(filtered)
            except Exception as e:
                logger.error(f"Error retrieving similar words for '{word}': {e}")
                continue

        unique_related = list(set(related_words))
        if len(unique_related) < 2:
            theme_scores[theme] = 0.0
            detailed_metrics[theme] = {"vector_count": len(unique_related)}
            continue

        vectors = np.array([model.wv[w] for w in unique_related if w in model.wv])
        if len(vectors) < 2:
            theme_scores[theme] = 0.0
            detailed_metrics[theme] = {"vector_count": len(vectors)}
            continue

        sim_matrix = np.dot(vectors, vectors.T)
        norms = np.linalg.norm(vectors, axis=1)
        cosine_sim = sim_matrix / (norms[:, None] * norms[None, :])
        n = cosine_sim.shape[0]
        total_sim = np.sum(cosine_sim) - n  # Exclude self-similarities
        count = n * n - n
        avg_sim = total_sim / count if count > 0 else 0.0

        theme_scores[theme] = round(avg_sim, 4)
        detailed_metrics[theme] = {"vector_count": n, "avg_similarity": round(avg_sim, 4)}

    return theme_scores, detailed_metrics

def evaluate_inter_theme_similarity(
        dynamic_seeds: Dict[str, List[str]],
        model: Word2Vec
) -> Dict[Tuple[str, str], float]:
    """
    Compute the mean cosine similarity between the seed words of different themes.
    
    Returns:
        A dictionary where keys are tuples of theme pairs and values are the mean similarities.
    """
    inter_theme = {}
    theme_vectors = {}

    for theme, seeds in dynamic_seeds.items():
        vecs = [model.wv[w] for w in seeds if w in model.wv]
        if vecs:
            theme_vectors[theme] = np.array(vecs)
        else:
            logger.warning(f"No in-vocab words for theme '{theme}' for inter-theme similarity.")

    for (theme_a, vecs_a), (theme_b, vecs_b) in combinations(theme_vectors.items(), 2):
        sim_matrix = np.dot(vecs_a, vecs_b.T)
        norms_a = np.linalg.norm(vecs_a, axis=1)
        norms_b = np.linalg.norm(vecs_b, axis=1)
        cosine_sim = sim_matrix / (norms_a[:, None] * norms_b[None, :])
        inter_theme[(theme_a, theme_b)] = round(np.mean(cosine_sim), 4)

    return inter_theme

def run_full_evaluation(model: Word2Vec, themes: List[str]) -> None:
    # 1. Generate dynamic keywords for each theme.
    dynamic_seeds = generate_dynamic_theme_keywords(themes, model, topn=50, similarity_threshold=0.6)
    print("---- Dynamic Theme Seeds ----")
    for theme, seeds in dynamic_seeds.items():
        print(f"{theme.capitalize():<12} → {seeds}")

    # 2. Generate and print topic clusters.
    print("\n---- Topic Clusters ----")
    for theme, seeds in dynamic_seeds.items():
        clusters = generate_topic_clusters(seeds, model, topn=6, threshold=0.65, merge=True)
        print_topic_clusters(clusters, title=f"Category: {theme}")

    # 3. Evaluate intra-theme seed similarity.
    print("\n---- Intra-Theme Seed Similarity ----")
    intra_metrics = evaluate_intra_theme_similarity(dynamic_seeds, model, threshold=0.65)
    for theme, metrics in intra_metrics.items():
        print(f"{theme.capitalize():<12} → Mean: {metrics['mean_similarity']:.2f}, "
              f"Std Dev: {metrics['std_similarity']:.2f}, Coverage: {metrics['coverage']*100:.1f}%")

    # 4. Evaluate similarity density of related words.
    print("\n---- Similarity Density (Related Words) ----")
    density_scores, density_details = evaluate_similarity_density_extended(dynamic_seeds, model, topn=10, threshold=0.65)
    for theme, score in density_scores.items():
        status = "✅" if score > 0.5 else "⚠️"
        details = density_details.get(theme, {})
        print(f"{status} {theme.capitalize():<12} → Avg Similarity: {score:.2f} "
              f"(Vectors Used: {details.get('vector_count', 0)})")

    # 5. Evaluate inter-theme seed similarity.
    print("\n---- Inter-Theme Seed Similarity ----")
    inter_theme = evaluate_inter_theme_similarity(dynamic_seeds, model)
    for (theme_a, theme_b), sim in inter_theme.items():
        print(f"Inter-similarity between {theme_a.capitalize()} & {theme_b.capitalize()}: {sim:.2f}")