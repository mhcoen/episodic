#!/usr/bin/env python3
"""
On-the-fly ambiguity detection using agglomerative clustering.

Detects when top-N retrieval candidates form multiple competitive clusters,
indicating the query may have multiple plausible interpretations.

No network calls, no polysemy lists, no LLM - purely structural analysis
of the candidate embedding space.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

# Production defaults
DEFAULT_N = 30           # Cap on candidates to consider
DEFAULT_K_MAX = 4        # Max clusters to consider (allows up to 3 options + "other")
DEFAULT_MIN_CLUSTER_SIZE = 3  # Clusters smaller than this are dropped
DEFAULT_DELTA = 0.03     # Competitiveness margin in cosine similarity units

# Stopwords for label extraction (minimal set)
STOPWORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "us", "him", "her", "them", "my", "our",
    "your", "his", "their", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also",
])


@dataclass
class Candidate:
    """A retrieval candidate with embedding and score."""
    id: int
    text: str
    emb: np.ndarray  # L2-normalized embedding
    retr_score: float  # Query-candidate similarity (higher = more relevant)
    ce_score: Optional[float] = None  # Cross-encoder score if available


@dataclass
class ClusterOption:
    """A disambiguation option derived from a cluster."""
    option_id: int
    label_terms: list[str]
    label_snippet: str
    representative_ids: list[int]
    representative_snippets: list[str]
    cluster_size: int
    best_score: float


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection."""
    ambiguous: bool
    reason: str
    options: list[ClusterOption] = field(default_factory=list)
    # Diagnostics
    chosen_k: Optional[int] = None
    cluster_sizes: list[int] = field(default_factory=list)
    best_scores: list[float] = field(default_factory=list)
    score_gap: Optional[float] = None
    delta: float = DEFAULT_DELTA


@dataclass
class AmbiguityConfig:
    """Configuration for ambiguity detection."""
    n: int = DEFAULT_N
    k_max: int = DEFAULT_K_MAX
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    delta: float = DEFAULT_DELTA


def _compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distances.

    Args:
        embeddings: (N, D) array of L2-normalized embeddings

    Returns:
        Condensed distance vector for scipy linkage
    """
    # For normalized vectors: dist = 1 - dot(e_i, e_j)
    # pdist with cosine metric does this
    return pdist(embeddings, metric='cosine')


def _extract_distinctive_terms(
    cluster_texts: list[str],
    other_texts: list[str],
    max_terms: int = 4,
) -> list[str]:
    """
    Extract terms that distinguish this cluster from others.

    Uses log-odds with add-one smoothing.
    """
    def tokenize(text: str) -> list[str]:
        # Simple word tokenization, lowercase, 3+ chars
        return [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]

    # Count tokens in cluster vs rest
    cluster_counts: dict[str, int] = {}
    other_counts: dict[str, int] = {}

    for text in cluster_texts:
        for token in tokenize(text):
            if token not in STOPWORDS:
                cluster_counts[token] = cluster_counts.get(token, 0) + 1

    for text in other_texts:
        for token in tokenize(text):
            if token not in STOPWORDS:
                other_counts[token] = other_counts.get(token, 0) + 1

    # Compute log-odds with add-one smoothing
    cluster_total = sum(cluster_counts.values()) + 1
    other_total = sum(other_counts.values()) + 1
    all_tokens = set(cluster_counts.keys()) | set(other_counts.keys())

    scores = {}
    for token in all_tokens:
        c_freq = (cluster_counts.get(token, 0) + 1) / cluster_total
        o_freq = (other_counts.get(token, 0) + 1) / other_total
        # Log-odds ratio
        scores[token] = np.log(c_freq / o_freq)

    # Sort by score, take top distinctive terms
    sorted_tokens = sorted(scores.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_tokens[:max_terms]]


def _get_first_sentence(text: str, max_chars: int = 100) -> str:
    """Extract first sentence or truncate."""
    # Find sentence boundary
    match = re.search(r'^[^.!?]*[.!?]', text)
    if match:
        sentence = match.group(0).strip()
        if len(sentence) <= max_chars:
            return sentence
    # Fallback: truncate
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def ambiguity_detect(
    query: str,
    candidates: list[Candidate],
    config: Optional[AmbiguityConfig] = None,
) -> AmbiguityResult:
    """
    Detect if retrieval candidates form ambiguous clusters.

    Args:
        query: The query string (for logging only; clustering uses embeddings)
        candidates: List of Candidate objects with embeddings and scores
        config: Configuration parameters

    Returns:
        AmbiguityResult indicating whether disambiguation is needed
    """
    if config is None:
        config = AmbiguityConfig()

    # Cap candidates and sort by retrieval score
    sorted_candidates = sorted(candidates, key=lambda c: -c.retr_score)[:config.n]
    n = len(sorted_candidates)

    logger.debug(
        f"ambiguity_detect: query='{query}', n={n}, k_max={config.k_max}, "
        f"m={config.min_cluster_size}, delta={config.delta}"
    )

    # Need at least 2*min_cluster_size candidates to have 2 valid clusters
    if n < 2 * config.min_cluster_size:
        logger.debug(f"Too few candidates ({n}) for ambiguity detection")
        return AmbiguityResult(
            ambiguous=False,
            reason=f"insufficient candidates (n={n} < 2*m={2*config.min_cluster_size})",
            delta=config.delta,
        )

    # Build embedding matrix
    embeddings = np.vstack([c.emb for c in sorted_candidates])

    # Compute condensed distance matrix
    dist_condensed = _compute_distance_matrix(embeddings)

    # Agglomerative clustering with average linkage
    Z = linkage(dist_condensed, method='average')

    # Try each k from 2 to k_max, find smallest k with 2+ competitive valid clusters
    chosen_k = None
    best_result = None

    for k in range(2, config.k_max + 1):
        # Get cluster labels for this k
        labels = fcluster(Z, t=k, criterion='maxclust')

        # Build clusters
        clusters: dict[int, list[int]] = {}  # cluster_id -> list of candidate indices
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Filter to valid clusters (size >= m)
        valid_clusters = {
            cid: indices for cid, indices in clusters.items()
            if len(indices) >= config.min_cluster_size
        }

        if len(valid_clusters) < 2:
            logger.debug(f"k={k}: only {len(valid_clusters)} valid clusters")
            continue

        # Compute best retr_score per valid cluster
        cluster_best_scores = {}
        for cid, indices in valid_clusters.items():
            scores = [sorted_candidates[i].retr_score for i in indices]
            cluster_best_scores[cid] = max(scores)

        # Sort clusters by best score
        sorted_clusters = sorted(
            valid_clusters.items(),
            key=lambda x: -cluster_best_scores[x[0]]
        )

        # Competitiveness test: (b1 - b2) <= delta
        b1 = cluster_best_scores[sorted_clusters[0][0]]
        b2 = cluster_best_scores[sorted_clusters[1][0]]
        score_gap = b1 - b2

        logger.debug(
            f"k={k}: {len(valid_clusters)} valid clusters, "
            f"b1={b1:.4f}, b2={b2:.4f}, gap={score_gap:.4f}, delta={config.delta}"
        )

        if score_gap <= config.delta:
            # Ambiguous! Build result
            chosen_k = k

            # Build options from top clusters (up to 3)
            options = []
            all_texts = [c.text for c in sorted_candidates]

            for cid, indices in sorted_clusters[:3]:
                cluster_texts = [sorted_candidates[i].text for i in indices]
                other_texts = [t for i, t in enumerate(all_texts) if i not in indices]

                # Get distinctive terms
                label_terms = _extract_distinctive_terms(cluster_texts, other_texts)

                # Get top-2 representatives by retr_score
                rep_indices = sorted(indices, key=lambda i: -sorted_candidates[i].retr_score)[:2]
                rep_ids = [sorted_candidates[i].id for i in rep_indices]
                rep_snippets = [_get_first_sentence(sorted_candidates[i].text) for i in rep_indices]

                # Label snippet from top representative
                label_snippet = rep_snippets[0] if rep_snippets else ""

                options.append(ClusterOption(
                    option_id=cid,
                    label_terms=label_terms,
                    label_snippet=label_snippet,
                    representative_ids=rep_ids,
                    representative_snippets=rep_snippets,
                    cluster_size=len(indices),
                    best_score=cluster_best_scores[cid],
                ))

            cluster_sizes = [len(indices) for _, indices in sorted_clusters]
            best_scores = [cluster_best_scores[cid] for cid, _ in sorted_clusters]

            best_result = AmbiguityResult(
                ambiguous=True,
                reason=f"found {len(valid_clusters)} competitive clusters at k={k}",
                options=options,
                chosen_k=chosen_k,
                cluster_sizes=cluster_sizes,
                best_scores=best_scores,
                score_gap=score_gap,
                delta=config.delta,
            )

            logger.info(
                f"Ambiguity detected: query='{query}', k={chosen_k}, "
                f"clusters={cluster_sizes}, gap={score_gap:.4f}"
            )
            break

    if best_result:
        return best_result

    # No ambiguity found
    logger.debug(f"No ambiguity detected for query='{query}'")
    return AmbiguityResult(
        ambiguous=False,
        reason="single coherent neighborhood (no competitive clusters found)",
        delta=config.delta,
    )


def format_disambiguation_prompt(query: str, result: AmbiguityResult) -> str:
    """Format a user-facing disambiguation prompt."""
    if not result.ambiguous:
        return ""

    lines = [f"I found multiple plausible topics for '{query}'. Which did you mean?"]
    lines.append("")

    for i, opt in enumerate(result.options, 1):
        # Use label terms if available, else snippet
        if opt.label_terms:
            label = ", ".join(opt.label_terms[:3])
        else:
            label = opt.label_snippet

        lines.append(f"  {i}. {label}")
        if opt.representative_snippets:
            lines.append(f"     e.g., \"{opt.representative_snippets[0]}\"")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    logging.basicConfig(level=logging.DEBUG)

    np.random.seed(42)

    # Create two well-separated clusters
    centroid_a = np.array([1.0, 0.0, 0.0])
    centroid_b = np.array([0.0, 1.0, 0.0])

    candidates = []

    # Cluster A: 5 points
    for i in range(5):
        noise = np.random.randn(3) * 0.1
        emb = centroid_a + noise
        emb = emb / np.linalg.norm(emb)  # L2 normalize
        candidates.append(Candidate(
            id=i,
            text=f"Cluster A item {i}: programming language syntax",
            emb=emb,
            retr_score=0.9 - i * 0.02,  # Scores: 0.9, 0.88, 0.86, 0.84, 0.82
        ))

    # Cluster B: 5 points
    for i in range(5):
        noise = np.random.randn(3) * 0.1
        emb = centroid_b + noise
        emb = emb / np.linalg.norm(emb)
        candidates.append(Candidate(
            id=i + 5,
            text=f"Cluster B item {i}: coffee brewing methods",
            emb=emb,
            retr_score=0.88 - i * 0.02,  # Scores: 0.88, 0.86, 0.84, 0.82, 0.80
        ))

    result = ambiguity_detect("java", candidates)
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULT")
    print("=" * 60)
    print(f"Ambiguous: {result.ambiguous}")
    print(f"Reason: {result.reason}")
    print(f"Chosen k: {result.chosen_k}")
    print(f"Cluster sizes: {result.cluster_sizes}")
    print(f"Best scores: {result.best_scores}")
    print(f"Score gap: {result.score_gap}")
    print()
    print(format_disambiguation_prompt("java", result))
