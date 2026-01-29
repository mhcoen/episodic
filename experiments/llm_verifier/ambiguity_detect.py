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

# Competitiveness: rank-gap based (score-scale invariant)
# Scales with N: ceil(0.1 * N) clamped to [2, 5]
DEFAULT_RANK_GAP_FRACTION = 0.1  # Fraction of N for rank gap
DEFAULT_RANK_GAP_MIN = 2
DEFAULT_RANK_GAP_MAX = 5

# Cohesion: relative to overall distance (prevents chain-connected garbage partitions)
# Uses max intra-cluster distance (diameter)
DEFAULT_COHESION_RATIO = 1.5  # Max cluster diameter / overall mean distance

# Separation: clusters must be well-separated (prevents chain splits)
# Uses MIN inter-cluster distance (strongest check for chain detection)
DEFAULT_SEPARATION_RATIO = 1.0  # Min inter-cluster distance / overall mean distance

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
    n_candidates: int = 0
    chosen_k: Optional[int] = None
    cluster_sizes: list[int] = field(default_factory=list)
    best_scores: list[float] = field(default_factory=list)
    best_ranks: list[int] = field(default_factory=list)  # Ranks of best items per cluster
    rank_gap: Optional[int] = None  # ρ2 - ρ1
    score_gap: Optional[float] = None  # b1 - b2 (for logging/calibration)
    cohesion_ratios: list[float] = field(default_factory=list)  # Per-cluster max intra dist / mean_dist
    separation_ratio: Optional[float] = None  # Min inter-cluster dist / mean_dist
    overall_mean_dist: Optional[float] = None  # For calibration logging
    # Config used (effective values)
    max_rank_gap: int = 3  # Computed from N
    max_cohesion_ratio: float = DEFAULT_COHESION_RATIO
    min_separation_ratio: float = DEFAULT_SEPARATION_RATIO


@dataclass
class AmbiguityConfig:
    """Configuration for ambiguity detection."""
    n: int = DEFAULT_N
    k_max: int = DEFAULT_K_MAX
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    # Competitiveness (rank-gap based, score-scale invariant)
    # If None, computed as ceil(0.1 * n) clamped to [2, 5]
    rank_gap: Optional[int] = None
    rank_gap_fraction: float = DEFAULT_RANK_GAP_FRACTION
    rank_gap_min: int = DEFAULT_RANK_GAP_MIN
    rank_gap_max: int = DEFAULT_RANK_GAP_MAX
    # Cohesion: max intra-cluster distance (diameter) / mean_dist
    cohesion_ratio: float = DEFAULT_COHESION_RATIO
    # Separation: min inter-cluster distance / mean_dist (strongest chain check)
    separation_ratio: float = DEFAULT_SEPARATION_RATIO

    def compute_rank_gap(self, n: int) -> int:
        """Compute effective rank gap, scaling with N if not explicitly set."""
        if self.rank_gap is not None:
            return self.rank_gap
        # Scale with N: ceil(fraction * n), clamped
        import math
        gap = math.ceil(self.rank_gap_fraction * n)
        return max(self.rank_gap_min, min(self.rank_gap_max, gap))


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


def _compute_cluster_cohesion(
    indices: list[int],
    dist_matrix: np.ndarray,
) -> float:
    """
    Compute cluster cohesion as max intra-cluster distance (diameter).

    Args:
        indices: Indices of cluster members
        dist_matrix: Full N×N distance matrix

    Returns:
        Maximum pairwise distance within cluster (diameter)
    """
    if len(indices) < 2:
        return 0.0

    max_dist = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = dist_matrix[indices[i], indices[j]]
            if d > max_dist:
                max_dist = d
    return max_dist


def _compute_inter_cluster_distance(
    indices1: list[int],
    indices2: list[int],
    dist_matrix: np.ndarray,
) -> float:
    """
    Compute minimum distance between two clusters.

    Args:
        indices1: Indices of first cluster members
        indices2: Indices of second cluster members
        dist_matrix: Full N×N distance matrix

    Returns:
        Minimum pairwise distance between clusters
    """
    min_dist = float('inf')
    for i in indices1:
        for j in indices2:
            d = dist_matrix[i, j]
            if d < min_dist:
                min_dist = d
    return min_dist if min_dist != float('inf') else 0.0


def ambiguity_detect(
    query: str,
    candidates: list[Candidate],
    config: Optional[AmbiguityConfig] = None,
) -> AmbiguityResult:
    """
    Detect if retrieval candidates form ambiguous clusters.

    Uses rank-gap for competitiveness (score-scale invariant) and
    cohesion checks to prevent garbage partitions of diffuse topics.

    Args:
        query: The query string (for logging only; clustering uses embeddings)
        candidates: List of Candidate objects with embeddings and scores
        config: Configuration parameters

    Returns:
        AmbiguityResult indicating whether disambiguation is needed
    """
    if config is None:
        config = AmbiguityConfig()

    # Cap candidates and sort by retrieval score (rank 0 = best)
    sorted_candidates = sorted(candidates, key=lambda c: -c.retr_score)[:config.n]
    n = len(sorted_candidates)

    # Build rank lookup: candidate index -> rank (0-based)
    rank_of = {i: i for i in range(n)}  # Already sorted, so index = rank

    # Compute effective rank gap (scales with N)
    effective_rank_gap = config.compute_rank_gap(n)

    logger.debug(
        f"ambiguity_detect: query='{query}', n={n}, k_max={config.k_max}, "
        f"m={config.min_cluster_size}, rank_gap={effective_rank_gap} (from n={n}), "
        f"cohesion_ratio={config.cohesion_ratio}, separation_ratio={config.separation_ratio}"
    )

    # Need at least 2*min_cluster_size candidates to have 2 valid clusters
    if n < 2 * config.min_cluster_size:
        logger.debug(f"Too few candidates ({n}) for ambiguity detection")
        return AmbiguityResult(
            ambiguous=False,
            reason=f"insufficient candidates (n={n} < 2*m={2*config.min_cluster_size})",
            n_candidates=n,
            max_rank_gap=config.compute_rank_gap(n),
            max_cohesion_ratio=config.cohesion_ratio,
            min_separation_ratio=config.separation_ratio,
        )

    # Build embedding matrix
    embeddings = np.vstack([c.emb for c in sorted_candidates])

    # Compute condensed distance matrix for linkage
    dist_condensed = _compute_distance_matrix(embeddings)

    # Also compute full N×N distance matrix for cohesion checks
    from scipy.spatial.distance import squareform
    dist_matrix = squareform(dist_condensed)

    # Compute overall mean distance for cohesion normalization
    overall_mean_dist = np.mean(dist_condensed) if len(dist_condensed) > 0 else 1.0

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

        # Compute cohesion for each valid cluster and filter out loose clusters
        cluster_cohesion = {}
        cohesion_ratios = {}
        for cid, indices in list(valid_clusters.items()):
            cohesion = _compute_cluster_cohesion(indices, dist_matrix)
            cluster_cohesion[cid] = cohesion
            ratio = cohesion / max(overall_mean_dist, 1e-6)
            cohesion_ratios[cid] = ratio

            if ratio > config.cohesion_ratio:
                logger.debug(
                    f"k={k}: cluster {cid} rejected (cohesion_ratio={ratio:.2f} > {config.cohesion_ratio})"
                )
                del valid_clusters[cid]

        if len(valid_clusters) < 2:
            logger.debug(f"k={k}: only {len(valid_clusters)} valid clusters after cohesion filter")
            continue

        # Separation check: ALL cluster pairs must be well-separated
        # Uses min inter-cluster distance (strongest check for chain detection)
        cluster_list = list(valid_clusters.items())
        min_separation = float('inf')
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                sep = _compute_inter_cluster_distance(
                    cluster_list[i][1], cluster_list[j][1], dist_matrix
                )
                if sep < min_separation:
                    min_separation = sep

        separation_ratio = min_separation / max(overall_mean_dist, 1e-6)

        if separation_ratio < config.separation_ratio:
            logger.debug(
                f"k={k}: clusters too close (min_inter_dist/mean_dist={separation_ratio:.2f} < {config.separation_ratio})"
            )
            continue

        # Compute best rank per valid cluster (lower rank = better)
        cluster_best_ranks = {}
        cluster_best_scores = {}
        for cid, indices in valid_clusters.items():
            ranks = [rank_of[i] for i in indices]
            best_rank = min(ranks)
            cluster_best_ranks[cid] = best_rank
            # Also track scores for logging/calibration
            scores = [sorted_candidates[i].retr_score for i in indices]
            cluster_best_scores[cid] = max(scores)

        # Sort clusters by best rank (ascending = best first)
        sorted_clusters = sorted(
            valid_clusters.items(),
            key=lambda x: cluster_best_ranks[x[0]]
        )

        # Competitiveness test: rank gap
        # ρ1 = rank of best item in cluster 1, ρ2 = rank of best item in cluster 2
        rho1 = cluster_best_ranks[sorted_clusters[0][0]]
        rho2 = cluster_best_ranks[sorted_clusters[1][0]]
        rank_gap = rho2 - rho1

        # Also compute score gap for logging/calibration
        b1 = cluster_best_scores[sorted_clusters[0][0]]
        b2 = cluster_best_scores[sorted_clusters[1][0]]
        score_gap = b1 - b2

        logger.debug(
            f"k={k}: {len(valid_clusters)} valid clusters, "
            f"ρ1={rho1}, ρ2={rho2}, rank_gap={rank_gap}, max_rank_gap={effective_rank_gap}, "
            f"b1={b1:.4f}, b2={b2:.4f}, score_gap={score_gap:.4f}"
        )

        if rank_gap <= effective_rank_gap:
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
            best_ranks = [cluster_best_ranks[cid] for cid, _ in sorted_clusters]
            cohesion_ratio_list = [cohesion_ratios.get(cid, 0.0) for cid, _ in sorted_clusters]

            best_result = AmbiguityResult(
                ambiguous=True,
                reason=f"found {len(valid_clusters)} competitive clusters at k={k}",
                options=options,
                n_candidates=n,
                chosen_k=chosen_k,
                cluster_sizes=cluster_sizes,
                best_scores=best_scores,
                best_ranks=best_ranks,
                rank_gap=rank_gap,
                score_gap=score_gap,
                cohesion_ratios=cohesion_ratio_list,
                separation_ratio=separation_ratio,
                overall_mean_dist=overall_mean_dist,
                max_rank_gap=effective_rank_gap,
                max_cohesion_ratio=config.cohesion_ratio,
                min_separation_ratio=config.separation_ratio,
            )

            logger.info(
                f"Ambiguity detected: query='{query}', k={chosen_k}, "
                f"clusters={cluster_sizes}, rank_gap={rank_gap}, score_gap={score_gap:.4f}"
            )
            break

    if best_result:
        return best_result

    # No ambiguity found
    logger.debug(f"No ambiguity detected for query='{query}'")
    return AmbiguityResult(
        ambiguous=False,
        reason="single coherent neighborhood (no competitive clusters found)",
        n_candidates=n,
        overall_mean_dist=overall_mean_dist,
        max_rank_gap=effective_rank_gap,
        max_cohesion_ratio=config.cohesion_ratio,
        min_separation_ratio=config.separation_ratio,
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

    # Cluster A: 5 points (ranks 0, 2, 4, 6, 8)
    for i in range(5):
        noise = np.random.randn(3) * 0.1
        emb = centroid_a + noise
        emb = emb / np.linalg.norm(emb)  # L2 normalize
        candidates.append(Candidate(
            id=i,
            text=f"Cluster A item {i}: programming language syntax",
            emb=emb,
            retr_score=0.90 - i * 0.02,  # Interleaved scores
        ))

    # Cluster B: 5 points (ranks 1, 3, 5, 7, 9)
    for i in range(5):
        noise = np.random.randn(3) * 0.1
        emb = centroid_b + noise
        emb = emb / np.linalg.norm(emb)
        candidates.append(Candidate(
            id=i + 5,
            text=f"Cluster B item {i}: coffee brewing methods",
            emb=emb,
            retr_score=0.89 - i * 0.02,  # Interleaved with cluster A
        ))

    result = ambiguity_detect("java", candidates)
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULT")
    print("=" * 60)
    print(f"Ambiguous: {result.ambiguous}")
    print(f"Reason: {result.reason}")
    print(f"Chosen k: {result.chosen_k}")
    print(f"Cluster sizes: {result.cluster_sizes}")
    print(f"Best ranks: {result.best_ranks}")
    print(f"Rank gap: {result.rank_gap}")
    print(f"Best scores: {result.best_scores}")
    print(f"Score gap: {result.score_gap}")
    print(f"Cohesion ratios: {result.cohesion_ratios}")
    print()
    print(format_disambiguation_prompt("java", result))
