#!/usr/bin/env python3
"""
Polysemy guard: detect ambiguous queries and surface clarification options.

Policy:
1. Detect ambiguity cheaply (static list + check for existing disambiguation)
2. On ambiguity: don't guess, return options derived from retrieval clusters
3. After user selects: fast bi-encoder + cross-encoder path

No LLM, no hand-coded sense cues, no ontology.
"""

import re
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Known ambiguous headwords - small static product list
# These are terms where polysemy causes retrieval confusion
AMBIGUOUS_HEADWORDS = {
    # Tech vs non-tech
    "java": ["programming language", "coffee/island"],
    "apple": ["company/computers", "fruit"],
    "rust": ["programming language", "corrosion/oxidation"],
    "shell": ["command line/bash", "seashell/mollusk"],
    "kernel": ["OS kernel", "corn/seed"],
    "python": ["programming language", "snake"],
    "branch": ["git/version control", "tree branch"],
    "model": ["ML/neural network", "fashion/person"],
    # Could extend with more as discovered
}

# Patterns that indicate the user already disambiguated
DISAMBIGUATION_PATTERNS = {
    "java": [r"java\s+(code|programming|jvm|class|spring|maven)", r"coffee|island|indonesi"],
    "apple": [r"apple\s+(computer|mac|ios|iphone|watch|silicon)", r"apple\s+(fruit|pie|tree|orchard)"],
    "rust": [r"rust\s+(lang|programming|cargo|crate|borrow)", r"rust\s+(corrosion|metal|iron|oxidat)"],
    "shell": [r"(bash|zsh|fish|sh)\s+shell", r"shell\s+(script|command|terminal)", r"sea\s*shell|shell\s*(fish|beach)"],
    "kernel": [r"(linux|os|operating)\s+kernel", r"kernel\s+(module|driver|space)", r"(corn|popcorn|seed)\s+kernel"],
    "python": [r"python\s+(code|script|pip|import|def|class)", r"python\s+(snake|reptile|animal)"],
    "branch": [r"git\s+branch", r"branch\s+(merge|checkout|pull)", r"tree\s+branch", r"branch\s+(leaf|leaves)"],
    "model": [r"(ml|machine\s+learning|neural|trained)\s+model", r"model\s+(weights|checkpoint|inference)", r"(fashion|runway)\s+model"],
}


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection."""
    is_ambiguous: bool
    headword: Optional[str] = None
    sense_options: Optional[list[str]] = None
    clusters: Optional[list[list[int]]] = None  # candidate IDs per cluster
    cluster_labels: Optional[list[str]] = None  # distinctive label per cluster


def detect_ambiguous_headword(query: str) -> Optional[str]:
    """Check if query contains an ambiguous headword without disambiguation."""
    query_lower = query.lower()

    for headword in AMBIGUOUS_HEADWORDS:
        if headword not in query_lower:
            continue

        # Check if already disambiguated
        patterns = DISAMBIGUATION_PATTERNS.get(headword, [])
        already_disambiguated = any(re.search(p, query_lower) for p in patterns)

        if not already_disambiguated:
            return headword

    return None


def is_recall_intent(query: str) -> bool:
    """Check if query has recall intent (asking about past discussions)."""
    recall_patterns = [
        r"^(when|what|where|how)\s+(did|do|have)\s+(we|i)\s+(discuss|talk|mention|say)",
        r"^(did|do|have)\s+(we|i)\s+(discuss|talk|mention|say)",
        r"^(what|tell me)\s+(about|regarding)",
        r"^(find|search|recall|remember)",
        r"conversations?\s+about",
        r"discussed?\s+about",
    ]
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in recall_patterns)


def cluster_candidates(
    candidate_ids: list[int],
    candidate_texts: list[str],
    embeddings: np.ndarray,
    k: int = 2,
) -> tuple[list[list[int]], list[str]]:
    """
    Cluster candidates into k groups using embeddings.

    Uses "most-dissimilar pair as seeds, assign by cosine" approach.
    Returns cluster assignments and distinctive labels for each cluster.
    """
    if len(candidate_ids) < 2:
        return [candidate_ids], ["all results"]

    n = len(candidate_ids)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings / norms

    # Find most dissimilar pair as seeds
    similarity_matrix = normalized @ normalized.T
    np.fill_diagonal(similarity_matrix, 1.0)  # ignore self-similarity

    # Find minimum similarity pair
    min_idx = np.unravel_index(np.argmin(similarity_matrix), similarity_matrix.shape)
    seed1, seed2 = min_idx[0], min_idx[1]

    # Assign each point to nearest seed
    clusters: list[list[int]] = [[], []]
    for i in range(n):
        sim_to_seed1 = similarity_matrix[i, seed1]
        sim_to_seed2 = similarity_matrix[i, seed2]
        if sim_to_seed1 >= sim_to_seed2:
            clusters[0].append(candidate_ids[i])
        else:
            clusters[1].append(candidate_ids[i])

    # Generate distinctive labels using tf-idf-like token scoring
    cluster_labels = []
    all_tokens = set()
    cluster_token_counts = []

    for cluster_ids in clusters:
        cluster_texts = [candidate_texts[candidate_ids.index(cid)] for cid in cluster_ids]
        tokens = set()
        for text in cluster_texts:
            # Simple tokenization
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            tokens.update(words)
        cluster_token_counts.append(tokens)
        all_tokens.update(tokens)

    # Find distinctive tokens (appear in one cluster but not other)
    for i, cluster_ids in enumerate(clusters):
        other_idx = 1 - i
        distinctive = cluster_token_counts[i] - cluster_token_counts[other_idx]

        # Pick most frequent distinctive tokens from cluster texts
        cluster_texts = [candidate_texts[candidate_ids.index(cid)] for cid in cluster_ids]
        combined = " ".join(cluster_texts).lower()

        token_freq = {}
        for token in distinctive:
            token_freq[token] = combined.count(token)

        # Get top 3 distinctive tokens
        top_tokens = sorted(token_freq.items(), key=lambda x: -x[1])[:3]
        if top_tokens:
            label = ", ".join(t[0] for t in top_tokens)
        else:
            # Fallback: use first few words from first result
            label = " ".join(cluster_texts[0].split()[:5]) + "..." if cluster_texts else "group"

        cluster_labels.append(label)

    return clusters, cluster_labels


def check_ambiguity(
    query: str,
    candidate_ids: list[int],
    candidate_texts: list[str],
    embeddings: Optional[np.ndarray] = None,
) -> AmbiguityResult:
    """
    Check if a query is ambiguous and needs clarification.

    Returns AmbiguityResult with:
    - is_ambiguous: whether clarification is needed
    - headword: the ambiguous term
    - sense_options: the predefined sense descriptions
    - clusters: candidate IDs grouped by sense
    - cluster_labels: distinctive labels for each cluster
    """
    # Only trigger for recall-intent queries
    if not is_recall_intent(query):
        return AmbiguityResult(is_ambiguous=False)

    # Check for ambiguous headword
    headword = detect_ambiguous_headword(query)
    if headword is None:
        return AmbiguityResult(is_ambiguous=False)

    # Get predefined sense options
    sense_options = AMBIGUOUS_HEADWORDS[headword]

    # Cluster candidates if embeddings provided
    if embeddings is not None and len(candidate_ids) >= 2:
        clusters, cluster_labels = cluster_candidates(
            candidate_ids, candidate_texts, embeddings, k=2
        )
    else:
        clusters = None
        cluster_labels = None

    return AmbiguityResult(
        is_ambiguous=True,
        headword=headword,
        sense_options=sense_options,
        clusters=clusters,
        cluster_labels=cluster_labels,
    )


def format_disambiguation_prompt(result: AmbiguityResult) -> str:
    """Format a user-facing disambiguation prompt."""
    if not result.is_ambiguous:
        return ""

    lines = [f"I found multiple meanings of '{result.headword}'. Which did you mean?"]

    for i, sense in enumerate(result.sense_options or [], 1):
        lines.append(f"  {i}. {sense}")

    if result.cluster_labels:
        lines.append("")
        lines.append("Based on your history, I found results related to:")
        for i, label in enumerate(result.cluster_labels, 1):
            lines.append(f"  {i}. {label}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test cases
    test_queries = [
        ("when did we discuss java", True, "java"),
        ("java programming patterns", False, None),  # already disambiguated
        ("what about rust", True, "rust"),
        ("rust language borrowing", False, None),  # already disambiguated
        ("tell me about the kernel", True, "kernel"),
        ("linux kernel modules", False, None),  # already disambiguated
        ("find python discussions", True, "python"),
        ("python code examples", False, None),  # already disambiguated
        ("how is the weather", False, None),  # no ambiguous term
    ]

    print("Ambiguity Detection Tests")
    print("=" * 60)

    for query, expected_ambiguous, expected_headword in test_queries:
        result = check_ambiguity(query, [], [], None)
        status = "✓" if result.is_ambiguous == expected_ambiguous else "✗"
        print(f"{status} '{query}'")
        print(f"    Expected: ambiguous={expected_ambiguous}, headword={expected_headword}")
        print(f"    Got:      ambiguous={result.is_ambiguous}, headword={result.headword}")
        print()
