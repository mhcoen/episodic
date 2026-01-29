#!/usr/bin/env python3
"""
Centroid Drift Diagnostics for Eligibility-Gated Segmentation

This script implements the diagnostic protocol specified by Desktop to evaluate
whether centroid drift is a viable eligibility signal for dialogue segmentation.

Diagnostics:
- D1: Oracle-reset centroid (upper bound) - reset at gold boundaries
- D2: No-reset rolling centroid (realistic) - continuous EMA
- D3: Pairwise last-two drift (baseline comparison)
- Shuffled sanity check
- Eligibility operating characteristics

Usage:
    python centroid_drift_diagnostics.py [--sample N] [--alpha 0.1] [--output results.json]
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Corpus configurations
CORPUS_CONFIGS = {
    "superseg": {
        "path": "datasets/superseg/segmentation_file_test.json",
        "key": "superseg-v2",
        "boundary_encoding": "segmentation_label",  # 1 = boundary after this turn
        "type": "task-oriented"
    },
    "dialseg711": {
        "path": "datasets/dialseg711/segmentation_file_test.json",
        "key": "dialseg711",
        "boundary_encoding": "segmentation_label",
        "type": "task-oriented"
    },
    "tiage": {
        "path": "datasets/tiage/segmentation_file_test.json",
        "key": "tiage",
        "boundary_encoding": "segmentation_label",
        "type": "task-oriented"
    },
    "multiwoz": {
        "path": "datasets/multiwoz/segmentation_file_test.json",
        "key": "multiwoz",
        "boundary_encoding": "topic_id_change",
        "type": "task-oriented"
    },
    "dailydialog": {
        "path": "datasets/dailydialog/segmentation_file_test.json",
        "key": "dailydialog-synthetic",
        "boundary_encoding": "topic_id_change",
        "type": "open-domain"
    },
    "taskmaster": {
        "path": "datasets/taskmaster/segmentation_file_test.json",
        "key": "taskmaster",
        "boundary_encoding": "topic_id_change",
        "type": "semi-structured"
    },
    "topical_chat": {
        "path": "datasets/topical_chat/segmentation_file_test.json",
        "key": "topical_chat",
        "boundary_encoding": "topic_id_change",
        "type": "open-domain"
    },
    "qmsum": {
        "path": "datasets/qmsum/segmentation_file_test.json",
        "key": "qmsum",
        "boundary_encoding": "topic_id_change",
        "type": "semi-structured"
    }
}


@dataclass
class Turn:
    """A single dialogue turn."""
    turn_id: int
    role: str
    utterance: str
    topic_id: int
    is_boundary: bool = False  # True if this is the first turn of a new segment
    embedding: Optional[np.ndarray] = None


@dataclass
class Dialogue:
    """A complete dialogue with turns and metadata."""
    dial_id: str
    turns: List[Turn]
    num_topics: int = 0
    user_turns: List[Turn] = field(default_factory=list)

    def __post_init__(self):
        self.user_turns = [t for t in self.turns if t.role in ('user', 'User')]


@dataclass
class CorpusStats:
    """Statistics for a corpus."""
    name: str
    num_dialogues: int
    total_turns: int
    total_user_turns: int
    gold_boundaries: int
    boundary_density: float  # boundaries per user turn
    mean_segment_length: float  # in user turns
    median_segment_length: float
    type: str
    path: str
    encoding: str
    quirks: str = ""


@dataclass
class DriftResult:
    """Results for a single drift diagnostic."""
    auroc: float
    boundary_median: float
    boundary_iqr: Tuple[float, float]
    nonboundary_median: float
    nonboundary_iqr: Tuple[float, float]
    nonboundary_p90: float
    overlap_ratio: float  # median(boundary) / p90(nonboundary)


@dataclass
class DiagnosticResults:
    """Complete diagnostic results for a corpus."""
    corpus: str
    stats: CorpusStats
    d1_results: Dict[float, DriftResult]  # keyed by alpha
    d2_results: Dict[float, DriftResult]
    d3_results: DriftResult
    shuffled_auroc: float
    best_alpha_d1: float
    best_alpha_d2: float
    separability_label: str  # good/marginal/poor
    eligibility_curves: Optional[Dict] = None


class EmbeddingProvider:
    """Efficient embedding provider with batching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._cache = {}

    def _init_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts, returning L2-normalized embeddings."""
        self._init_model()

        # Check cache for already computed embeddings
        to_compute = []
        to_compute_idx = []
        for i, text in enumerate(texts):
            if text not in self._cache:
                to_compute.append(text)
                to_compute_idx.append(i)

        # Compute missing embeddings
        if to_compute:
            embeddings = self.model.encode(to_compute, show_progress_bar=False, convert_to_numpy=True)
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            for text, emb in zip(to_compute, embeddings):
                self._cache[text] = emb

        # Build result array
        result = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        for i, text in enumerate(texts):
            result[i] = self._cache[text]

        return result

    def clear_cache(self):
        self._cache.clear()


def load_corpus(corpus_name: str, config: dict, project_root: Path, sample_n: Optional[int] = None) -> Tuple[List[Dialogue], CorpusStats]:
    """Load a corpus and compute basic statistics."""

    file_path = project_root / config["path"]
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)
    key = config["key"]

    if key not in dial_data:
        # Try to find the key
        available_keys = list(dial_data.keys())
        if len(available_keys) == 1:
            key = available_keys[0]
        else:
            raise KeyError(f"Key '{key}' not found. Available: {available_keys}")

    raw_dialogues = dial_data[key]

    # Sample if requested
    if sample_n and len(raw_dialogues) > sample_n:
        np.random.seed(42)
        indices = np.random.choice(len(raw_dialogues), sample_n, replace=False)
        raw_dialogues = [raw_dialogues[i] for i in sorted(indices)]

    dialogues = []
    total_turns = 0
    total_user_turns = 0
    gold_boundaries = 0
    segment_lengths = []

    encoding = config["boundary_encoding"]

    for dial in raw_dialogues:
        dial_id = dial.get("dial_id", "unknown")
        raw_turns = dial.get("turns", [])

        if not raw_turns:
            continue

        turns = []
        prev_topic_id = None
        current_segment_user_count = 0

        for i, t in enumerate(raw_turns):
            role = t.get("role", "").lower()
            # Normalize role names
            if role in ("agent", "assistant", "system"):
                role = "assistant"
            elif role in ("user",):
                role = "user"

            topic_id = t.get("topic_id", 0)

            # Determine if this turn starts a new segment
            if encoding == "segmentation_label":
                # segmentation_label=1 on turn that ENDS a segment
                # So boundary is on the NEXT turn
                if i > 0 and raw_turns[i-1].get("segmentation_label", 0) == 1:
                    is_boundary = True
                else:
                    is_boundary = False
            else:  # topic_id_change
                if prev_topic_id is not None and topic_id != prev_topic_id:
                    is_boundary = True
                else:
                    is_boundary = False

            turn = Turn(
                turn_id=t.get("turn_id", i+1),
                role=role,
                utterance=t.get("utterance", ""),
                topic_id=topic_id,
                is_boundary=is_boundary
            )
            turns.append(turn)
            prev_topic_id = topic_id

            # Count user turns and boundaries
            if role == "user":
                total_user_turns += 1
                if is_boundary:
                    gold_boundaries += 1
                    if current_segment_user_count > 0:
                        segment_lengths.append(current_segment_user_count)
                    current_segment_user_count = 1
                else:
                    current_segment_user_count += 1

        # Add final segment
        if current_segment_user_count > 0:
            segment_lengths.append(current_segment_user_count)

        total_turns += len(turns)

        dialogue = Dialogue(
            dial_id=dial_id,
            turns=turns,
            num_topics=dial.get("num_topics", len(set(t.topic_id for t in turns)))
        )
        dialogues.append(dialogue)

    # Compute statistics
    boundary_density = gold_boundaries / total_user_turns if total_user_turns > 0 else 0
    mean_seg_len = np.mean(segment_lengths) if segment_lengths else 0
    median_seg_len = np.median(segment_lengths) if segment_lengths else 0

    stats = CorpusStats(
        name=corpus_name,
        num_dialogues=len(dialogues),
        total_turns=total_turns,
        total_user_turns=total_user_turns,
        gold_boundaries=gold_boundaries,
        boundary_density=boundary_density,
        mean_segment_length=mean_seg_len,
        median_segment_length=median_seg_len,
        type=config["type"],
        path=str(file_path),
        encoding=encoding
    )

    return dialogues, stats


def compute_embeddings(dialogues: List[Dialogue], embedder: EmbeddingProvider) -> None:
    """Compute embeddings for all user turns in dialogues."""
    all_texts = []
    text_to_turns = defaultdict(list)

    for dialogue in dialogues:
        for turn in dialogue.user_turns:
            all_texts.append(turn.utterance)
            text_to_turns[turn.utterance].append(turn)

    if not all_texts:
        return

    print(f"  Computing embeddings for {len(all_texts)} user turns...")

    # Batch embed
    unique_texts = list(set(all_texts))
    embeddings = embedder.embed_batch(unique_texts)
    text_to_embedding = {text: emb for text, emb in zip(unique_texts, embeddings)}

    # Assign to turns
    for text, turns in text_to_turns.items():
        for turn in turns:
            turn.embedding = text_to_embedding[text]


def compute_centroid_drift_d1(dialogues: List[Dialogue], alpha: float) -> Tuple[List[float], List[bool]]:
    """
    D1: Oracle-reset centroid (upper bound).
    Reset centroid at gold boundaries, compute drift for each user turn.

    Key insight: Compute drift FIRST using previous segment's centroid,
    THEN reset if boundary. This gives us the drift score for boundary turns.

    Returns: (drift_scores, is_boundary_labels)
    """
    all_drifts = []
    all_labels = []

    for dialogue in dialogues:
        user_turns = dialogue.user_turns
        if len(user_turns) < 2:
            continue

        centroid = None

        for i, turn in enumerate(user_turns):
            if turn.embedding is None:
                continue

            # First turn initializes centroid, no drift to compute
            if centroid is None:
                centroid = turn.embedding.copy()
                continue

            # Compute drift BEFORE any reset
            cos_sim = np.dot(centroid, turn.embedding)
            drift = 1.0 - cos_sim

            all_drifts.append(drift)
            all_labels.append(turn.is_boundary)

            # AFTER computing drift: if boundary, reset centroid
            if turn.is_boundary:
                # Oracle reset: start fresh segment
                centroid = turn.embedding.copy()
            else:
                # Update centroid (EMA)
                centroid = (1 - alpha) * centroid + alpha * turn.embedding
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

    return all_drifts, all_labels


def compute_centroid_drift_d2(dialogues: List[Dialogue], alpha: float) -> Tuple[List[float], List[bool]]:
    """
    D2: No-reset rolling centroid (realistic).
    Maintain centroid EMA continuously, no oracle resets.

    Returns: (drift_scores, is_boundary_labels)
    """
    all_drifts = []
    all_labels = []

    for dialogue in dialogues:
        user_turns = dialogue.user_turns
        if len(user_turns) < 2:
            continue

        centroid = None

        for i, turn in enumerate(user_turns):
            if turn.embedding is None:
                continue

            if centroid is None:
                centroid = turn.embedding.copy()
                continue

            # Compute drift
            cos_sim = np.dot(centroid, turn.embedding)
            drift = 1.0 - cos_sim

            all_drifts.append(drift)
            all_labels.append(turn.is_boundary)

            # Update centroid (EMA) - NO reset
            centroid = (1 - alpha) * centroid + alpha * turn.embedding
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

    return all_drifts, all_labels


def compute_pairwise_drift_d3(dialogues: List[Dialogue]) -> Tuple[List[float], List[bool]]:
    """
    D3: Pairwise last-two-user-turn drift (baseline).
    Compute cosine distance between consecutive user turns.

    Returns: (drift_scores, is_boundary_labels)
    """
    all_drifts = []
    all_labels = []

    for dialogue in dialogues:
        user_turns = dialogue.user_turns
        if len(user_turns) < 2:
            continue

        prev_embedding = None

        for turn in user_turns:
            if turn.embedding is None:
                continue

            if prev_embedding is None:
                prev_embedding = turn.embedding
                continue

            # Compute pairwise drift
            cos_sim = np.dot(prev_embedding, turn.embedding)
            drift = 1.0 - cos_sim

            all_drifts.append(drift)
            all_labels.append(turn.is_boundary)

            prev_embedding = turn.embedding

    return all_drifts, all_labels


def compute_shuffled_d2(dialogues: List[Dialogue], alpha: float) -> float:
    """
    Sanity check: shuffle embeddings within each dialogue and compute D2 AUROC.
    Should be ~0.5 if there's no artifact.
    """
    all_drifts = []
    all_labels = []

    for dialogue in dialogues:
        user_turns = [t for t in dialogue.user_turns if t.embedding is not None]
        if len(user_turns) < 2:
            continue

        # Shuffle embeddings
        embeddings = [t.embedding.copy() for t in user_turns]
        np.random.shuffle(embeddings)

        centroid = None

        for i, (turn, emb) in enumerate(zip(user_turns, embeddings)):
            if centroid is None:
                centroid = emb.copy()
                continue

            cos_sim = np.dot(centroid, emb)
            drift = 1.0 - cos_sim

            all_drifts.append(drift)
            all_labels.append(turn.is_boundary)

            centroid = (1 - alpha) * centroid + alpha * emb
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

    return compute_auroc(all_drifts, all_labels)


def compute_auroc(scores: List[float], labels: List[bool]) -> float:
    """Compute AUROC for binary classification."""
    if not scores or not any(labels) or all(labels):
        return 0.5

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def compute_drift_result(drifts: List[float], labels: List[bool]) -> DriftResult:
    """Compute full drift result with quantiles."""
    if not drifts:
        return DriftResult(0.5, 0.0, (0.0, 0.0), 0.0, (0.0, 0.0), 0.0, 0.0)

    drifts = np.array(drifts)
    labels = np.array(labels)

    boundary_drifts = drifts[labels]
    nonboundary_drifts = drifts[~labels]

    if len(boundary_drifts) == 0 or len(nonboundary_drifts) == 0:
        return DriftResult(0.5, 0.0, (0.0, 0.0), 0.0, (0.0, 0.0), 0.0, 0.0)

    auroc = compute_auroc(drifts.tolist(), labels.tolist())

    b_median = float(np.median(boundary_drifts))
    b_q25 = float(np.percentile(boundary_drifts, 25))
    b_q75 = float(np.percentile(boundary_drifts, 75))

    nb_median = float(np.median(nonboundary_drifts))
    nb_q25 = float(np.percentile(nonboundary_drifts, 25))
    nb_q75 = float(np.percentile(nonboundary_drifts, 75))
    nb_p90 = float(np.percentile(nonboundary_drifts, 90))

    overlap_ratio = b_median / nb_p90 if nb_p90 > 0 else 0.0

    return DriftResult(
        auroc=auroc,
        boundary_median=b_median,
        boundary_iqr=(b_q25, b_q75),
        nonboundary_median=nb_median,
        nonboundary_iqr=(nb_q25, nb_q75),
        nonboundary_p90=nb_p90,
        overlap_ratio=overlap_ratio
    )


def compute_eligibility_curves(drifts: List[float], labels: List[bool],
                                thresholds: Optional[List[float]] = None) -> Dict:
    """
    Compute eligibility operating characteristics across thresholds.
    """
    if not drifts:
        return {}

    drifts = np.array(drifts)
    labels = np.array(labels)

    if thresholds is None:
        # Use quantiles of non-boundary distribution + fixed grid
        nonboundary_drifts = drifts[~labels]
        if len(nonboundary_drifts) > 0:
            quantile_thresholds = [np.percentile(nonboundary_drifts, q) for q in [50, 70, 80, 90, 95]]
        else:
            quantile_thresholds = []
        fixed_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        thresholds = sorted(set(quantile_thresholds + fixed_thresholds))

    results = []
    total = len(drifts)
    total_boundaries = labels.sum()

    for theta in thresholds:
        eligible = drifts >= theta
        eligible_count = eligible.sum()
        eligible_boundaries = (eligible & labels).sum()

        eligibility_rate = eligible_count / total if total > 0 else 0
        eligibility_recall = eligible_boundaries / total_boundaries if total_boundaries > 0 else 0
        eligibility_precision = eligible_boundaries / eligible_count if eligible_count > 0 else 0

        results.append({
            "theta": float(theta),
            "eligibility_rate": float(eligibility_rate),
            "eligibility_recall": float(eligibility_recall),
            "eligibility_precision": float(eligibility_precision)
        })

    # Find "useful band" - high recall (>=0.8), rate < 1, meaningful precision
    useful_band = [r for r in results
                   if r["eligibility_recall"] >= 0.8
                   and r["eligibility_rate"] < 0.95
                   and r["eligibility_precision"] > 0.01]

    return {
        "curves": results,
        "useful_band": useful_band,
        "has_useful_band": len(useful_band) > 0
    }


def get_separability_label(d2_auroc: float) -> str:
    """Classify separability as good/marginal/poor."""
    if d2_auroc >= 0.7:
        return "good"
    elif d2_auroc >= 0.6:
        return "marginal"
    else:
        return "poor"


def run_diagnostics(corpus_name: str, config: dict, project_root: Path,
                    embedder: EmbeddingProvider, alphas: List[float],
                    sample_n: Optional[int] = None) -> DiagnosticResults:
    """Run all diagnostics for a single corpus."""

    print(f"\n{'='*60}")
    print(f"Processing corpus: {corpus_name}")
    print(f"{'='*60}")

    # Load corpus
    dialogues, stats = load_corpus(corpus_name, config, project_root, sample_n)
    print(f"  Loaded {stats.num_dialogues} dialogues, {stats.total_user_turns} user turns, {stats.gold_boundaries} boundaries")
    print(f"  Boundary density: {stats.boundary_density:.4f}, Mean segment length: {stats.mean_segment_length:.2f}")

    # Compute embeddings
    compute_embeddings(dialogues, embedder)

    # D1 and D2 for each alpha
    d1_results = {}
    d2_results = {}

    for alpha in alphas:
        print(f"  Running D1/D2 with alpha={alpha}...")

        drifts_d1, labels_d1 = compute_centroid_drift_d1(dialogues, alpha)
        drifts_d2, labels_d2 = compute_centroid_drift_d2(dialogues, alpha)

        d1_results[alpha] = compute_drift_result(drifts_d1, labels_d1)
        d2_results[alpha] = compute_drift_result(drifts_d2, labels_d2)

        print(f"    D1 AUROC: {d1_results[alpha].auroc:.4f}, D2 AUROC: {d2_results[alpha].auroc:.4f}")

    # D3: Pairwise baseline
    print("  Running D3 (pairwise baseline)...")
    drifts_d3, labels_d3 = compute_pairwise_drift_d3(dialogues)
    d3_results = compute_drift_result(drifts_d3, labels_d3)
    print(f"    D3 AUROC: {d3_results.auroc:.4f}")

    # Shuffled sanity check
    print("  Running shuffled sanity check...")
    best_alpha = max(alphas, key=lambda a: d2_results[a].auroc)
    shuffled_auroc = compute_shuffled_d2(dialogues, best_alpha)
    print(f"    Shuffled D2 AUROC: {shuffled_auroc:.4f} (expected ~0.5)")

    # Find best alphas
    best_alpha_d1 = max(alphas, key=lambda a: d1_results[a].auroc)
    best_alpha_d2 = max(alphas, key=lambda a: d2_results[a].auroc)

    # Separability label
    separability = get_separability_label(d2_results[best_alpha_d2].auroc)

    # Eligibility curves (if D2 is at least modest)
    eligibility_curves = None
    if d2_results[best_alpha_d2].auroc >= 0.55:
        print("  Computing eligibility curves...")
        drifts_d2, labels_d2 = compute_centroid_drift_d2(dialogues, best_alpha_d2)
        eligibility_curves = compute_eligibility_curves(drifts_d2, labels_d2)
        if eligibility_curves.get("has_useful_band"):
            print(f"    Found useful band with {len(eligibility_curves['useful_band'])} configurations")
        else:
            print("    No useful band found")

    # Detect quirks
    quirks = []
    if stats.median_segment_length < 3:
        quirks.append("short segments (<3 user turns)")
    if abs(shuffled_auroc - 0.5) > 0.1:
        quirks.append(f"shuffled AUROC={shuffled_auroc:.2f} (expected ~0.5)")
    if d1_results[best_alpha_d1].auroc - d2_results[best_alpha_d2].auroc > 0.15:
        quirks.append("large D1-D2 gap (oracle artifact)")
    stats.quirks = "; ".join(quirks) if quirks else "none"

    return DiagnosticResults(
        corpus=corpus_name,
        stats=stats,
        d1_results=d1_results,
        d2_results=d2_results,
        d3_results=d3_results,
        shuffled_auroc=shuffled_auroc,
        best_alpha_d1=best_alpha_d1,
        best_alpha_d2=best_alpha_d2,
        separability_label=separability,
        eligibility_curves=eligibility_curves
    )


def format_table(results: List[DiagnosticResults]) -> str:
    """Format results as a markdown table."""

    lines = []
    lines.append("## Summary Table: Centroid Drift Diagnostics")
    lines.append("")
    lines.append("| Corpus | Type | Dialogues | User Turns | Boundaries | Density | Med Seg Len | D1 (best α) | D2 (best α) | D3 (pairwise) | Shuffled | Label |")
    lines.append("|--------|------|-----------|------------|------------|---------|-------------|-------------|-------------|---------------|----------|-------|")

    for r in results:
        s = r.stats
        d1_best = r.d1_results[r.best_alpha_d1]
        d2_best = r.d2_results[r.best_alpha_d2]

        line = f"| {s.name} | {s.type} | {s.num_dialogues} | {s.total_user_turns} | {s.gold_boundaries} | {s.boundary_density:.3f} | {s.median_segment_length:.1f} | {d1_best.auroc:.3f} (α={r.best_alpha_d1}) | {d2_best.auroc:.3f} (α={r.best_alpha_d2}) | {r.d3_results.auroc:.3f} | {r.shuffled_auroc:.3f} | {r.separability_label} |"
        lines.append(line)

    return "\n".join(lines)


def format_corpus_notes(results: List[DiagnosticResults]) -> str:
    """Format per-corpus notes."""

    lines = []
    lines.append("\n## Per-Corpus Notes")

    for r in results:
        s = r.stats
        lines.append(f"\n### {s.name}")
        lines.append(f"- **Path**: `{s.path}`")
        lines.append(f"- **Format**: JSON with `dial_data.{CORPUS_CONFIGS[s.name]['key']}` array")
        lines.append(f"- **Boundary encoding**: {s.encoding}")
        lines.append(f"- **Type**: {s.type}")
        lines.append(f"- **Quirks**: {s.quirks}")

        # Drift distribution summary
        d2_best = r.d2_results[r.best_alpha_d2]
        lines.append(f"- **D2 Distribution**: boundary median={d2_best.boundary_median:.4f} vs non-boundary p90={d2_best.nonboundary_p90:.4f}")
        lines.append(f"  - Overlap ratio (median_b / p90_nb): {d2_best.overlap_ratio:.2f}")

        # Eligibility band
        if r.eligibility_curves and r.eligibility_curves.get("has_useful_band"):
            best_config = max(r.eligibility_curves["useful_band"],
                            key=lambda x: x["eligibility_recall"] * x["eligibility_precision"])
            lines.append(f"- **Useful eligibility band**: θ={best_config['theta']:.3f} → recall={best_config['eligibility_recall']:.2f}, precision={best_config['eligibility_precision']:.3f}, rate={best_config['eligibility_rate']:.2f}")
        elif r.eligibility_curves:
            lines.append("- **Eligibility band**: None found (recall <0.8 or rate ≥0.95)")
        else:
            lines.append("- **Eligibility**: Skipped (D2 AUROC too low)")

    return "\n".join(lines)


def format_recommendations(results: List[DiagnosticResults]) -> str:
    """Format recommendations."""

    lines = []
    lines.append("\n## Recommendations")

    # Sort by D2 AUROC
    sorted_results = sorted(results, key=lambda r: r.d2_results[r.best_alpha_d2].auroc, reverse=True)

    # Categorize
    good = [r for r in sorted_results if r.separability_label == "good"]
    marginal = [r for r in sorted_results if r.separability_label == "marginal"]
    poor = [r for r in sorted_results if r.separability_label == "poor"]

    lines.append("\n### Separability Classification")
    lines.append(f"- **Good** (D2 AUROC ≥0.70): {', '.join(r.corpus for r in good) or 'None'}")
    lines.append(f"- **Marginal** (0.60-0.70): {', '.join(r.corpus for r in marginal) or 'None'}")
    lines.append(f"- **Poor** (<0.60): {', '.join(r.corpus for r in poor) or 'None'}")

    # Pick 3 for ablation
    lines.append("\n### Recommended Corpora for Deeper Ablation")
    picks = []
    if good:
        picks.append(good[0])
    if marginal:
        picks.append(marginal[0])
    if poor:
        picks.append(poor[0])

    if len(picks) < 3:
        # Fill from sorted list
        for r in sorted_results:
            if r not in picks and len(picks) < 3:
                picks.append(r)

    for i, r in enumerate(picks, 1):
        lines.append(f"{i}. **{r.corpus}** ({r.separability_label}) - D2 AUROC={r.d2_results[r.best_alpha_d2].auroc:.3f}")

    # Overall recommendation
    lines.append("\n### Overall Assessment")

    # Check if centroid drift beats pairwise
    centroid_wins = sum(1 for r in results
                        if r.d2_results[r.best_alpha_d2].auroc > r.d3_results.auroc + 0.02)
    pairwise_wins = sum(1 for r in results
                        if r.d3_results.auroc > r.d2_results[r.best_alpha_d2].auroc + 0.02)
    ties = len(results) - centroid_wins - pairwise_wins

    lines.append(f"- Centroid drift vs pairwise: {centroid_wins} wins, {pairwise_wins} losses, {ties} ties")

    # Eligibility assessment
    have_useful_band = sum(1 for r in results
                          if r.eligibility_curves and r.eligibility_curves.get("has_useful_band"))
    lines.append(f"- Corpora with useful eligibility band: {have_useful_band}/{len(results)}")

    # Final verdict
    avg_d2 = np.mean([r.d2_results[r.best_alpha_d2].auroc for r in results])

    if avg_d2 >= 0.65 and centroid_wins >= pairwise_wins and have_useful_band >= len(results) // 2:
        verdict = "**PROCEED** with eligibility-gated method. Centroid drift shows meaningful separability across corpora."
    elif avg_d2 >= 0.55 and have_useful_band >= 2:
        verdict = "**PROCEED with caution**. Separability is marginal but some corpora show promise. Consider corpus-specific tuning."
    else:
        verdict = "**DO NOT PROCEED**. Centroid drift does not provide sufficient separability advantage over pairwise baseline."

    lines.append(f"\n{verdict}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Centroid Drift Diagnostics")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N dialogues per corpus (default: all)")
    parser.add_argument("--alphas", type=str, default="0.05,0.1,0.2",
                        help="Alpha values to test (comma-separated)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--corpora", type=str, default=None,
                        help="Comma-separated list of corpora to process (default: all)")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    alphas = [float(a) for a in args.alphas.split(",")]

    print("=" * 60)
    print("CENTROID DRIFT DIAGNOSTICS")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Embedding model: {args.model}")
    print(f"Alpha values: {alphas}")
    print(f"Sample size: {args.sample or 'all'}")

    # Initialize embedder
    embedder = EmbeddingProvider(args.model)

    # Select corpora
    if args.corpora:
        corpus_names = [c.strip() for c in args.corpora.split(",")]
    else:
        corpus_names = list(CORPUS_CONFIGS.keys())

    # Run diagnostics for each corpus
    all_results = []
    for corpus_name in corpus_names:
        if corpus_name not in CORPUS_CONFIGS:
            print(f"Warning: Unknown corpus '{corpus_name}', skipping")
            continue

        try:
            result = run_diagnostics(
                corpus_name,
                CORPUS_CONFIGS[corpus_name],
                project_root,
                embedder,
                alphas,
                args.sample
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {corpus_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results to report!")
        return

    # Format output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    table = format_table(all_results)
    notes = format_corpus_notes(all_results)
    recommendations = format_recommendations(all_results)

    full_report = table + notes + recommendations
    print(full_report)

    # Save to file
    output_path = args.output or (project_root / "paper/experiments/centroid_drift_results.md")
    with open(output_path, 'w') as f:
        f.write(full_report)
    print(f"\nResults saved to: {output_path}")

    # Also save JSON for programmatic access
    json_output = {
        "config": {
            "model": args.model,
            "alphas": alphas,
            "sample_size": args.sample
        },
        "results": []
    }

    for r in all_results:
        json_result = {
            "corpus": r.corpus,
            "stats": {
                "num_dialogues": r.stats.num_dialogues,
                "total_turns": r.stats.total_turns,
                "total_user_turns": r.stats.total_user_turns,
                "gold_boundaries": r.stats.gold_boundaries,
                "boundary_density": r.stats.boundary_density,
                "mean_segment_length": r.stats.mean_segment_length,
                "median_segment_length": r.stats.median_segment_length,
                "type": r.stats.type,
                "encoding": r.stats.encoding,
                "quirks": r.stats.quirks
            },
            "d1_results": {str(k): {"auroc": v.auroc, "boundary_median": v.boundary_median,
                                    "nonboundary_p90": v.nonboundary_p90}
                          for k, v in r.d1_results.items()},
            "d2_results": {str(k): {"auroc": v.auroc, "boundary_median": v.boundary_median,
                                    "nonboundary_p90": v.nonboundary_p90}
                          for k, v in r.d2_results.items()},
            "d3_results": {"auroc": r.d3_results.auroc, "boundary_median": r.d3_results.boundary_median,
                          "nonboundary_p90": r.d3_results.nonboundary_p90},
            "shuffled_auroc": r.shuffled_auroc,
            "best_alpha_d1": r.best_alpha_d1,
            "best_alpha_d2": r.best_alpha_d2,
            "separability_label": r.separability_label,
            "eligibility_curves": r.eligibility_curves
        }
        json_output["results"].append(json_result)

    json_path = str(output_path).replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
