#!/usr/bin/env python3
"""
Pairwise Eligibility Diagnostics for Eligibility-Gated Segmentation

Go/No-Go test for whether eligibility gating with pairwise drift is viable.

Measures:
- Eligibility rate: |E|/T
- Eligibility recall: |G ∩ E|/|G|
- Eligibility precision: |G ∩ E|/|E|

For each corpus, finds the best θ that achieves recall ≥ 0.8 with minimal eligibility rate.

Usage:
    python pairwise_eligibility_diagnostics.py [--output results.md]
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Corpus configurations (same as centroid diagnostics)
CORPUS_CONFIGS = {
    "superseg": {
        "path": "datasets/superseg/segmentation_file_test.json",
        "key": "superseg-v2",
        "boundary_encoding": "segmentation_label",
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
    is_boundary: bool = False
    embedding: Optional[np.ndarray] = None


@dataclass
class Dialogue:
    """A complete dialogue with turns."""
    dial_id: str
    turns: List[Turn]
    user_turns: List[Turn] = field(default_factory=list)

    def __post_init__(self):
        self.user_turns = [t for t in self.turns if t.role in ('user', 'User')]


@dataclass
class EligibilityResult:
    """Results for a single θ threshold."""
    theta: float
    eligibility_rate: float
    eligibility_recall: float
    eligibility_precision: float


@dataclass
class CorpusResult:
    """Complete results for a corpus."""
    corpus: str
    base_boundary_rate: float
    total_user_turns: int
    total_boundaries: int
    all_thresholds: List[EligibilityResult]
    best_theta: Optional[float]
    best_rate: Optional[float]
    best_recall: Optional[float]
    best_precision: Optional[float]
    max_recall: float
    max_recall_theta: float
    is_useful: bool
    notes: str


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

        to_compute = []
        to_compute_idx = []
        for i, text in enumerate(texts):
            if text not in self._cache:
                to_compute.append(text)
                to_compute_idx.append(i)

        if to_compute:
            embeddings = self.model.encode(to_compute, show_progress_bar=False, convert_to_numpy=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            for text, emb in zip(to_compute, embeddings):
                self._cache[text] = emb

        result = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        for i, text in enumerate(texts):
            result[i] = self._cache[text]

        return result


def load_corpus(corpus_name: str, config: dict, project_root: Path) -> List[Dialogue]:
    """Load a corpus."""
    file_path = project_root / config["path"]
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)
    key = config["key"]

    if key not in dial_data:
        available_keys = list(dial_data.keys())
        if len(available_keys) == 1:
            key = available_keys[0]
        else:
            raise KeyError(f"Key '{key}' not found. Available: {available_keys}")

    raw_dialogues = dial_data[key]
    dialogues = []
    encoding = config["boundary_encoding"]

    for dial in raw_dialogues:
        dial_id = dial.get("dial_id", "unknown")
        raw_turns = dial.get("turns", [])

        if not raw_turns:
            continue

        turns = []
        prev_topic_id = None

        for i, t in enumerate(raw_turns):
            role = t.get("role", "").lower()
            if role in ("agent", "assistant", "system"):
                role = "assistant"
            elif role in ("user",):
                role = "user"

            topic_id = t.get("topic_id", 0)

            if encoding == "segmentation_label":
                if i > 0 and raw_turns[i-1].get("segmentation_label", 0) == 1:
                    is_boundary = True
                else:
                    is_boundary = False
            else:
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

        dialogue = Dialogue(dial_id=dial_id, turns=turns)
        dialogues.append(dialogue)

    return dialogues


def compute_embeddings(dialogues: List[Dialogue], embedder: EmbeddingProvider) -> None:
    """Compute embeddings for all user turns."""
    all_texts = []
    text_to_turns = {}

    for dialogue in dialogues:
        for turn in dialogue.user_turns:
            if turn.utterance not in text_to_turns:
                text_to_turns[turn.utterance] = []
            text_to_turns[turn.utterance].append(turn)
            all_texts.append(turn.utterance)

    if not all_texts:
        return

    print(f"  Computing embeddings for {len(text_to_turns)} unique user turns...")

    unique_texts = list(text_to_turns.keys())
    embeddings = embedder.embed_batch(unique_texts)
    text_to_embedding = {text: emb for text, emb in zip(unique_texts, embeddings)}

    for text, turns in text_to_turns.items():
        for turn in turns:
            turn.embedding = text_to_embedding[text]


def compute_pairwise_drifts(dialogues: List[Dialogue]) -> Tuple[List[float], List[bool]]:
    """
    Compute pairwise drift for all user turns.
    d_t = 1 - cos(e_{t-1}, e_t)
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

            cos_sim = np.dot(prev_embedding, turn.embedding)
            drift = 1.0 - cos_sim

            all_drifts.append(drift)
            all_labels.append(turn.is_boundary)

            prev_embedding = turn.embedding

    return all_drifts, all_labels


def compute_eligibility_metrics(drifts: List[float], labels: List[bool],
                                 thresholds: List[float]) -> List[EligibilityResult]:
    """Compute eligibility metrics for each threshold."""
    drifts = np.array(drifts)
    labels = np.array(labels)

    total = len(drifts)
    total_boundaries = labels.sum()

    results = []

    for theta in thresholds:
        eligible = drifts >= theta
        eligible_count = eligible.sum()
        eligible_boundaries = (eligible & labels).sum()

        rate = eligible_count / total if total > 0 else 0
        recall = eligible_boundaries / total_boundaries if total_boundaries > 0 else 0
        precision = eligible_boundaries / eligible_count if eligible_count > 0 else 0

        results.append(EligibilityResult(
            theta=theta,
            eligibility_rate=rate,
            eligibility_recall=recall,
            eligibility_precision=precision
        ))

    return results


def find_best_threshold(results: List[EligibilityResult],
                        min_recall: float = 0.8) -> Tuple[Optional[EligibilityResult], EligibilityResult]:
    """
    Find best threshold that achieves recall >= min_recall with minimal eligibility rate.
    Returns: (best_result, max_recall_result)
    """
    # Find all thresholds with recall >= min_recall
    qualifying = [r for r in results if r.eligibility_recall >= min_recall]

    # Find max recall result
    max_recall_result = max(results, key=lambda r: r.eligibility_recall)

    if not qualifying:
        return None, max_recall_result

    # Among qualifying, find the one with minimum eligibility rate
    best = min(qualifying, key=lambda r: r.eligibility_rate)
    return best, max_recall_result


def run_corpus_analysis(corpus_name: str, config: dict, project_root: Path,
                        embedder: EmbeddingProvider) -> CorpusResult:
    """Run eligibility analysis for a single corpus."""

    print(f"\n{'='*60}")
    print(f"Processing corpus: {corpus_name}")
    print(f"{'='*60}")

    # Load corpus
    dialogues = load_corpus(corpus_name, config, project_root)

    # Count statistics
    total_user_turns = sum(len(d.user_turns) for d in dialogues)
    total_boundaries = sum(1 for d in dialogues for t in d.user_turns if t.is_boundary)
    base_rate = total_boundaries / total_user_turns if total_user_turns > 0 else 0

    print(f"  Loaded {len(dialogues)} dialogues, {total_user_turns} user turns, {total_boundaries} boundaries")
    print(f"  Base boundary rate: {base_rate:.4f}")

    # Compute embeddings
    compute_embeddings(dialogues, embedder)

    # Compute pairwise drifts
    drifts, labels = compute_pairwise_drifts(dialogues)
    print(f"  Computed {len(drifts)} pairwise drifts")

    # Define threshold grid
    fixed_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                        0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    # Add corpus-specific quantiles of non-boundary drifts
    nonboundary_drifts = [d for d, l in zip(drifts, labels) if not l]
    if nonboundary_drifts:
        quantile_thresholds = [
            np.percentile(nonboundary_drifts, 80),
            np.percentile(nonboundary_drifts, 90),
            np.percentile(nonboundary_drifts, 95)
        ]
    else:
        quantile_thresholds = []

    all_thresholds = sorted(set(fixed_thresholds + quantile_thresholds))

    # Compute eligibility metrics
    eligibility_results = compute_eligibility_metrics(drifts, labels, all_thresholds)

    # Find best threshold
    best_result, max_recall_result = find_best_threshold(eligibility_results, min_recall=0.8)

    # Determine if useful
    # Criteria: recall >= 0.8, rate <= 0.5 (or rate < 2*base_rate), precision > 1.5*base_rate
    is_useful = False
    notes = ""

    if best_result is not None:
        # Check if selective enough given base rate
        selectivity_ok = best_result.eligibility_rate <= 0.5 or best_result.eligibility_rate < 2 * base_rate
        precision_ok = best_result.eligibility_precision > 1.5 * base_rate

        if selectivity_ok and precision_ok:
            is_useful = True
            notes = f"θ={best_result.theta:.2f}: rate={best_result.eligibility_rate:.3f}, " \
                   f"recall={best_result.eligibility_recall:.3f}, precision={best_result.eligibility_precision:.3f}"
        else:
            notes = f"Recall achieved but not selective enough. " \
                   f"rate={best_result.eligibility_rate:.3f}, precision={best_result.eligibility_precision:.3f}"
    else:
        notes = f"Max recall = {max_recall_result.eligibility_recall:.3f} at θ={max_recall_result.theta:.2f}"

    print(f"  Best θ for recall≥0.8: {best_result.theta if best_result else 'N/A'}")
    print(f"  Max recall: {max_recall_result.eligibility_recall:.4f} at θ={max_recall_result.theta}")
    print(f"  Useful: {is_useful}")

    return CorpusResult(
        corpus=corpus_name,
        base_boundary_rate=base_rate,
        total_user_turns=total_user_turns,
        total_boundaries=total_boundaries,
        all_thresholds=eligibility_results,
        best_theta=best_result.theta if best_result else None,
        best_rate=best_result.eligibility_rate if best_result else None,
        best_recall=best_result.eligibility_recall if best_result else None,
        best_precision=best_result.eligibility_precision if best_result else None,
        max_recall=max_recall_result.eligibility_recall,
        max_recall_theta=max_recall_result.theta,
        is_useful=is_useful,
        notes=notes
    )


def format_results_table(results: List[CorpusResult]) -> str:
    """Format results as markdown table."""
    lines = []
    lines.append("## Pairwise Eligibility Diagnostics: Summary Table")
    lines.append("")
    lines.append("| Corpus | Base Rate | Best θ | Rate | Recall | Precision | Max Recall (θ) | Useful? |")
    lines.append("|--------|-----------|--------|------|--------|-----------|----------------|---------|")

    for r in results:
        if r.best_theta is not None:
            best_str = f"{r.best_theta:.2f}"
            rate_str = f"{r.best_rate:.3f}"
            recall_str = f"{r.best_recall:.3f}"
            prec_str = f"{r.best_precision:.3f}"
        else:
            best_str = "N/A"
            rate_str = "-"
            recall_str = "-"
            prec_str = "-"

        max_recall_str = f"{r.max_recall:.3f} ({r.max_recall_theta:.2f})"
        useful_str = "YES" if r.is_useful else "no"

        line = f"| {r.corpus} | {r.base_boundary_rate:.3f} | {best_str} | {rate_str} | {recall_str} | {prec_str} | {max_recall_str} | {useful_str} |"
        lines.append(line)

    return "\n".join(lines)


def format_detailed_notes(results: List[CorpusResult]) -> str:
    """Format detailed notes per corpus."""
    lines = []
    lines.append("\n## Per-Corpus Analysis")

    for r in results:
        lines.append(f"\n### {r.corpus}")
        lines.append(f"- **Base boundary rate**: {r.base_boundary_rate:.4f} ({r.total_boundaries}/{r.total_user_turns})")

        if r.best_theta is not None:
            lines.append(f"- **Best θ for recall≥0.8**: {r.best_theta:.2f}")
            lines.append(f"  - Eligibility rate: {r.best_rate:.3f}")
            lines.append(f"  - Recall: {r.best_recall:.3f}")
            lines.append(f"  - Precision: {r.best_precision:.3f}")
            lines.append(f"  - Precision / Base rate: {r.best_precision / r.base_boundary_rate:.2f}x")
        else:
            lines.append(f"- **No θ achieves recall≥0.8**")
            lines.append(f"  - Max recall: {r.max_recall:.3f} at θ={r.max_recall_theta:.2f}")

        lines.append(f"- **Useful eligibility band**: {'YES' if r.is_useful else 'NO'}")
        lines.append(f"- **Notes**: {r.notes}")

    return "\n".join(lines)


def format_conclusion(results: List[CorpusResult]) -> str:
    """Format go/no-go conclusion."""
    lines = []
    lines.append("\n## Go/No-Go Conclusion")

    useful_corpora = [r.corpus for r in results if r.is_useful]
    lines.append(f"\n**Useful eligibility bands found**: {len(useful_corpora)}/8")

    if len(useful_corpora) >= 3:
        lines.append(f"\n### VERDICT: ELIGIBILITY GATE PLAUSIBLE")
        lines.append(f"\nCorpora with useful eligibility bands: {', '.join(useful_corpora)}")
        lines.append(f"\nThese corpora show that pairwise drift can achieve recall≥0.8 with meaningful selectivity.")
    else:
        lines.append(f"\n### VERDICT: STOP - ELIGIBILITY GATE NOT SELECTIVE ENOUGH")
        lines.append(f"\nOnly {len(useful_corpora)} corpora have useful eligibility bands.")
        if useful_corpora:
            lines.append(f"Partial success in: {', '.join(useful_corpora)}")
        lines.append(f"\nThe pairwise drift signal does not provide sufficient selectivity for an eligibility gate.")

    # Detailed breakdown
    lines.append("\n### Breakdown by Corpus Type")

    task_oriented = [r for r in results if r.corpus in ["superseg", "dialseg711", "tiage", "multiwoz"]]
    open_domain = [r for r in results if r.corpus in ["dailydialog", "topical_chat"]]
    semi_structured = [r for r in results if r.corpus in ["taskmaster", "qmsum"]]

    for group_name, group in [("Task-oriented", task_oriented), ("Open-domain", open_domain), ("Semi-structured", semi_structured)]:
        useful_in_group = [r.corpus for r in group if r.is_useful]
        lines.append(f"\n**{group_name}**: {len(useful_in_group)}/{len(group)} useful")
        for r in group:
            status = "✓ useful" if r.is_useful else "✗ not useful"
            lines.append(f"  - {r.corpus}: {status}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pairwise Eligibility Diagnostics")
    parser.add_argument("--output", type=str, default=None, help="Output markdown file path")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model")
    args = parser.parse_args()

    project_root = PROJECT_ROOT

    print("=" * 60)
    print("PAIRWISE ELIGIBILITY DIAGNOSTICS")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Embedding model: {args.model}")

    embedder = EmbeddingProvider(args.model)

    all_results = []
    for corpus_name, config in CORPUS_CONFIGS.items():
        try:
            result = run_corpus_analysis(corpus_name, config, project_root, embedder)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {corpus_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results!")
        return

    # Format output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    table = format_results_table(all_results)
    notes = format_detailed_notes(all_results)
    conclusion = format_conclusion(all_results)

    full_report = table + notes + conclusion
    print(full_report)

    # Save
    output_path = args.output or (project_root / "paper/experiments/pairwise_eligibility_results.md")
    with open(output_path, 'w') as f:
        f.write(full_report)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
