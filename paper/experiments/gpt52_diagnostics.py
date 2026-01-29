#!/usr/bin/env python3
"""
GPT-5.2 Boundary Scorer Diagnostics

Analyzes the split-half reliability failure from the sanity check run.
Uses cached score data and sanity results.

Usage:
    python paper/experiments/gpt52_diagnostics.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
CACHE_FILE = PROJECT_ROOT / ".gpt52_cache" / "cache.json"
RESULTS_FILE = PROJECT_ROOT / "paper" / "experiments" / "gpt52_sanity_results.json"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments" / "gpt52_diagnostics"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_cache() -> Dict:
    """Load the cached API responses."""
    with open(CACHE_FILE) as f:
        return json.load(f)


def load_results() -> Dict:
    """Load the sanity check results."""
    with open(RESULTS_FILE) as f:
        return json.load(f)


def load_gold_boundaries(dataset_name: str) -> Dict[int, Set[int]]:
    """
    Load gold boundaries for a dataset.
    Returns: Dict mapping dialogue_id -> set of gold boundary positions
    """
    test_file = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    with open(test_file) as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)
    gold_by_dialogue = {}
    dialogue_id = 0

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            boundaries = set()
            prev_topic = None
            user_idx = 0

            for turn in turns:
                if turn.get("role") == "user":
                    topic = turn.get("topic_id") or turn.get("topic_name")
                    if prev_topic is not None and topic != prev_topic:
                        boundaries.add(user_idx)
                    prev_topic = topic
                    user_idx += 1

            gold_by_dialogue[dialogue_id] = boundaries
            dialogue_id += 1

    return gold_by_dialogue


def extract_scores_by_dataset(cache: Dict) -> Dict[str, List[Dict]]:
    """Extract scores grouped by dataset."""
    by_dataset = {"dialseg711": [], "dailydialog": []}

    for key, entry in cache.items():
        # Parse key format: dataset_dialogueId_position_hash
        parts = key.split("_")
        dataset = parts[0]

        if dataset in by_dataset:
            by_dataset[dataset].append(entry)

    return by_dataset


def compute_score_stats(scores: List[float]) -> Dict:
    """Compute score distribution statistics."""
    scores = np.array(scores)
    return {
        "n": len(scores),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "q10": float(np.percentile(scores, 10)),
        "q25": float(np.percentile(scores, 25)),
        "q50": float(np.percentile(scores, 50)),
        "q75": float(np.percentile(scores, 75)),
        "q90": float(np.percentile(scores, 90)),
    }


def compute_auroc(scores: List[float], labels: List[bool]) -> float:
    """Compute AUROC for binary classification."""
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def create_plots(scores_by_dataset: Dict, gold_by_dataset: Dict, results: Dict):
    """Create diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, entries in scores_by_dataset.items():
        if not entries:
            continue

        gold = gold_by_dataset.get(dataset_name, {})

        # Extract scores and labels
        scores = []
        labels = []

        for entry in entries:
            if entry.get("missing_yn_in_toplogprobs") or entry.get("invalid_first_token"):
                continue

            score = entry["score"]
            dialogue_id = entry["dialogue_id"]
            position = entry["position"]

            is_gold = position in gold.get(dialogue_id, set())

            scores.append(score)
            labels.append(is_gold)

        scores = np.array(scores)
        labels = np.array(labels)

        # ============================================================
        # 1. Score Histogram
        # ============================================================
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)

        # Add quantile lines
        q10 = np.percentile(scores, 10)
        q50 = np.percentile(scores, 50)
        q90 = np.percentile(scores, 90)

        ax.axvline(q10, color='red', linestyle='--', label=f'q10={q10:.2f}')
        ax.axvline(q50, color='green', linestyle='--', label=f'q50={q50:.2f}')
        ax.axvline(q90, color='blue', linestyle='--', label=f'q90={q90:.2f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='s=0 (uncertain)')

        ax.set_xlabel('Score s_i = log P(Y) - log P(N)')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset_name}: Score Distribution (n={len(scores)})')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f'{dataset_name}_score_histogram.png', dpi=150)
        plt.close(fig)

        # ============================================================
        # 2. Score vs Gold Label (Box Plot)
        # ============================================================
        gold_scores = scores[labels]
        non_gold_scores = scores[~labels]

        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot([non_gold_scores, gold_scores],
                        labels=['Non-Gold', 'Gold'],
                        patch_artist=True)

        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        # Add individual points (jittered)
        for i, (data, color) in enumerate([(non_gold_scores, 'blue'), (gold_scores, 'red')]):
            jitter = np.random.normal(0, 0.04, len(data))
            ax.scatter(i + 1 + jitter, data, alpha=0.3, s=10, c=color)

        ax.set_ylabel('Score s_i')
        ax.set_title(f'{dataset_name}: Score by Gold Label')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Compute and display metrics
        if len(gold_scores) > 0 and len(non_gold_scores) > 0:
            auroc = compute_auroc(scores.tolist(), labels.tolist())
            cohens_d = compute_cohens_d(gold_scores.tolist(), non_gold_scores.tolist())
            ax.text(0.02, 0.98, f"AUC-ROC: {auroc:.3f}\nCohen's d: {cohens_d:.3f}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f'{dataset_name}_score_vs_gold.png', dpi=150)
        plt.close(fig)

        # ============================================================
        # 3. Split-Half Curve Overlay
        # ============================================================
        # Get sweep points from results
        dataset_results = results.get("datasets", {}).get(dataset_name, {})
        sweep_points = dataset_results.get("sweep_points", [])

        if sweep_points:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Full sample curve
            bors = [sp["bor"] for sp in sweep_points]
            wf1s = [sp["wf1"] for sp in sweep_points]

            ax.plot(bors, wf1s, 'b-', linewidth=2, label='Full Sample', marker='o', markersize=3)

            # We need to recompute split-half curves using the per_dialogue_wf1 data
            # For now, just show the full curve and note the split-half deviation

            ax.set_xlabel('BOR (Boundary Oversegmentation Ratio)')
            ax.set_ylabel('W-F1')
            ax.set_title(f'{dataset_name}: W-F1 vs BOR\nSplit-half max deviation: {dataset_results.get("split_half_max_deviation", "N/A"):.4f}')
            ax.grid(True, alpha=0.3)
            ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='BOR=1')
            ax.legend()

            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f'{dataset_name}_wf1_vs_bor.png', dpi=150)
            plt.close(fig)


def main():
    print("=" * 60)
    print("GPT-5.2 BOUNDARY SCORER DIAGNOSTICS")
    print("=" * 60)

    # Load data
    print("\nLoading cached data...")
    cache = load_cache()
    results = load_results()

    print(f"  Cache entries: {len(cache)}")

    # Extract scores by dataset
    scores_by_dataset = extract_scores_by_dataset(cache)

    # Load gold boundaries
    print("\nLoading gold boundaries...")
    gold_by_dataset = {}
    for dataset in ["dialseg711", "dailydialog"]:
        gold_by_dataset[dataset] = load_gold_boundaries(dataset)
        print(f"  {dataset}: {sum(len(g) for g in gold_by_dataset[dataset].values())} gold boundaries")

    # ============================================================
    # Diagnostic 1: Score Histogram Statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("1. SCORE DISTRIBUTION (per dataset)")
    print("=" * 60)

    for dataset, entries in scores_by_dataset.items():
        valid_scores = [e["score"] for e in entries
                       if not e.get("missing_yn_in_toplogprobs")
                       and not e.get("invalid_first_token")]

        if not valid_scores:
            print(f"\n{dataset}: No valid scores")
            continue

        stats = compute_score_stats(valid_scores)

        print(f"\n{dataset}:")
        print(f"  N valid scores: {stats['n']}")
        print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        print(f"  Quantiles: q10={stats['q10']:.2f}, q50={stats['q50']:.2f}, q90={stats['q90']:.2f}")

    # ============================================================
    # Diagnostic 2: Score vs Gold Label
    # ============================================================
    print("\n" + "=" * 60)
    print("2. SCORE vs GOLD LABEL (separation analysis)")
    print("=" * 60)

    for dataset, entries in scores_by_dataset.items():
        gold = gold_by_dataset.get(dataset, {})

        gold_scores = []
        non_gold_scores = []

        for entry in entries:
            if entry.get("missing_yn_in_toplogprobs") or entry.get("invalid_first_token"):
                continue

            score = entry["score"]
            dialogue_id = entry["dialogue_id"]
            position = entry["position"]

            is_gold = position in gold.get(dialogue_id, set())

            if is_gold:
                gold_scores.append(score)
            else:
                non_gold_scores.append(score)

        print(f"\n{dataset}:")
        print(f"  Gold boundaries scored: {len(gold_scores)}")
        print(f"  Non-gold boundaries scored: {len(non_gold_scores)}")

        if gold_scores and non_gold_scores:
            print(f"  Gold mean: {np.mean(gold_scores):.3f} (std: {np.std(gold_scores):.3f})")
            print(f"  Non-gold mean: {np.mean(non_gold_scores):.3f} (std: {np.std(non_gold_scores):.3f})")

            all_scores = gold_scores + non_gold_scores
            all_labels = [True] * len(gold_scores) + [False] * len(non_gold_scores)

            auroc = compute_auroc(all_scores, all_labels)
            cohens_d = compute_cohens_d(gold_scores, non_gold_scores)

            print(f"  AUC-ROC: {auroc:.4f}")
            print(f"  Cohen's d: {cohens_d:.4f}")

            if auroc < 0.55:
                print(f"  ⚠️  VERY POOR: AUC-ROC near random (0.5)")
            elif auroc < 0.6:
                print(f"  ⚠️  POOR: Minimal separation")
            elif auroc < 0.7:
                print(f"  MARGINAL: Some separation")
            else:
                print(f"  OK: Reasonable separation")

    # ============================================================
    # Diagnostic 4: Score Entropy/Calibration
    # ============================================================
    print("\n" + "=" * 60)
    print("4. SCORE ENTROPY/CALIBRATION CHECK")
    print("=" * 60)

    for dataset, entries in scores_by_dataset.items():
        valid_scores = [e["score"] for e in entries
                       if not e.get("missing_yn_in_toplogprobs")
                       and not e.get("invalid_first_token")]

        if not valid_scores:
            continue

        scores = np.array(valid_scores)
        n_total = len(scores)

        # Near-uncertain: |s| < 0.5
        n_uncertain = np.sum(np.abs(scores) < 0.5)
        pct_uncertain = 100 * n_uncertain / n_total

        # Confident: |s| > 2.0
        n_confident = np.sum(np.abs(scores) > 2.0)
        pct_confident = 100 * n_confident / n_total

        # Very confident: |s| > 5.0
        n_very_confident = np.sum(np.abs(scores) > 5.0)
        pct_very_confident = 100 * n_very_confident / n_total

        # Positive (Y more likely): s > 0
        n_positive = np.sum(scores > 0)
        pct_positive = 100 * n_positive / n_total

        print(f"\n{dataset}:")
        print(f"  Near-uncertain (|s| < 0.5): {n_uncertain}/{n_total} ({pct_uncertain:.1f}%)")
        print(f"  Confident (|s| > 2.0): {n_confident}/{n_total} ({pct_confident:.1f}%)")
        print(f"  Very confident (|s| > 5.0): {n_very_confident}/{n_total} ({pct_very_confident:.1f}%)")
        print(f"  Positive (Y favored, s > 0): {n_positive}/{n_total} ({pct_positive:.1f}%)")

        if pct_confident < 50:
            print(f"  ⚠️  LOW CONFIDENCE: Model is hedging on most boundaries")

        if pct_positive < 10 or pct_positive > 90:
            print(f"  ⚠️  BIASED: Model strongly favors {'N' if pct_positive < 10 else 'Y'}")

    # ============================================================
    # Create plots
    # ============================================================
    print("\n" + "=" * 60)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 60)

    create_plots(scores_by_dataset, gold_by_dataset, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots saved to: {OUTPUT_DIR}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
The split-half reliability failure is likely caused by:

1. MODEL BIAS: Check if model strongly favors Y or N across all boundaries
   (If >90% predictions lean one way, model isn't discriminating)

2. POOR SEPARATION: If AUC-ROC is near 0.5, model can't distinguish
   gold from non-gold boundaries

3. HIGH VARIANCE: If score distributions are wide and overlapping,
   the threshold sweep produces unstable curves

Key questions:
- Does the score distribution show any discriminative structure?
- Are gold boundaries actually receiving higher scores on average?
- Is the model calibrated (confident when correct, uncertain otherwise)?
""")


if __name__ == "__main__":
    main()
