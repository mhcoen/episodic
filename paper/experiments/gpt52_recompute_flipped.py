#!/usr/bin/env python3
"""
GPT-5.2 Recompute with Flipped Score Polarity

Uses cached logprobs to recompute scores as:
    s_i = logP(N) - logP(Y)  # Flipped from original

This addresses the inverted scoring discovered in diagnostics where the model
interprets "Y" as "continue conversation" rather than "topic boundary".

Usage:
    python paper/experiments/gpt52_recompute_flipped.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
CACHE_FILE = PROJECT_ROOT / ".gpt52_cache" / "cache.json"
ORIGINAL_RESULTS = PROJECT_ROOT / "paper" / "experiments" / "gpt52_sanity_results.json"
OUTPUT_FILE = PROJECT_ROOT / "paper" / "experiments" / "gpt52_sanity_results_flipped.json"
DIAGNOSTICS_DIR = PROJECT_ROOT / "paper" / "experiments" / "gpt52_diagnostics_flipped"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Constants (from original script)
MIN_GAP = 2
TAU_PERCENTILES = [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
SPLIT_HALF_MAX_DEVIATION = 0.02


def load_cache() -> Dict:
    """Load the cached API responses."""
    with open(CACHE_FILE) as f:
        return json.load(f)


def load_original_results() -> Dict:
    """Load the original sanity check results."""
    with open(ORIGINAL_RESULTS) as f:
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


def flip_scores(cache: Dict) -> Dict:
    """
    Flip score polarity for all cache entries.
    Original: score = logprob_y - logprob_n
    Flipped:  score = logprob_n - logprob_y
    """
    flipped_cache = {}
    for key, entry in cache.items():
        flipped_entry = entry.copy()
        # Flip the score
        flipped_entry["score"] = entry["logprob_n"] - entry["logprob_y"]
        flipped_cache[key] = flipped_entry
    return flipped_cache


def extract_scores_by_dataset(cache: Dict) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Extract scores grouped by dataset and dialogue.
    Returns: {dataset: {dialogue_id: {position: score}}}
    """
    by_dataset = {"dialseg711": defaultdict(dict), "dailydialog": defaultdict(dict)}

    for key, entry in cache.items():
        if entry.get("missing_yn_in_toplogprobs") or entry.get("invalid_first_token"):
            continue

        parts = key.split("_")
        dataset = parts[0]

        if dataset in by_dataset:
            dialogue_id = entry["dialogue_id"]
            position = entry["position"]
            score = entry["score"]
            by_dataset[dataset][dialogue_id][position] = score

    return by_dataset


def greedy_nms_predict(scores_by_pos: Dict[int, float], tau: float, min_gap: int) -> Set[int]:
    """
    Greedy NMS prediction: select positions with score > tau, enforcing min_gap.
    """
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])  # Sort by score descending

    predicted = set()
    for pos, score in candidates:
        # Check if too close to any already-predicted boundary
        too_close = any(abs(pos - p) < min_gap for p in predicted)
        if not too_close:
            predicted.add(pos)

    return predicted


def compute_wf1_and_bor(predicted: Set[int], gold: Set[int], k: int = 3) -> Tuple[float, float, float, float]:
    """
    Compute W-F1, purity, coverage, and BOR.
    k: window size for boundary tolerance matching
    """
    if not gold:
        # No gold boundaries - can only compute BOR
        n_pred = len(predicted)
        return 0.0, 0.0, 0.0, float('inf') if n_pred > 0 else 1.0

    # Match predicted to gold with window tolerance
    matched_pred = set()
    matched_gold = set()

    for p in predicted:
        for g in gold:
            if abs(p - g) <= k and g not in matched_gold:
                matched_pred.add(p)
                matched_gold.add(g)
                break

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    wf1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Purity and coverage (from TopicTiling literature)
    purity = precision  # How many predictions are correct
    coverage = recall   # How many gold boundaries are found

    # BOR: Boundary Oversegmentation Ratio
    # BOR = predicted_boundaries / gold_boundaries
    bor = len(predicted) / len(gold) if gold else float('inf')

    return wf1, purity, coverage, bor


def compute_sweep(
    scores_by_dialogue: Dict[int, Dict[int, float]],
    gold_by_dialogue: Dict[int, Set[int]],
    tau_percentiles: List[int]
) -> List[Dict]:
    """
    Sweep tau thresholds and compute metrics at each point.
    """
    # Collect all scores to compute percentiles
    all_scores = []
    for dialogue_id, scores_by_pos in scores_by_dialogue.items():
        all_scores.extend(scores_by_pos.values())

    if not all_scores:
        return []

    all_scores = np.array(all_scores)

    sweep_points = []
    for pct in tau_percentiles:
        tau = np.percentile(all_scores, pct)

        # Predict for each dialogue
        total_wf1 = 0.0
        total_purity = 0.0
        total_coverage = 0.0
        total_pred = 0
        total_gold = 0
        n_dialogues = 0

        per_dialogue_wf1 = {}

        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            if not gold:
                continue

            predicted = greedy_nms_predict(scores_by_pos, tau, MIN_GAP)

            wf1, purity, coverage, _ = compute_wf1_and_bor(predicted, gold)

            per_dialogue_wf1[dialogue_id] = wf1
            total_wf1 += wf1
            total_purity += purity
            total_coverage += coverage
            total_pred += len(predicted)
            total_gold += len(gold)
            n_dialogues += 1

        if n_dialogues == 0:
            continue

        avg_wf1 = total_wf1 / n_dialogues
        avg_purity = total_purity / n_dialogues
        avg_coverage = total_coverage / n_dialogues
        bor = total_pred / total_gold if total_gold > 0 else 0

        sweep_points.append({
            "percentile": pct,
            "tau": float(tau),
            "wf1": avg_wf1,
            "purity": avg_purity,
            "coverage": avg_coverage,
            "bor": bor,
            "n_dialogues": n_dialogues,
            "total_pred": total_pred,
            "total_gold": total_gold,
            "per_dialogue_wf1": per_dialogue_wf1,
        })

    return sweep_points


def compute_split_half_deviation(sweep_points: List[Dict]) -> Tuple[float, List[Dict], List[Dict]]:
    """
    Compute split-half reliability by splitting dialogues and comparing curves.
    Returns: (max_deviation, half1_curve, half2_curve)
    """
    if not sweep_points:
        return 0.0, [], []

    # Get all dialogue IDs from the first sweep point
    all_dialogue_ids = list(sweep_points[0]["per_dialogue_wf1"].keys())

    if len(all_dialogue_ids) < 4:
        return 0.0, [], []

    # Split dialogues
    np.random.seed(42)
    shuffled = np.random.permutation(all_dialogue_ids)
    mid = len(shuffled) // 2
    half1_ids = set(shuffled[:mid])
    half2_ids = set(shuffled[mid:])

    half1_curve = []
    half2_curve = []

    for sp in sweep_points:
        per_dialogue = sp["per_dialogue_wf1"]

        # Half 1
        h1_wf1s = [wf1 for did, wf1 in per_dialogue.items() if did in half1_ids]
        h1_avg = np.mean(h1_wf1s) if h1_wf1s else 0

        # Half 2
        h2_wf1s = [wf1 for did, wf1 in per_dialogue.items() if did in half2_ids]
        h2_avg = np.mean(h2_wf1s) if h2_wf1s else 0

        half1_curve.append({"bor": sp["bor"], "wf1": h1_avg})
        half2_curve.append({"bor": sp["bor"], "wf1": h2_avg})

    # Compute max deviation
    max_dev = 0.0
    for h1, h2 in zip(half1_curve, half2_curve):
        dev = abs(h1["wf1"] - h2["wf1"])
        if dev > max_dev:
            max_dev = dev

    return max_dev, half1_curve, half2_curve


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


def create_plots(
    flipped_cache: Dict,
    gold_by_dataset: Dict,
    sweep_by_dataset: Dict,
    split_half_by_dataset: Dict
):
    """Create diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")
        return

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["dialseg711", "dailydialog"]:
        # Extract dataset entries
        entries = [e for key, e in flipped_cache.items() if key.startswith(dataset_name)]
        if not entries:
            continue

        gold = gold_by_dataset.get(dataset_name, {})

        # Extract valid scores and labels
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

        q10 = np.percentile(scores, 10)
        q50 = np.percentile(scores, 50)
        q90 = np.percentile(scores, 90)

        ax.axvline(q10, color='red', linestyle='--', label=f'q10={q10:.2f}')
        ax.axvline(q50, color='green', linestyle='--', label=f'q50={q50:.2f}')
        ax.axvline(q90, color='blue', linestyle='--', label=f'q90={q90:.2f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='s=0 (uncertain)')

        ax.set_xlabel('Score s_i = log P(N) - log P(Y)  [FLIPPED]')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset_name}: Score Distribution (n={len(scores)}) [FLIPPED]')
        ax.legend()

        fig.tight_layout()
        fig.savefig(DIAGNOSTICS_DIR / f'{dataset_name}_score_histogram.png', dpi=150)
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

        for i, (data, color) in enumerate([(non_gold_scores, 'blue'), (gold_scores, 'red')]):
            jitter = np.random.normal(0, 0.04, len(data))
            ax.scatter(i + 1 + jitter, data, alpha=0.3, s=10, c=color)

        ax.set_ylabel('Score s_i [FLIPPED]')
        ax.set_title(f'{dataset_name}: Score by Gold Label [FLIPPED]')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        if len(gold_scores) > 0 and len(non_gold_scores) > 0:
            auroc = compute_auroc(scores.tolist(), labels.tolist())
            cohens_d = compute_cohens_d(gold_scores.tolist(), non_gold_scores.tolist())
            ax.text(0.02, 0.98, f"AUC-ROC: {auroc:.3f}\nCohen's d: {cohens_d:.3f}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.tight_layout()
        fig.savefig(DIAGNOSTICS_DIR / f'{dataset_name}_score_vs_gold.png', dpi=150)
        plt.close(fig)

        # ============================================================
        # 3. W-F1 vs BOR Curve
        # ============================================================
        sweep_points = sweep_by_dataset.get(dataset_name, [])
        if sweep_points:
            fig, ax = plt.subplots(figsize=(10, 6))

            bors = [sp["bor"] for sp in sweep_points]
            wf1s = [sp["wf1"] for sp in sweep_points]

            ax.plot(bors, wf1s, 'b-', linewidth=2, label='Full Sample', marker='o', markersize=3)

            # Plot split-half curves if available
            split_half = split_half_by_dataset.get(dataset_name, {})
            h1_curve = split_half.get("half1_curve", [])
            h2_curve = split_half.get("half2_curve", [])

            if h1_curve and h2_curve:
                ax.plot([p["bor"] for p in h1_curve], [p["wf1"] for p in h1_curve],
                        'g--', linewidth=1, alpha=0.7, label='Half 1')
                ax.plot([p["bor"] for p in h2_curve], [p["wf1"] for p in h2_curve],
                        'r--', linewidth=1, alpha=0.7, label='Half 2')

            ax.set_xlabel('BOR (Boundary Oversegmentation Ratio)')
            ax.set_ylabel('W-F1')
            max_dev = split_half.get("max_deviation", 0)
            ax.set_title(f'{dataset_name}: W-F1 vs BOR [FLIPPED]\nSplit-half max deviation: {max_dev:.4f}')
            ax.grid(True, alpha=0.3)
            ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='BOR=1')
            ax.legend()

            fig.tight_layout()
            fig.savefig(DIAGNOSTICS_DIR / f'{dataset_name}_wf1_vs_bor.png', dpi=150)
            plt.close(fig)


def main():
    print("=" * 60)
    print("GPT-5.2 RECOMPUTE WITH FLIPPED SCORE POLARITY")
    print("=" * 60)

    # Load cached data
    print("\nLoading cached data...")
    cache = load_cache()
    print(f"  Cache entries: {len(cache)}")

    # Flip scores
    print("\nFlipping score polarity (s_i = logP(N) - logP(Y))...")
    flipped_cache = flip_scores(cache)

    # Verify flipping
    sample_key = list(cache.keys())[0]
    orig_score = cache[sample_key]["score"]
    flipped_score = flipped_cache[sample_key]["score"]
    print(f"  Sample: original={orig_score:.3f} -> flipped={flipped_score:.3f}")

    # Load gold boundaries
    print("\nLoading gold boundaries...")
    gold_by_dataset = {}
    for dataset in ["dialseg711", "dailydialog"]:
        gold_by_dataset[dataset] = load_gold_boundaries(dataset)
        n_gold = sum(len(g) for g in gold_by_dataset[dataset].values())
        print(f"  {dataset}: {n_gold} gold boundaries across {len(gold_by_dataset[dataset])} dialogues")

    # Extract scores by dataset
    scores_by_dataset = extract_scores_by_dataset(flipped_cache)

    # ============================================================
    # Compute metrics for each dataset
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPUTING METRICS WITH FLIPPED SCORES")
    print("=" * 60)

    results = {
        "description": "GPT-5.2 sanity check with FLIPPED score polarity",
        "score_formula": "s_i = logP(N) - logP(Y)",
        "datasets": {}
    }

    sweep_by_dataset = {}
    split_half_by_dataset = {}

    for dataset_name in ["dialseg711", "dailydialog"]:
        print(f"\n{dataset_name}:")

        scores_by_dialogue = scores_by_dataset[dataset_name]
        gold_by_dialogue = gold_by_dataset[dataset_name]

        # Compute sweep
        sweep_points = compute_sweep(dict(scores_by_dialogue), gold_by_dialogue, TAU_PERCENTILES)
        sweep_by_dataset[dataset_name] = sweep_points

        if not sweep_points:
            print("  No sweep points computed")
            continue

        # Find best W-F1
        best_point = max(sweep_points, key=lambda x: x["wf1"])
        print(f"  Best W-F1: {best_point['wf1']:.4f} at tau={best_point['tau']:.3f} (BOR={best_point['bor']:.3f})")
        print(f"  Purity: {best_point['purity']:.4f}, Coverage: {best_point['coverage']:.4f}")

        # Compute split-half
        max_dev, h1_curve, h2_curve = compute_split_half_deviation(sweep_points)
        split_half_by_dataset[dataset_name] = {
            "max_deviation": max_dev,
            "half1_curve": h1_curve,
            "half2_curve": h2_curve
        }
        print(f"  Split-half max deviation: {max_dev:.4f}")

        # Score separation analysis
        gold_scores = []
        non_gold_scores = []
        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            for pos, score in scores_by_pos.items():
                if pos in gold:
                    gold_scores.append(score)
                else:
                    non_gold_scores.append(score)

        if gold_scores and non_gold_scores:
            all_scores = gold_scores + non_gold_scores
            all_labels = [True] * len(gold_scores) + [False] * len(non_gold_scores)
            auroc = compute_auroc(all_scores, all_labels)
            cohens_d = compute_cohens_d(gold_scores, non_gold_scores)

            print(f"  Gold mean: {np.mean(gold_scores):.3f}, Non-gold mean: {np.mean(non_gold_scores):.3f}")
            print(f"  AUC-ROC: {auroc:.4f}")
            print(f"  Cohen's d: {cohens_d:.4f}")
        else:
            auroc = 0.5
            cohens_d = 0.0

        # Store results
        results["datasets"][dataset_name] = {
            "best_wf1": best_point["wf1"],
            "best_tau": best_point["tau"],
            "best_bor": best_point["bor"],
            "best_purity": best_point["purity"],
            "best_coverage": best_point["coverage"],
            "split_half_max_deviation": max_dev,
            "auroc": auroc,
            "cohens_d": cohens_d,
            "n_dialogues": len(scores_by_dialogue),
            "n_gold_boundaries": sum(len(g) for g in gold_by_dialogue.values()),
            "sweep_points": [{k: v for k, v in sp.items() if k != "per_dialogue_wf1"} for sp in sweep_points],
        }

    # ============================================================
    # Acceptance Criteria
    # ============================================================
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA (FLIPPED)")
    print("=" * 60)

    criteria = {
        "split_half_passed": True,
        "auc_roc_passed": True,
        "cohens_d_passed": True,
        "wf1_reasonable": True,
    }

    for dataset_name, dr in results["datasets"].items():
        print(f"\n{dataset_name}:")

        # Split-half: max deviation < 0.02
        sh_pass = dr["split_half_max_deviation"] < SPLIT_HALF_MAX_DEVIATION
        criteria["split_half_passed"] &= sh_pass
        status = "PASS" if sh_pass else "FAIL"
        print(f"  Split-half (dev < 0.02): {dr['split_half_max_deviation']:.4f} [{status}]")

        # AUC-ROC > 0.6
        auc_pass = dr["auroc"] > 0.6
        criteria["auc_roc_passed"] &= auc_pass
        status = "PASS" if auc_pass else "FAIL"
        print(f"  AUC-ROC (> 0.6): {dr['auroc']:.4f} [{status}]")

        # Cohen's d > 0.3 (positive, meaning gold > non-gold)
        cd_pass = dr["cohens_d"] > 0.3
        criteria["cohens_d_passed"] &= cd_pass
        status = "PASS" if cd_pass else "FAIL"
        print(f"  Cohen's d (> 0.3): {dr['cohens_d']:.4f} [{status}]")

        # W-F1 > 0.3
        wf1_pass = dr["best_wf1"] > 0.3
        criteria["wf1_reasonable"] &= wf1_pass
        status = "PASS" if wf1_pass else "FAIL"
        print(f"  W-F1 (> 0.3): {dr['best_wf1']:.4f} [{status}]")

    # Convert numpy bools to Python bools for JSON serialization
    results["criteria"] = {k: bool(v) for k, v in criteria.items()}
    all_passed = all(criteria.values())
    results["all_passed"] = bool(all_passed)

    print("\n" + "=" * 60)
    overall = "ALL CRITERIA PASSED" if all_passed else "SOME CRITERIA FAILED"
    print(f"OVERALL: {overall}")
    print("=" * 60)

    # Save results
    print(f"\nSaving results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Create plots
    print(f"\nCreating diagnostic plots in: {DIAGNOSTICS_DIR}")
    create_plots(flipped_cache, gold_by_dataset, sweep_by_dataset, split_half_by_dataset)

    print("\nDone!")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
