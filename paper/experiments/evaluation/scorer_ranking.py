#!/usr/bin/env python3
"""
Threshold-free scorer evaluation using Average Precision (AP) and AUROC.

This module evaluates the boundary scorer's ranking quality independent of
any selection threshold, supporting the "scoring vs selection" decomposition
in the paper's evaluation framework.

Key functions reused from density_quality_curves.py:
- load_dataset(dataset_name) -> List[DialogueData]
  Returns dialogues with gold_boundaries: Set[int] (boundary positions where topic changes)

- get_neural_scores(dialogues) -> List[Dict[int, float]]
  Returns per-dialogue dicts mapping user_turn_idx -> confidence score
  Positions range from 1 to num_user_turns-1 (can't have boundary before first turn)

Usage:
    python -m paper.experiments.evaluation.scorer_ranking \
        --datasets dialseg711 superseg tiage \
        --out results/scorer_ranking.csv \
        --seed 0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import average_precision_score, roc_auc_score


def compute_dialogue_metrics(
    scores: Dict[int, float],
    gold_boundaries: set,
    num_user_turns: int
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Compute AP and AUROC for a single dialogue.

    Args:
        scores: Dict mapping position -> score (positions 1 to num_user_turns-1)
        gold_boundaries: Set of gold boundary positions
        num_user_turns: Total number of user turns in dialogue

    Returns:
        (ap, auroc, status) where status is one of:
        - "valid": metrics computed successfully
        - "no_positives": no gold boundaries (degenerate)
        - "all_positives": all positions are boundaries (degenerate)
        - "insufficient": fewer than 2 positions to evaluate
    """
    # Positions to evaluate: 1 to num_user_turns-1
    positions = list(range(1, num_user_turns))

    if len(positions) < 2:
        return None, None, "insufficient"

    # Build arrays
    y_true = np.array([1 if p in gold_boundaries else 0 for p in positions])
    y_scores = np.array([scores.get(p, 0.0) for p in positions])

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0:
        return None, None, "no_positives"
    if n_neg == 0:
        return None, None, "all_positives"

    # Compute metrics
    ap = average_precision_score(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    return ap, auroc, "valid"


def evaluate_dataset(
    dataset_name: str,
    seed: int = 0
) -> Dict:
    """
    Evaluate scorer ranking quality on a dataset.

    Args:
        dataset_name: One of dialseg711, superseg, tiage
        seed: Random seed (for reproducibility, though not used in deterministic metrics)

    Returns:
        Dict with keys:
        - dataset: dataset name
        - n_dialogues_total: total dialogues
        - n_used: dialogues with valid metrics
        - n_excl_no_pos: excluded for no positives
        - n_excl_all_pos: excluded for all positives
        - n_excl_insufficient: excluded for insufficient positions
        - mean_ap: macro-averaged AP
        - std_ap: std of per-dialogue AP
        - mean_auroc: macro-averaged AUROC
        - std_auroc: std of per-dialogue AUROC
    """
    # Import here to avoid circular imports and ensure proper path setup
    from paper.density_quality_curves import load_dataset, get_neural_scores

    np.random.seed(seed)

    # Load dataset and get scores
    # load_dataset() returns List[DialogueData] with:
    #   - messages: list of message dicts
    #   - gold_boundaries: Set[int] of boundary positions
    #   - num_messages: total message count
    dialogues = load_dataset(dataset_name)

    # get_neural_scores() returns List[Dict[int, float]]
    # mapping user_turn_idx -> confidence score for each dialogue
    all_scores = get_neural_scores(dialogues)

    # Compute per-dialogue metrics
    aps = []
    aurocs = []
    n_no_pos = 0
    n_all_pos = 0
    n_insufficient = 0

    for dialogue, scores in zip(dialogues, all_scores):
        # Count user turns (messages with role "user")
        num_user_turns = sum(1 for m in dialogue.messages if m.get("role") == "user")

        ap, auroc, status = compute_dialogue_metrics(
            scores, dialogue.gold_boundaries, num_user_turns
        )

        if status == "valid":
            aps.append(ap)
            aurocs.append(auroc)
        elif status == "no_positives":
            n_no_pos += 1
        elif status == "all_positives":
            n_all_pos += 1
        elif status == "insufficient":
            n_insufficient += 1

    n_total = len(dialogues)
    n_used = len(aps)

    return {
        "dataset": dataset_name,
        "n_dialogues_total": n_total,
        "n_used": n_used,
        "n_excl_no_pos": n_no_pos,
        "n_excl_all_pos": n_all_pos,
        "n_excl_insufficient": n_insufficient,
        "mean_ap": np.mean(aps) if aps else float("nan"),
        "std_ap": np.std(aps) if aps else float("nan"),
        "mean_auroc": np.mean(aurocs) if aurocs else float("nan"),
        "std_auroc": np.std(aurocs) if aurocs else float("nan"),
    }


def write_csv(results: List[Dict], output_path: Path) -> None:
    """Write results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "dataset", "n_dialogues_total", "n_used", "n_excl_no_pos",
        "n_excl_all_pos", "n_excl_insufficient", "mean_ap", "std_ap",
        "mean_auroc", "std_auroc"
    ]

    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [str(r[h]) for h in headers]
            f.write(",".join(row) + "\n")


def write_latex(results: List[Dict], output_path: Path) -> None:
    """Write results to LaTeX table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset name formatting
    name_map = {
        "dialseg711": "DialSeg711",
        "superseg": "SuperSeg",
        "tiage": "TIAGE",
    }

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & $N_\text{total}$ & $N_\text{used}$ & AP & AUROC \\",
        r"\midrule",
    ]

    for r in results:
        name = name_map.get(r["dataset"], r["dataset"])
        n_total = r["n_dialogues_total"]
        n_used = r["n_used"]
        ap = r["mean_ap"]
        auroc = r["mean_auroc"]
        std_ap = r["std_ap"]
        std_auroc = r["std_auroc"]

        # Format with std in parentheses
        ap_str = f"{ap:.3f} ({std_ap:.3f})" if not np.isnan(ap) else "---"
        auroc_str = f"{auroc:.3f} ({std_auroc:.3f})" if not np.isnan(auroc) else "---"

        lines.append(f"{name} & {n_total} & {n_used} & {ap_str} & {auroc_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Threshold-free scorer evaluation using AP and AUROC"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dialseg711", "superseg", "tiage"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/scorer_ranking.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    args = parser.parse_args()

    results = []
    for dataset in args.datasets:
        print(f"Evaluating {dataset}...")
        result = evaluate_dataset(dataset, seed=args.seed)
        results.append(result)

        print(f"  Total: {result['n_dialogues_total']}, Used: {result['n_used']}")
        print(f"  Excluded: no_pos={result['n_excl_no_pos']}, all_pos={result['n_excl_all_pos']}, insufficient={result['n_excl_insufficient']}")
        print(f"  AP: {result['mean_ap']:.4f} (±{result['std_ap']:.4f})")
        print(f"  AUROC: {result['mean_auroc']:.4f} (±{result['std_auroc']:.4f})")

    # Write outputs
    csv_path = Path(args.out)
    write_csv(results, csv_path)
    print(f"\nCSV written to {csv_path}")

    tex_path = csv_path.with_suffix(".tex")
    write_latex(results, tex_path)
    print(f"LaTeX written to {tex_path}")


if __name__ == "__main__":
    main()
