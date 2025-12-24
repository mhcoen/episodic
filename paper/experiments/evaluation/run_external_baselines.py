#!/usr/bin/env python3
"""
Run external baseline segmenters and evaluate under unified framework.

Usage:
    python -m tacl.experiments.evaluation.run_external_baselines \
        --methods texttiling csm_nsp random even \
        --datasets dialseg711 \
        --out results/external_methods.csv

This script:
1. Loads datasets in canonical format (boundaries indexed by message position)
2. Runs each segmentation method
3. Computes metrics: Strict F1, W-F1, BOR, Purity, Coverage
4. Outputs CSV and LaTeX table fragments
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EvalResult:
    """Evaluation result for one method on one dataset."""
    method: str
    dataset: str
    n_dialogues: int
    total_gold: int
    total_pred: int
    total_tp: int
    strict_f1: float
    strict_prec: float
    strict_rec: float
    w_f1: float
    w_prec: float
    w_rec: float
    bor: float
    purity: float
    coverage: float


def load_csm_format_data(filepath: str) -> List[Dict]:
    """
    Load CSM-format dataset (used by Dialogue-Topic-Segmenter repo).

    Format: List of dicts with 'utterances', 'segments', 'set' keys.
    """
    with open(filepath) as f:
        return json.load(f)


def load_canonical_format_data(filepath: str) -> List[Dict]:
    """
    Load canonical format dataset (our preprocessing output).

    Format: List of dicts with 'messages', 'gold_boundaries', 'dialogue_id' keys.
    """
    with open(filepath) as f:
        return json.load(f)


def csm_segments_to_boundaries(segment_sizes: List[int]) -> Set[int]:
    """
    Convert CSM segment sizes to canonical boundary indices.

    CSM format: segments = [4, 6, 3] means 3 segments of those sizes.
    Boundaries are at cumulative positions: 4, 10 (4+6).

    Returns canonical boundary set (boundary at t means new segment starts at message t).
    """
    boundaries = set()
    cumsum = 0
    for size in segment_sizes[:-1]:  # No boundary after last segment
        cumsum += size
        boundaries.add(cumsum)  # Canonical: boundary at start of new segment
    return boundaries


def boundaries_to_segments(boundaries: Set[int], num_messages: int) -> List[Set[int]]:
    """
    Convert boundary positions to segment membership sets.

    Used for purity/coverage computation.
    """
    sorted_bounds = sorted(boundaries)
    segments = []
    prev = 0

    for bound in sorted_bounds:
        if bound > prev:
            segments.append(set(range(prev, bound)))
        prev = bound

    if prev < num_messages:
        segments.append(set(range(prev, num_messages)))

    return segments if segments else [set(range(num_messages))]


def compute_windowed_metrics(
    gold: Set[int],
    pred: Set[int],
    num_messages: int,
    window: int = 1
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 with tolerance window."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0

    matched_pred = set()
    matched_gold = set()

    for p in pred:
        for g in gold:
            if abs(p - g) <= window:
                matched_pred.add(p)
                matched_gold.add(g)
                break

    precision = len(matched_pred) / len(pred) if pred else 0.0
    recall = len(matched_gold) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_purity_coverage(
    gold_segments: List[Set[int]],
    pred_segments: List[Set[int]]
) -> Tuple[float, float]:
    """Compute segment purity and coverage."""
    if not gold_segments or not pred_segments:
        return 0.0, 0.0

    purities = []
    for pred_seg in pred_segments:
        if pred_seg:
            max_overlap = max(len(pred_seg & gold_seg) for gold_seg in gold_segments)
            purities.append(max_overlap / len(pred_seg))

    coverages = []
    for gold_seg in gold_segments:
        if gold_seg:
            max_overlap = max(len(gold_seg & pred_seg) for pred_seg in pred_segments)
            coverages.append(max_overlap / len(gold_seg))

    purity = sum(purities) / len(purities) if purities else 0.0
    coverage = sum(coverages) / len(coverages) if coverages else 0.0

    return purity, coverage


def compute_bor(num_gold: int, num_pred: int) -> float:
    """Compute Boundary Oversegmentation Ratio."""
    if num_gold == 0:
        return float('inf') if num_pred > 0 else 1.0
    return num_pred / num_gold


def evaluate_predictions(
    method_name: str,
    dataset_name: str,
    predictions: List[Set[int]],
    gold_boundaries: List[Set[int]],
    num_messages_list: List[int],
) -> EvalResult:
    """
    Evaluate predictions against gold boundaries.

    Args:
        method_name: Name of the method
        dataset_name: Name of the dataset
        predictions: List of predicted boundary sets
        gold_boundaries: List of gold boundary sets
        num_messages_list: List of dialogue lengths

    Returns:
        EvalResult with all metrics
    """
    n = len(predictions)

    # Accumulators
    total_gold = 0
    total_pred = 0
    total_tp = 0

    # Per-dialogue metrics for macro-averaging
    strict_f1_sum = 0.0
    w_f1_sum = 0.0
    w_prec_sum = 0.0
    w_rec_sum = 0.0
    purity_sum = 0.0
    coverage_sum = 0.0

    for pred, gold, num_msg in zip(predictions, gold_boundaries, num_messages_list):
        # Strict F1
        tp = len(pred & gold)
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(gold) if gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        strict_f1_sum += f1

        # Windowed metrics
        w_prec, w_rec, w_f1 = compute_windowed_metrics(gold, pred, num_msg, window=1)
        w_f1_sum += w_f1
        w_prec_sum += w_prec
        w_rec_sum += w_rec

        # Purity/coverage
        gold_segs = boundaries_to_segments(gold, num_msg)
        pred_segs = boundaries_to_segments(pred, num_msg)
        purity, coverage = compute_purity_coverage(gold_segs, pred_segs)
        purity_sum += purity
        coverage_sum += coverage

        # Totals
        total_gold += len(gold)
        total_pred += len(pred)
        total_tp += tp

    # Micro-averaged strict metrics
    micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
    micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return EvalResult(
        method=method_name,
        dataset=dataset_name,
        n_dialogues=n,
        total_gold=total_gold,
        total_pred=total_pred,
        total_tp=total_tp,
        strict_f1=micro_f1,
        strict_prec=micro_prec,
        strict_rec=micro_rec,
        w_f1=w_f1_sum / n,
        w_prec=w_prec_sum / n,
        w_rec=w_rec_sum / n,
        bor=compute_bor(total_gold, total_pred),
        purity=purity_sum / n,
        coverage=coverage_sum / n,
    )


def get_segmenter(method: str, **kwargs):
    """Get segmenter instance by name."""
    from tacl.experiments.segmenters import (
        TextTilingSegmenter,
        CSMSegmenter,
        RandomSegmenter,
        EvenSegmenter,
    )

    if method == "texttiling":
        return TextTilingSegmenter(alpha=kwargs.get("alpha", 0.0))
    elif method == "csm_nsp":
        return CSMSegmenter(alpha=kwargs.get("alpha", 0.0))
    elif method == "random":
        return RandomSegmenter(target_ratio=0.1, seed=42)
    elif method == "even":
        return EvenSegmenter(match_gold=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_segmenter_on_dataset(
    segmenter,
    dialogues: List[Dict],
    dataset_format: str = "canonical",
    use_dev_for_alpha: bool = True,
) -> Tuple[List[Set[int]], List[Set[int]], List[int]]:
    """
    Run segmenter on a dataset.

    Args:
        segmenter: Segmenter instance
        dialogues: List of dialogue dicts
        dataset_format: "canonical" or "csm"
        use_dev_for_alpha: If True, tune alpha on dev set (for TextTiling/CSM)

    Returns:
        (predictions, gold_boundaries, num_messages_list)
    """
    predictions = []
    gold_boundaries = []
    num_messages_list = []

    # Separate dev and test if present
    if dataset_format == "csm":
        dev_data = [d for d in dialogues if d.get('set') == 'dev']
        test_data = [d for d in dialogues if d.get('set') != 'dev']
    else:
        # For canonical format, use first 10% as dev if no split marker
        split_idx = max(1, len(dialogues) // 10)
        dev_data = dialogues[:split_idx]
        test_data = dialogues[split_idx:]

    # Alpha tuning for TextTiling/CSM
    if use_dev_for_alpha and hasattr(segmenter, 'find_best_alpha') and dev_data:
        print(f"  Tuning alpha on {len(dev_data)} dev dialogues...")

        if dataset_format == "csm":
            dev_messages = [
                [{"role": "user", "content": u} for u in d["utterances"]]
                for d in dev_data
            ]
            dev_gold = [
                list(csm_segments_to_boundaries(d["segments"]))
                for d in dev_data
            ]
        else:
            dev_messages = [d["messages"] for d in dev_data]
            dev_gold = [d["gold_boundaries"] for d in dev_data]

        best_alpha, best_score = segmenter.find_best_alpha(dev_messages, dev_gold)
        print(f"  Best alpha: {best_alpha:.2f} (score: {best_score:.3f})")
        segmenter.alpha = best_alpha

    # Run on test set
    for dialogue in tqdm(test_data, desc=f"  Running {segmenter.name}"):
        if dataset_format == "csm":
            messages = [{"role": "user", "content": u} for u in dialogue["utterances"]]
            gold = csm_segments_to_boundaries(dialogue["segments"])
            num_msg = len(dialogue["utterances"])
        else:
            messages = dialogue["messages"]
            gold = set(dialogue["gold_boundaries"])
            num_msg = len(messages)

        # Get prediction
        if hasattr(segmenter, 'predict_boundaries'):
            # Pass gold count for EvenSegmenter
            result = segmenter.predict_boundaries(
                messages,
                num_gold_boundaries=len(gold),
            )
            pred = result.to_set()
        else:
            pred = set()

        predictions.append(pred)
        gold_boundaries.append(gold)
        num_messages_list.append(num_msg)

    return predictions, gold_boundaries, num_messages_list


def results_to_csv(results: List[EvalResult], output_path: str):
    """Write results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Dataset', 'N', 'Gold', 'Pred', 'TP',
            'Strict_F1', 'Strict_P', 'Strict_R',
            'W_F1', 'W_P', 'W_R',
            'BOR', 'Purity', 'Coverage'
        ])
        for r in results:
            writer.writerow([
                r.method, r.dataset, r.n_dialogues,
                r.total_gold, r.total_pred, r.total_tp,
                f"{r.strict_f1:.3f}", f"{r.strict_prec:.3f}", f"{r.strict_rec:.3f}",
                f"{r.w_f1:.3f}", f"{r.w_prec:.3f}", f"{r.w_rec:.3f}",
                f"{r.bor:.2f}", f"{r.purity:.3f}", f"{r.coverage:.3f}"
            ])


def results_to_latex(results: List[EvalResult]) -> str:
    """Generate LaTeX table rows from results."""
    lines = []
    lines.append("% Method & Dataset & F1 & W-F1 & BOR & Purity & Coverage \\\\")
    lines.append("\\midrule")

    for r in results:
        line = f"{r.method} & {r.dataset} & {r.strict_f1:.3f} & {r.w_f1:.3f} & {r.bor:.2f} & {r.purity:.3f} & {r.coverage:.3f} \\\\"
        lines.append(line)

    return "\n".join(lines)


def print_results_table(results: List[EvalResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print(f"{'Method':<15} {'Dataset':<15} {'F1':>6} {'W-F1':>6} {'BOR':>5} {'Purity':>7} {'Coverage':>8}")
    print("=" * 90)

    for r in results:
        regime = "CONSERV" if r.bor < 0.8 else ("AGGRESS" if r.bor > 1.2 else "BALANCE")
        print(f"{r.method:<15} {r.dataset:<15} {r.strict_f1:>6.3f} {r.w_f1:>6.3f} {r.bor:>5.2f} {r.purity:>7.3f} {r.coverage:>8.3f}  [{regime}]")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Run external baseline segmenters and evaluate"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["texttiling", "csm_nsp", "random", "even"],
        choices=["texttiling", "csm_nsp", "random", "even"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dialseg711"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/Dialogue-Topic-Segmenter/data/eval"),
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-alpha-tune",
        action="store_true",
        help="Skip alpha tuning on dev set",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("External Baseline Evaluation")
    print("=" * 60)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print()

    all_results = []

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print("=" * 60)

        # Find dataset file
        dataset_path = args.data_dir / f"{dataset_name}.json"
        if not dataset_path.exists():
            # Try alternative names
            for alt in [f"{dataset_name}_711.json", f"dialseg_{dataset_name}.json"]:
                alt_path = args.data_dir / alt
                if alt_path.exists():
                    dataset_path = alt_path
                    break

        if not dataset_path.exists():
            print(f"  Dataset not found: {dataset_path}")
            continue

        print(f"  Loading: {dataset_path}")

        # Determine format and load
        with open(dataset_path) as f:
            data = json.load(f)

        # CSM format has 'utterances' and 'segments' keys
        if isinstance(data, list) and data and "utterances" in data[0]:
            dataset_format = "csm"
            dialogues = data
        else:
            dataset_format = "canonical"
            dialogues = data

        print(f"  Format: {dataset_format}, Dialogues: {len(dialogues)}")

        for method_name in args.methods:
            print(f"\n  Running: {method_name}")

            try:
                segmenter = get_segmenter(method_name)

                predictions, gold_boundaries, num_messages = run_segmenter_on_dataset(
                    segmenter,
                    dialogues,
                    dataset_format=dataset_format,
                    use_dev_for_alpha=not args.no_alpha_tune,
                )

                result = evaluate_predictions(
                    method_name=segmenter.name,
                    dataset_name=dataset_name,
                    predictions=predictions,
                    gold_boundaries=gold_boundaries,
                    num_messages_list=num_messages,
                )

                all_results.append(result)

                print(f"    F1: {result.strict_f1:.3f}, W-F1: {result.w_f1:.3f}, BOR: {result.bor:.2f}")

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()

    # Print summary table
    print_results_table(all_results)

    # Output files
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results_to_csv(all_results, str(args.out))
        print(f"\nResults written to: {args.out}")

    # Print LaTeX
    print("\n--- LaTeX Table Fragment ---")
    print(results_to_latex(all_results))


if __name__ == "__main__":
    main()
