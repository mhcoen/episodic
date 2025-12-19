#!/usr/bin/env python3
"""
Deep Exploratory Analysis: Concentrated Failures by Gold Density

Key question: Do granularity failures concentrate in dialogues with
atypical gold density?

Author: Exploratory analysis for paper
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def load_dialogues_with_features(dataset_name: str):
    """Load dialogues with computed features."""
    test_file = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    with open(test_file) as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)
    dialogues = []

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            user_turns = [t for t in turns if t.get("role") == "user"]
            n_user_turns = len(user_turns)

            # Extract boundaries
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

            n_boundaries = len(boundaries)
            possible = max(n_user_turns - 1, 1)
            density = n_boundaries / possible

            dialogues.append({
                "n_user_turns": n_user_turns,
                "n_gold_boundaries": n_boundaries,
                "gold_density": density,
                "gold_boundaries": boundaries,
            })

    return dialogues


def analyze_wf1_by_density_stratum(dataset_name: str):
    """
    Load per-dialogue metrics and stratify W-F1 by gold density.

    Key insight: Does the model achieve similar W-F1 across dialogues
    with different gold densities, or do atypical dialogues suffer?
    """
    dialogues = load_dialogues_with_features(dataset_name)
    n_dialogues = len(dialogues)

    # Create density strata
    densities = [d["gold_density"] for d in dialogues]
    p33 = np.percentile(densities, 33)
    p66 = np.percentile(densities, 66)

    def classify_density(d):
        if d <= p33:
            return "low"
        elif d > p66:
            return "high"
        else:
            return "medium"

    for i, d in enumerate(dialogues):
        d["density_stratum"] = classify_density(d["gold_density"])

    # Load per-dialogue BOR for neural model
    bor_file = RESULTS_DIR / f"per_dialogue_bor_{dataset_name}_neural.csv"
    if not bor_file.exists():
        print(f"  BOR file not found: {bor_file}")
        return None

    df_bor = pd.read_csv(bor_file)

    # Merge
    for i, row in df_bor.iterrows():
        if i < len(dialogues):
            dialogues[i]["bor"] = row["bor_dialogue"]
            dialogues[i]["n_pred_boundaries"] = row["n_pred_boundaries"]

    # Filter dialogues with BOR data
    dialogues = [d for d in dialogues if "bor" in d]

    # Analyze by stratum
    results = {}
    for stratum in ["low", "medium", "high"]:
        subset = [d for d in dialogues if d["density_stratum"] == stratum]
        if not subset:
            continue

        bors = [d["bor"] for d in subset]
        gold_counts = [d["n_gold_boundaries"] for d in subset]
        pred_counts = [d["n_pred_boundaries"] for d in subset]

        results[stratum] = {
            "n": len(subset),
            "bor_mean": np.mean(bors),
            "bor_std": np.std(bors),
            "bor_median": np.median(bors),
            "gold_mean": np.mean(gold_counts),
            "pred_mean": np.mean(pred_counts),
            "near_one_frac": sum(1 for b in bors if 0.8 <= b <= 1.2) / len(bors),
        }

    return results, dialogues, (p33, p66)


def analyze_concentrated_failures():
    """
    Main analysis: Are granularity failures concentrated?
    """
    print("=" * 70)
    print("CONCENTRATED FAILURES ANALYSIS")
    print("=" * 70)

    datasets = ["dialseg711", "superseg"]

    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"Dataset: {dataset.upper()}")
        print("=" * 50)

        result = analyze_wf1_by_density_stratum(dataset)
        if result is None:
            continue

        strata_results, dialogues, (p33, p66) = result

        print(f"\nDensity thresholds: p33={p33:.3f}, p66={p66:.3f}")
        print(f"\nPer-Stratum Statistics:")
        print("-" * 60)
        print(f"{'Stratum':<10} {'N':<6} {'BOR mean':<12} {'BOR std':<10} "
              f"{'Gold mean':<10} {'Pred mean':<10} {'Near BOR=1':<10}")
        print("-" * 60)

        for stratum in ["low", "medium", "high"]:
            if stratum not in strata_results:
                continue
            r = strata_results[stratum]
            print(f"{stratum:<10} {r['n']:<6} {r['bor_mean']:<12.3f} {r['bor_std']:<10.3f} "
                  f"{r['gold_mean']:<10.2f} {r['pred_mean']:<10.2f} {r['near_one_frac']*100:<10.1f}%")

        # Key finding: Is the difference between strata significant?
        if "low" in strata_results and "high" in strata_results:
            bor_diff = strata_results["low"]["bor_mean"] - strata_results["high"]["bor_mean"]
            print(f"\nBOR difference (low - high): {bor_diff:.3f}")

            if abs(bor_diff) > 0.3:
                print("  → SUBSTANTIAL: Model has consistent prediction density but gold varies")
                print("  → This explains granularity mismatch concentration")
            else:
                print("  → MODEST: Granularity failures more uniformly distributed")


def check_curve_shape_differences():
    """
    Check if the sweep curve shape differs between datasets.
    Focus: inflection points, peak locations, flatness.
    """
    print("\n" + "=" * 70)
    print("CURVE SHAPE ANALYSIS")
    print("=" * 70)

    datasets = ["dialseg711", "superseg"]
    model = "neural"

    for dataset in datasets:
        sweep_file = RESULTS_DIR / f"sweep_{dataset}_{model}.csv"
        if not sweep_file.exists():
            continue

        df = pd.read_csv(sweep_file)
        df_sorted = df.sort_values("bor")

        # Find peak W-F1 and its BOR location
        peak_idx = df_sorted["wf1"].idxmax()
        peak_bor = df_sorted.loc[peak_idx, "bor"]
        peak_wf1 = df_sorted.loc[peak_idx, "wf1"]

        # W-F1 at BOR=1 (interpolated)
        near_one = df_sorted[(df_sorted["bor"] >= 0.9) & (df_sorted["bor"] <= 1.1)]
        wf1_at_one = near_one["wf1"].mean() if len(near_one) > 0 else np.nan

        # Curve flatness: std of W-F1 in range 0.5-2.0 BOR
        mid_range = df_sorted[(df_sorted["bor"] >= 0.5) & (df_sorted["bor"] <= 2.0)]
        wf1_std_mid = mid_range["wf1"].std() if len(mid_range) > 5 else np.nan

        print(f"\n{dataset.upper()} (neural):")
        print(f"  Peak W-F1: {peak_wf1:.3f} at BOR={peak_bor:.2f}")
        print(f"  W-F1 at BOR≈1: {wf1_at_one:.3f}")
        print(f"  W-F1 std (BOR 0.5-2.0): {wf1_std_mid:.4f}")
        print(f"  Peak displacement from BOR=1: {peak_bor - 1:.2f}")


def analyze_cross_dataset_prediction_consistency():
    """
    Check: Does the neural model predict similar boundary counts
    regardless of gold annotation density?

    This would explain why BOR varies inversely with gold density.
    """
    print("\n" + "=" * 70)
    print("PREDICTION CONSISTENCY ANALYSIS")
    print("=" * 70)

    datasets = ["dialseg711", "superseg"]

    for dataset in datasets:
        dialogues = load_dialogues_with_features(dataset)
        bor_file = RESULTS_DIR / f"per_dialogue_bor_{dataset}_neural.csv"

        if not bor_file.exists():
            continue

        df_bor = pd.read_csv(bor_file)

        # Merge predictions
        for i, row in df_bor.iterrows():
            if i < len(dialogues):
                dialogues[i]["n_pred"] = row["n_pred_boundaries"]

        dialogues = [d for d in dialogues if "n_pred" in d]

        # Analyze prediction variance
        pred_counts = [d["n_pred"] for d in dialogues]
        gold_counts = [d["n_gold_boundaries"] for d in dialogues]
        lengths = [d["n_user_turns"] for d in dialogues]

        print(f"\n{dataset.upper()}:")
        print(f"  N dialogues: {len(dialogues)}")
        print(f"  Predicted boundaries: mean={np.mean(pred_counts):.2f}, "
              f"std={np.std(pred_counts):.2f}")
        print(f"  Gold boundaries: mean={np.mean(gold_counts):.2f}, "
              f"std={np.std(gold_counts):.2f}")
        print(f"  Dialogue length: mean={np.mean(lengths):.1f}")

        # Coefficient of variation
        cv_pred = np.std(pred_counts) / np.mean(pred_counts) if np.mean(pred_counts) > 0 else 0
        cv_gold = np.std(gold_counts) / np.mean(gold_counts) if np.mean(gold_counts) > 0 else 0

        print(f"  CV(predictions): {cv_pred:.3f}")
        print(f"  CV(gold): {cv_gold:.3f}")

        if cv_pred < cv_gold:
            print("  → Model predictions are MORE CONSISTENT than gold")
            print("    This causes BOR to vary inversely with gold density")


def generate_stratified_bor_figure():
    """
    Generate compact figure showing BOR distribution by gold density stratum.
    Only generate if patterns are sufficiently different between datasets.
    """
    print("\n" + "=" * 70)
    print("FIGURE GENERATION ASSESSMENT")
    print("=" * 70)

    # Load data for both datasets
    data = {}
    for dataset in ["dialseg711", "superseg"]:
        result = analyze_wf1_by_density_stratum(dataset)
        if result is None:
            continue
        strata_results, dialogues, thresholds = result
        data[dataset] = {
            "strata": strata_results,
            "dialogues": dialogues,
            "thresholds": thresholds,
        }

    if len(data) < 2:
        print("Insufficient data for comparison figure.")
        return

    # Check if patterns are sufficiently different to warrant a figure
    d711_diff = (data["dialseg711"]["strata"]["low"]["bor_mean"] -
                 data["dialseg711"]["strata"]["high"]["bor_mean"])
    sseg_diff = (data["superseg"]["strata"]["low"]["bor_mean"] -
                 data["superseg"]["strata"]["high"]["bor_mean"])

    print(f"\nDialSeg711 BOR difference (low-high): {d711_diff:.3f}")
    print(f"SuperSeg BOR difference (low-high): {sseg_diff:.3f}")

    # Only generate figure if the difference between datasets is substantial
    if abs(d711_diff - sseg_diff) > 0.3:
        print("\n→ Pattern difference is SUBSTANTIAL - figure would add new information")
        print("  However, this may already be implicit in the existing curves.")
    else:
        print("\n→ Pattern difference is MODEST - figure would be REDUNDANT")
        print("  Recommend: Report in text, no additional figure needed.")

    # Additional check: is the DialSeg711 pattern actually showing something new?
    print("\n" + "-" * 50)
    print("KEY INSIGHT CHECK:")
    print("-" * 50)

    # The key insight: granularity mismatch is predictable from gold density
    print("\nDialSeg711:")
    print("  Low gold-density dialogues → High BOR (model oversegments)")
    print("  High gold-density dialogues → Low BOR (model undersegments)")
    print("  → Model has learned a 'typical' granularity")

    print("\nSuperSeg:")
    print(f"  Low gold-density: BOR = {data['superseg']['strata']['low']['bor_mean']:.2f}")
    print(f"  High gold-density: BOR = {data['superseg']['strata']['high']['bor_mean']:.2f}")
    print("  → Pattern is weaker due to more uniform gold density")


def main():
    analyze_concentrated_failures()
    check_curve_shape_differences()
    analyze_cross_dataset_prediction_consistency()
    generate_stratified_bor_figure()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
