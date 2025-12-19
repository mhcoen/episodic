#!/usr/bin/env python3
"""
Exploratory Analysis for Granularity Mismatch in Dialogue Topic Segmentation

Part A: Dataset-level regime scan
Part B: Conditional density-quality curves (if new regime found)
Part C: Dialogue-level conditioning

Author: Exploratory analysis for paper
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"

# All datasets with segmentation files
ALL_DATASETS = [
    "dialseg711", "superseg", "tiage", "dailydialog",
    "topical_chat", "qmsum", "taskmaster", "multiwoz"
]


def load_dataset_stats(dataset_name: str) -> Dict:
    """Load dataset and compute key statistics."""
    test_file = DATASETS_DIR / dataset_name / "segmentation_file_test.json"
    if not test_file.exists():
        return None

    with open(test_file) as f:
        data = json.load(f)

    dial_data = data.get("dial_data", data)

    dialogue_lengths = []
    gold_boundary_counts = []
    gold_densities = []  # boundaries / (user_turns - 1)
    zero_boundary_count = 0

    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue

        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            # Count user turns
            user_turns = [t for t in turns if t.get("role") == "user"]
            n_user_turns = len(user_turns)

            # Extract boundaries (topic changes)
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
            dialogue_lengths.append(n_user_turns)
            gold_boundary_counts.append(n_boundaries)

            # Density: boundaries per possible boundary position
            possible_positions = max(n_user_turns - 1, 1)
            density = n_boundaries / possible_positions
            gold_densities.append(density)

            if n_boundaries == 0:
                zero_boundary_count += 1

    n_dialogues = len(dialogue_lengths)

    return {
        "dataset": dataset_name,
        "n_dialogues": n_dialogues,
        "dialogue_lengths": dialogue_lengths,
        "gold_boundary_counts": gold_boundary_counts,
        "gold_densities": gold_densities,
        "length_mean": np.mean(dialogue_lengths),
        "length_std": np.std(dialogue_lengths),
        "length_median": np.median(dialogue_lengths),
        "boundaries_mean": np.mean(gold_boundary_counts),
        "boundaries_std": np.std(gold_boundary_counts),
        "boundaries_median": np.median(gold_boundary_counts),
        "density_mean": np.mean(gold_densities),
        "density_std": np.std(gold_densities),
        "density_median": np.median(gold_densities),
        "zero_boundary_frac": zero_boundary_count / n_dialogues if n_dialogues > 0 else 0,
        "total_boundaries": sum(gold_boundary_counts),
    }


def classify_regime(stats: Dict) -> str:
    """Classify dataset into regime based on density statistics."""
    density_mean = stats["density_mean"]
    density_median = stats["density_median"]
    zero_frac = stats["zero_boundary_frac"]

    # Classification logic:
    # - Sparse: low density mean (<0.15), higher zero-boundary fraction
    # - Dense: high density mean (>0.25)
    # - Medium/transitional: in between

    if density_mean < 0.12:
        return "sparse-gold"
    elif density_mean > 0.20:
        return "dense-gold"
    else:
        return "medium"


def part_a_regime_scan():
    """Part A: Dataset-level regime scan."""
    print("=" * 70)
    print("PART A: Dataset-Level Regime Scan")
    print("=" * 70)

    all_stats = []

    for ds in ALL_DATASETS:
        stats = load_dataset_stats(ds)
        if stats is None:
            print(f"  {ds}: NOT FOUND")
            continue
        all_stats.append(stats)

    # Create summary table
    rows = []
    for stats in all_stats:
        regime = classify_regime(stats)
        rows.append({
            "Dataset": stats["dataset"],
            "N Dialogues": stats["n_dialogues"],
            "Length (mean±std)": f"{stats['length_mean']:.1f}±{stats['length_std']:.1f}",
            "Gold Boundaries (mean±std)": f"{stats['boundaries_mean']:.2f}±{stats['boundaries_std']:.2f}",
            "Density (mean)": f"{stats['density_mean']:.3f}",
            "Zero-boundary %": f"{stats['zero_boundary_frac']*100:.1f}%",
            "Regime": regime,
        })

    df_summary = pd.DataFrame(rows)
    print("\nDataset Statistics Summary:")
    print(df_summary.to_string(index=False))

    # Group by regime
    print("\n" + "-" * 50)
    print("Regime Classification:")
    regimes = defaultdict(list)
    for stats in all_stats:
        regime = classify_regime(stats)
        regimes[regime].append(stats["dataset"])

    for regime, datasets in sorted(regimes.items()):
        print(f"  {regime}: {', '.join(datasets)}")

    return all_stats, regimes


def part_c_dialogue_conditioning():
    """Part C: Dialogue-level conditioning analysis."""
    print("\n" + "=" * 70)
    print("PART C: Dialogue-Level Conditioning")
    print("=" * 70)

    # Load per-dialogue BOR data for the two main datasets
    datasets = ["dialseg711", "superseg"]
    models = ["neural", "texttiling", "csm", "random"]

    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")

        # Load dataset stats for stratification
        stats = load_dataset_stats(dataset)
        if stats is None:
            print(f"  Dataset not found")
            continue

        # Create dialogue-level features for stratification
        dialogue_lengths = stats["dialogue_lengths"]
        gold_densities = stats["gold_densities"]
        n_dialogues = len(dialogue_lengths)

        # Stratify dialogues
        # Length: short (<= median), long (> median)
        length_median = np.median(dialogue_lengths)
        length_strata = ["short" if l <= length_median else "long" for l in dialogue_lengths]

        # Gold density: low (<= 33rd pctl), medium, high (> 66th pctl)
        density_p33 = np.percentile(gold_densities, 33)
        density_p66 = np.percentile(gold_densities, 66)

        def classify_density(d):
            if d <= density_p33:
                return "low"
            elif d > density_p66:
                return "high"
            else:
                return "medium"

        density_strata = [classify_density(d) for d in gold_densities]

        print(f"  Length median: {length_median:.1f}")
        print(f"  Density percentiles: p33={density_p33:.3f}, p66={density_p66:.3f}")

        # Load per-dialogue BOR data for neural model (primary analysis)
        model = "neural"
        bor_file = RESULTS_DIR / f"per_dialogue_bor_{dataset}_{model}.csv"

        if not bor_file.exists():
            print(f"  BOR file not found: {bor_file}")
            continue

        df_bor = pd.read_csv(bor_file)

        # Merge with stratification
        df_bor["length_stratum"] = length_strata[:len(df_bor)]
        df_bor["density_stratum"] = density_strata[:len(df_bor)]
        df_bor["gold_density"] = gold_densities[:len(df_bor)]
        df_bor["dialogue_length"] = dialogue_lengths[:len(df_bor)]

        # Compute statistics by stratum
        print(f"\n  BOR Distribution by Gold Density Stratum:")
        for stratum in ["low", "medium", "high"]:
            subset = df_bor[df_bor["density_stratum"] == stratum]
            if len(subset) == 0:
                continue
            bor_vals = subset["bor_dialogue"].values
            # Focus on dialogues near BOR=1 (0.8 to 1.2)
            near_one = subset[(subset["bor_dialogue"] >= 0.5) & (subset["bor_dialogue"] <= 1.5)]
            print(f"    {stratum}: n={len(subset)}, BOR mean={np.mean(bor_vals):.2f}±{np.std(bor_vals):.2f}, "
                  f"near BOR=1: {len(near_one)} ({100*len(near_one)/len(subset):.1f}%)")

        print(f"\n  BOR Distribution by Dialogue Length Stratum:")
        for stratum in ["short", "long"]:
            subset = df_bor[df_bor["length_stratum"] == stratum]
            if len(subset) == 0:
                continue
            bor_vals = subset["bor_dialogue"].values
            print(f"    {stratum}: n={len(subset)}, BOR mean={np.mean(bor_vals):.2f}±{np.std(bor_vals):.2f}")

        # Analyze variance concentration
        print(f"\n  Per-Dialogue BOR Variance Analysis:")

        # High variance dialogues (BOR far from 1)
        df_bor["bor_deviation"] = np.abs(df_bor["bor_dialogue"] - 1.0)

        # Correlation between deviation and dialogue features
        corr_length = np.corrcoef(df_bor["dialogue_length"], df_bor["bor_deviation"])[0, 1]
        corr_density = np.corrcoef(df_bor["gold_density"], df_bor["bor_deviation"])[0, 1]

        print(f"    Corr(dialogue_length, |BOR-1|): {corr_length:.3f}")
        print(f"    Corr(gold_density, |BOR-1|): {corr_density:.3f}")

        # Check if high-variance dialogues are concentrated
        high_variance_threshold = df_bor["bor_deviation"].quantile(0.75)
        high_var_dialogues = df_bor[df_bor["bor_deviation"] > high_variance_threshold]

        print(f"\n  High-Variance Dialogues (top 25% by |BOR-1|):")
        print(f"    n={len(high_var_dialogues)}")
        print(f"    By density: {high_var_dialogues['density_stratum'].value_counts().to_dict()}")
        print(f"    By length: {high_var_dialogues['length_stratum'].value_counts().to_dict()}")


def load_sweep_data_near_bor1():
    """Load sweep data and compute sensitivity metrics near BOR=1."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS NEAR BOR=1")
    print("=" * 70)

    datasets = ["dialseg711", "superseg"]
    models = ["neural", "texttiling", "csm"]

    results = []

    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")

        for model in models:
            sweep_file = RESULTS_DIR / f"sweep_{dataset}_{model}.csv"
            if not sweep_file.exists():
                continue

            df = pd.read_csv(sweep_file)

            # Filter to BOR range near 1 (0.8 to 1.2)
            near_one = df[(df["bor"] >= 0.8) & (df["bor"] <= 1.2)]

            if len(near_one) < 2:
                continue

            # Compute slope (W-F1 change per unit BOR change)
            near_one_sorted = near_one.sort_values("bor")
            bor_vals = near_one_sorted["bor"].values
            wf1_vals = near_one_sorted["wf1"].values

            # Linear regression slope
            if len(bor_vals) >= 2:
                slope = np.polyfit(bor_vals, wf1_vals, 1)[0]
                wf1_range = wf1_vals.max() - wf1_vals.min()

                results.append({
                    "dataset": dataset,
                    "model": model,
                    "slope_near_bor1": slope,
                    "wf1_range_near_bor1": wf1_range,
                    "n_points": len(near_one),
                })

                print(f"  {model}: slope={slope:.3f}, W-F1 range={wf1_range:.3f} (n={len(near_one)})")

    if results:
        df_results = pd.DataFrame(results)
        print("\nSummary Table:")
        print(df_results.to_string(index=False))


def main():
    """Run all exploratory analyses."""
    print("EXPLORATORY ANALYSIS FOR GRANULARITY MISMATCH")
    print("=" * 70)
    print()

    # Part A
    all_stats, regimes = part_a_regime_scan()

    # Part B: Check if any dataset is structurally distinct
    print("\n" + "=" * 70)
    print("PART B: New Regime Assessment")
    print("=" * 70)

    # Already analyzed in Part A
    sparse = regimes.get("sparse-gold", [])
    dense = regimes.get("dense-gold", [])
    medium = regimes.get("medium", [])

    if medium:
        print(f"\nDatasets in 'medium' regime: {medium}")
        print("These may represent transitional cases.")

        # Check if they're distinct enough to warrant additional figures
        for ds in medium:
            stats = next(s for s in all_stats if s["dataset"] == ds)
            print(f"\n  {ds}:")
            print(f"    Density: {stats['density_mean']:.3f} (median: {stats['density_median']:.3f})")
            print(f"    Compared to dialseg711 (dense): ~0.15-0.20")
            print(f"    Compared to superseg (sparse): ~0.07-0.10")
    else:
        print("\nNo datasets fall into a third 'medium' regime.")
        print("All datasets are either sparse-gold-like or dense-gold-like.")

    # Part C
    part_c_dialogue_conditioning()

    # Sensitivity analysis
    load_sweep_data_near_bor1()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
