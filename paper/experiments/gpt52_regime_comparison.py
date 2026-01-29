#!/usr/bin/env python3
"""
GPT-5.2 Regime Separation Plot

Generates cross-dataset curve comparison directly addressing the paper's thesis.

Usage:
    python paper/experiments/gpt52_regime_comparison.py
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
FLIPPED_RESULTS = PROJECT_ROOT / "paper" / "experiments" / "gpt52_sanity_results_flipped.json"
NEURAL_DIALSEG = PROJECT_ROOT / "paper" / "results" / "sweep_dialseg711_neural_per_dialogue.json"
NEURAL_SUPERSEG = PROJECT_ROOT / "paper" / "results" / "sweep_superseg_neural_per_dialogue.json"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments" / "gpt52_diagnostics_flipped"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def interpolate_wf1_at_bor(bors, wf1s, target_bor):
    """Linear interpolation to find W-F1 at a specific BOR."""
    bors = np.array(bors)
    wf1s = np.array(wf1s)

    # Sort by BOR
    idx = np.argsort(bors)
    bors = bors[idx]
    wf1s = wf1s[idx]

    if target_bor < bors.min() or target_bor > bors.max():
        return None

    return np.interp(target_bor, bors, wf1s)


def interpolate_bor_at_wf1(bors, wf1s, target_wf1):
    """Find BOR at a specific W-F1 threshold."""
    bors = np.array(bors)
    wf1s = np.array(wf1s)

    # Sort by BOR
    idx = np.argsort(bors)
    bors = bors[idx]
    wf1s = wf1s[idx]

    # Find first crossing point (may not be monotonic)
    for i in range(len(wf1s) - 1):
        if (wf1s[i] <= target_wf1 <= wf1s[i+1]) or (wf1s[i] >= target_wf1 >= wf1s[i+1]):
            # Linear interpolation
            t = (target_wf1 - wf1s[i]) / (wf1s[i+1] - wf1s[i]) if wf1s[i+1] != wf1s[i] else 0
            return bors[i] + t * (bors[i+1] - bors[i])

    return None


def find_crossing_point(bors1, wf1s1, bors2, wf1s2):
    """Find BOR where two curves cross."""
    # Create common BOR grid
    min_bor = max(min(bors1), min(bors2))
    max_bor = min(max(bors1), max(bors2))

    if min_bor >= max_bor:
        return None

    grid = np.linspace(min_bor, max_bor, 100)

    interp1 = np.interp(grid, np.array(bors1)[np.argsort(bors1)],
                        np.array(wf1s1)[np.argsort(bors1)])
    interp2 = np.interp(grid, np.array(bors2)[np.argsort(bors2)],
                        np.array(wf1s2)[np.argsort(bors2)])

    diff = interp1 - interp2

    # Find sign changes
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            # Crossing detected
            return grid[i]

    return None


def main():
    print("=" * 60)
    print("GPT-5.2 REGIME SEPARATION COMPARISON")
    print("=" * 60)

    # Load GPT-5.2 flipped results
    print("\nLoading GPT-5.2 flipped results...")
    gpt_results = load_json(FLIPPED_RESULTS)

    # Extract GPT-5.2 curves
    gpt_curves = {}
    for dataset_name in ["dialseg711", "dailydialog"]:
        sweep = gpt_results["datasets"][dataset_name]["sweep_points"]
        gpt_curves[dataset_name] = {
            "bor": [p["bor"] for p in sweep],
            "wf1": [p["wf1"] for p in sweep]
        }
        print(f"  {dataset_name}: {len(sweep)} points")

    # Load DistilBERT curves if available
    distilbert_curves = {}
    if NEURAL_DIALSEG.exists():
        print("\nLoading DistilBERT dialseg711 curves...")
        neural_data = load_json(NEURAL_DIALSEG)
        distilbert_curves["dialseg711"] = {
            "bor": [p["bor"] for p in neural_data["points"]],
            "wf1": [p["wf1"] for p in neural_data["points"]]
        }
        print(f"  dialseg711: {len(neural_data['points'])} points")

    if NEURAL_SUPERSEG.exists():
        print("Loading DistilBERT superseg curves (coarse-grained comparator)...")
        neural_data = load_json(NEURAL_SUPERSEG)
        distilbert_curves["superseg"] = {
            "bor": [p["bor"] for p in neural_data["points"]],
            "wf1": [p["wf1"] for p in neural_data["points"]]
        }
        print(f"  superseg: {len(neural_data['points'])} points")

    # ============================================================
    # Analysis Questions
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    print("\n1. W-F1 at BOR = 0.10:")
    for dataset, curve in gpt_curves.items():
        wf1 = interpolate_wf1_at_bor(curve["bor"], curve["wf1"], 0.10)
        if wf1 is not None:
            print(f"   GPT-5.2 {dataset}: {wf1:.4f}")
        else:
            print(f"   GPT-5.2 {dataset}: N/A (BOR out of range)")

    print("\n2. W-F1 at BOR = 0.30:")
    for dataset, curve in gpt_curves.items():
        wf1 = interpolate_wf1_at_bor(curve["bor"], curve["wf1"], 0.30)
        if wf1 is not None:
            print(f"   GPT-5.2 {dataset}: {wf1:.4f}")
        else:
            print(f"   GPT-5.2 {dataset}: N/A (BOR out of range)")

    print("\n3. BOR required for W-F1 = 0.50:")
    for dataset, curve in gpt_curves.items():
        bor = interpolate_bor_at_wf1(curve["bor"], curve["wf1"], 0.50)
        if bor is not None:
            print(f"   GPT-5.2 {dataset}: {bor:.4f}")
        else:
            print(f"   GPT-5.2 {dataset}: N/A (never reaches W-F1=0.50)")

    print("\n4. Do GPT-5.2 curves cross?")
    crossing = find_crossing_point(
        gpt_curves["dialseg711"]["bor"], gpt_curves["dialseg711"]["wf1"],
        gpt_curves["dailydialog"]["bor"], gpt_curves["dailydialog"]["wf1"]
    )
    if crossing is not None:
        print(f"   Yes, at BOR ~ {crossing:.3f}")
    else:
        print("   No crossing detected in overlapping range")

    # ============================================================
    # Create Plot
    # ============================================================
    print("\n" + "=" * 60)
    print("CREATING PLOT")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[ERROR] matplotlib not available")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot GPT-5.2 curves
    colors = {"dialseg711": "#2E86AB", "dailydialog": "#E94F37"}

    for dataset, curve in gpt_curves.items():
        bors = curve["bor"]
        wf1s = curve["wf1"]
        # Sort by BOR
        idx = np.argsort(bors)
        bors = np.array(bors)[idx]
        wf1s = np.array(wf1s)[idx]

        label_name = "DialSeg711 (fine)" if dataset == "dialseg711" else "DailyDialog (coarse)"
        ax.plot(bors, wf1s, '-', color=colors[dataset], linewidth=2.5,
                label=f"GPT-5.2: {label_name}", marker='o', markersize=4)

    # Plot DistilBERT curves (dashed)
    if distilbert_curves:
        distilbert_colors = {"dialseg711": "#2E86AB", "superseg": "#E94F37"}
        for dataset, curve in distilbert_curves.items():
            bors = curve["bor"]
            wf1s = curve["wf1"]
            idx = np.argsort(bors)
            bors = np.array(bors)[idx]
            wf1s = np.array(wf1s)[idx]

            # Subsample for clarity (200 points is too dense)
            step = max(1, len(bors) // 20)
            bors = bors[::step]
            wf1s = wf1s[::step]

            label_name = "DialSeg711 (fine)" if dataset == "dialseg711" else "SuperSeg (coarse)"
            ax.plot(bors, wf1s, '--', color=distilbert_colors[dataset], linewidth=1.5,
                    alpha=0.7, label=f"DistilBERT: {label_name}")

    # Add reference lines
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='BOR=1 (balanced)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel('BOR (Boundary Oversegmentation Ratio)', fontsize=12)
    ax.set_ylabel('W-F1', fontsize=12)
    ax.set_title('Regime Separation Under GPT-5.2 Scorer\nFine vs Coarse Granularity Datasets', fontsize=14)
    ax.set_xlim(0, 2.7)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "regime_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nPlot saved to: {output_path}")

    # ============================================================
    # DistilBERT Comparison
    # ============================================================
    if distilbert_curves:
        print("\n" + "=" * 60)
        print("DISTILBERT vs GPT-5.2 COMPARISON")
        print("=" * 60)

        # Compare at BOR = 1.0
        print("\nW-F1 at BOR = 1.0:")
        for dataset, curve in gpt_curves.items():
            wf1 = interpolate_wf1_at_bor(curve["bor"], curve["wf1"], 1.0)
            if wf1 is not None:
                print(f"  GPT-5.2 {dataset}: {wf1:.4f}")
        for dataset, curve in distilbert_curves.items():
            wf1 = interpolate_wf1_at_bor(curve["bor"], curve["wf1"], 1.0)
            if wf1 is not None:
                print(f"  DistilBERT {dataset}: {wf1:.4f}")

        # Check ordering consistency
        if "dialseg711" in distilbert_curves:
            gpt_dial = interpolate_wf1_at_bor(gpt_curves["dialseg711"]["bor"],
                                              gpt_curves["dialseg711"]["wf1"], 1.0)
            gpt_daily = interpolate_wf1_at_bor(gpt_curves["dailydialog"]["bor"],
                                               gpt_curves["dailydialog"]["wf1"], 1.0)
            db_dial = interpolate_wf1_at_bor(distilbert_curves["dialseg711"]["bor"],
                                             distilbert_curves["dialseg711"]["wf1"], 1.0)

            if gpt_dial and gpt_daily and db_dial:
                gpt_order = "dialseg711 > dailydialog" if gpt_dial > gpt_daily else "dailydialog > dialseg711"
                print(f"\nGPT-5.2 ordering at BOR=1.0: {gpt_order}")
                print("(DistilBERT comparison requires superseg for coarse granularity)")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
