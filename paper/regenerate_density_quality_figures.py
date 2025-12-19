#!/usr/bin/env python3
"""
Quick script to regenerate density-quality figures from existing sweep data.
This avoids re-running the expensive neural scoring.

Usage:
    python paper/regenerate_density_quality_figures.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from paper.density_quality_curves import (
    plot_density_quality_curves,
    FIGURES_DIR,
    RESULTS_DIR,
    DATASETS,
    log,
)


def main():
    """Regenerate figures from existing sweep CSVs."""
    log("Regenerating density-quality figures from existing sweep data...")
    log(f"Results dir: {RESULTS_DIR}")
    log(f"Figures dir: {FIGURES_DIR}")

    for dataset_name, dataset_info in DATASETS.items():
        display_name = dataset_info["display"]
        log(f"\n--- {display_name} ---")

        # Load existing sweep results
        results = {}
        for model in ["neural", "texttiling", "csm", "random"]:
            csv_path = RESULTS_DIR / f"sweep_{dataset_name}_{model}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                results[model] = df
                log(f"  Loaded {model}: {len(df)} points")
            else:
                log(f"  SKIP {model}: {csv_path} not found")

        if not results:
            log(f"  No data found for {dataset_name}, skipping...")
            continue

        # Generate original plots
        log(f"\n  Generating original plots...")
        plot_density_quality_curves(results, dataset_name, display_name, FIGURES_DIR)

        # Generate plots with regime overlays
        log(f"\n  Generating plots with regime overlays...")
        plot_density_quality_curves(
            results, dataset_name, display_name, FIGURES_DIR,
            with_regime_overlays=True, output_suffix="_regimes"
        )

    log("\nDone.")


if __name__ == "__main__":
    main()
