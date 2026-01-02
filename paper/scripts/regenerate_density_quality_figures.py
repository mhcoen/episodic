#!/usr/bin/env python3
"""
Quick script to regenerate density-quality figures from existing sweep data.
This avoids re-running the expensive neural scoring.

Usage:
    python regenerate_density_quality_figures.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also add scripts directory for local imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from density_quality_curves import (
    plot_density_quality_curves,
    plot_density_quality_curves_with_ci,
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
        ci_data = {}
        for model in ["neural", "texttiling", "csm", "random"]:
            csv_path = RESULTS_DIR / f"sweep_{dataset_name}_{model}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                results[model] = df
                log(f"  Loaded {model}: {len(df)} points")
            else:
                log(f"  SKIP {model}: {csv_path} not found")

            # Load CI data if available
            ci_path = RESULTS_DIR / f"bootstrap_ci_{dataset_name}_{model}.csv"
            if ci_path.exists():
                ci_df = pd.read_csv(ci_path)
                ci_data[model] = ci_df
                log(f"  Loaded {model} CI: {len(ci_df)} points")

        if not results:
            log(f"  No data found for {dataset_name}, skipping...")
            continue

        # Generate plots with CI (no regime overlays)
        if ci_data:
            log(f"\n  Generating plots with CI...")
            plot_density_quality_curves_with_ci(
                results, ci_data, dataset_name, display_name, FIGURES_DIR,
                with_regime_overlays=False, output_suffix=""
            )

        # Generate plots with CI and regime overlays
        if ci_data:
            log(f"\n  Generating plots with CI and regime overlays...")
            plot_density_quality_curves_with_ci(
                results, ci_data, dataset_name, display_name, FIGURES_DIR,
                with_regime_overlays=True, output_suffix="_regimes"
            )

        # Generate matching comparison plots (many-to-one vs one-to-one)
        log(f"\n  Generating matching comparison plot...")
        from density_quality_curves import plot_matching_comparison
        plot_matching_comparison(
            results, dataset_name, display_name, FIGURES_DIR,
            ci_data=ci_data if ci_data else None,
            with_regime_overlays=True,
        )

    log("\nDone.")


if __name__ == "__main__":
    main()
