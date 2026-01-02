#!/usr/bin/env python3
"""
Compute dialogue-level bootstrap 95% confidence intervals for density-quality curves.

This script:
1. Runs the threshold sweep with per-dialogue metric storage
2. Computes bootstrap CIs by resampling dialogues
3. Saves CI artifacts for reproducibility
4. Generates updated plots with CI bands

Bootstrap methodology:
- Unit: Dialogues (resample dialogues with replacement)
- CI method: Percentile bootstrap (2.5%, 97.5%)
- Default replicates: 1000 (configurable)
- Fixed random seed for reproducibility
- BOR is fixed per operating point (dataset-level quantity)

Usage:
    python paper/compute_density_quality_cis.py
    python paper/compute_density_quality_cis.py --bootstrap-n 2000 --seed 123

Outputs:
    - paper/results/sweep_{dataset}_{model}_per_dialogue.json: Per-dialogue metrics
    - paper/results/bootstrap_ci_{dataset}_{model}.csv: CI data
    - paper/figures/density_quality_{dataset}_ci.pdf/png: Plots with CI bands
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also add scripts directory for local imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from density_quality_curves import (
    # Configuration
    DATASETS,
    FIGURES_DIR,
    RESULTS_DIR,
    BOOTSTRAP_N_REPLICATES,
    BOOTSTRAP_SEED,
    # Data loading
    load_dataset,
    # Scoring functions
    get_neural_scores,
    get_texttiling_scores,
    get_csm_scores,
    # Sweep with per-dialogue data
    run_sweep_with_per_dialogue,
    # Bootstrap CI computation
    compute_sweep_bootstrap_cis,
    # I/O functions
    save_per_dialogue_data,
    load_per_dialogue_data,
    # Plotting
    plot_density_quality_curves_with_ci,
    # Utilities
    log,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CIs for density-quality curves"
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=BOOTSTRAP_N_REPLICATES,
        help=f"Number of bootstrap replicates (default: {BOOTSTRAP_N_REPLICATES})"
    )
    parser.add_argument(
        "--seed", type=int, default=BOOTSTRAP_SEED,
        help=f"Random seed for bootstrap (default: {BOOTSTRAP_SEED})"
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip model scoring, load existing per-dialogue data"
    )
    parser.add_argument(
        "--error-bars", action="store_true",
        help="Use error bars instead of shaded bands"
    )
    args = parser.parse_args()

    log("=" * 70)
    log("Density-Quality Curves with Bootstrap CIs")
    log(f"Bootstrap replicates: {args.bootstrap_n}")
    log(f"Random seed: {args.seed}")
    log("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_info in DATASETS.items():
        display_name = dataset_info["display"]
        log(f"\n{'='*70}")
        log(f"Dataset: {display_name}")
        log("=" * 70)

        # Load dialogues
        try:
            dialogues = load_dataset(dataset_name)
            log(f"Loaded {len(dialogues)} dialogues")
            total_gold = sum(len(d.gold_boundaries) for d in dialogues)
            log(f"Total gold boundaries: {total_gold}")
        except FileNotFoundError as e:
            log(f"SKIP: {e}")
            continue

        # Store results and CI data
        results = {}  # For plotting (DataFrame format)
        ci_data = {}  # For CI bands

        # Process each model (except random, which uses std across seeds)
        for model_name, get_scores_fn in [
            ("neural", get_neural_scores),
            ("texttiling", get_texttiling_scores),
            ("csm", get_csm_scores),
        ]:
            log(f"\n--- {model_name.upper()} ---")

            per_dialogue_path = RESULTS_DIR / f"sweep_{dataset_name}_{model_name}_per_dialogue.json"
            ci_csv_path = RESULTS_DIR / f"bootstrap_ci_{dataset_name}_{model_name}.csv"

            if args.skip_scoring and per_dialogue_path.exists():
                # Load existing per-dialogue data
                log(f"  Loading cached per-dialogue data...")
                sweep_points = load_per_dialogue_data(per_dialogue_path)
                log(f"  Loaded {len(sweep_points)} operating points")
            else:
                # Run scoring and sweep
                log(f"  Computing scores...")
                try:
                    scores = get_scores_fn(dialogues)
                except Exception as e:
                    log(f"  ERROR computing scores: {e}")
                    continue

                log(f"  Running sweep with per-dialogue storage...")
                sweep_points = run_sweep_with_per_dialogue(
                    dialogues, scores, dataset_name, model_name
                )
                log(f"  Generated {len(sweep_points)} operating points")

                # Save per-dialogue data
                save_per_dialogue_data(sweep_points, per_dialogue_path)
                log(f"  Saved: {per_dialogue_path}")

            # Compute bootstrap CIs
            log(f"  Computing bootstrap CIs ({args.bootstrap_n} replicates)...")
            ci_df = compute_sweep_bootstrap_cis(
                sweep_points,
                n_replicates=args.bootstrap_n,
                seed=args.seed
            )
            log(f"  Computed CIs for {len(ci_df)} metric-point pairs")

            # Save CI data
            ci_df.to_csv(ci_csv_path, index=False)
            log(f"  Saved: {ci_csv_path}")

            # Convert sweep points to DataFrame for plotting
            sweep_df = pd.DataFrame([
                {
                    "step": p.step,
                    "tau": p.tau,
                    "bor": p.bor,
                    "wf1": p.wf1,
                    "wf1_1to1": p.wf1_1to1,
                    "coverage": p.coverage,
                }
                for p in sweep_points
            ])

            results[model_name] = sweep_df
            ci_data[model_name] = ci_df

        # Load random baseline from existing sweep (uses std, not bootstrap)
        random_csv = RESULTS_DIR / f"sweep_{dataset_name}_random.csv"
        if random_csv.exists():
            log(f"\n--- RANDOM (from existing sweep) ---")
            results["random"] = pd.read_csv(random_csv)
            log(f"  Loaded {len(results['random'])} points")

        # Generate plots with CIs
        if results:
            log(f"\n--- Generating plots with CIs ---")

            # Standard plot with CI bands
            plot_density_quality_curves_with_ci(
                results, ci_data, dataset_name, display_name, FIGURES_DIR,
                show_error_bars=args.error_bars
            )

            # Plot with regime overlays
            plot_density_quality_curves_with_ci(
                results, ci_data, dataset_name, display_name, FIGURES_DIR,
                with_regime_overlays=True,
                output_suffix="_regimes",
                show_error_bars=args.error_bars
            )

    # Summary of artifacts
    log("\n" + "=" * 70)
    log("ARTIFACTS GENERATED")
    log("=" * 70)
    log("\nPer-dialogue data (JSON):")
    for f in sorted(RESULTS_DIR.glob("sweep_*_per_dialogue.json")):
        log(f"  {f}")
    log("\nBootstrap CI data (CSV):")
    for f in sorted(RESULTS_DIR.glob("bootstrap_ci_*.csv")):
        log(f"  {f}")
    log("\nPlots with CIs:")
    for f in sorted(FIGURES_DIR.glob("*_ci*.pdf")):
        log(f"  {f}")

    log("\nDone.")


if __name__ == "__main__":
    main()
