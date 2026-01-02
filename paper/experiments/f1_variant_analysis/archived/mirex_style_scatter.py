#!/usr/bin/env python3
"""
Generate MIREX-style scatter plots for dialogue segmentation.

Creates plots showing:
- X-axis: Density Ratio (|P|/|G|) = BOR
- Y-axis: Boundary F1 (W-F1 or Exact-F1)
- Each point = (method, threshold) operating point

Goal: Show that dialogue segmentation exhibits positive correlation
between density and F1, unlike MIREX which shows negative correlation
with optimal at BOR ≈ 1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent
results_dir = project_root / "paper" / "results"

# Colors for methods
METHOD_COLORS = {
    "neural": "steelblue",
    "csm": "coral",
    "texttiling": "forestgreen",
    "random": "gray"
}

METHOD_LABELS = {
    "neural": "Neural (Ours)",
    "csm": "CSM",
    "texttiling": "TextTiling",
    "random": "Random"
}


def load_sweep_data(dataset: str, methods: list) -> pd.DataFrame:
    """Load sweep data for multiple methods."""
    all_data = []

    for method in methods:
        csv_file = results_dir / f"sweep_{dataset}_{method}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df["method"] = method
            all_data.append(df)
        else:
            print(f"Warning: {csv_file} not found")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def plot_mirex_style_scatter(
    dataset: str,
    methods: list = ["neural", "csm", "texttiling"],
    output_prefix: str = "dialogue_density_scatter"
):
    """Create MIREX-style scatter plot for dialogue segmentation."""

    df = load_sweep_data(dataset, methods)
    if df.empty:
        print(f"No data for {dataset}")
        return

    # Create figure with two subplots: W-F1 and strict W-F1 (1-to-1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (f1_col, f1_label) in enumerate([
        ("wf1", "W-F1 (many-to-one)"),
        ("wf1_1to1", "W-F1 (one-to-one)")
    ]):
        ax = axes[ax_idx]

        all_bors = []
        all_f1s = []

        for method in methods:
            method_df = df[df["method"] == method]
            if method_df.empty:
                continue

            bors = method_df["bor"].values
            f1s = method_df[f1_col].values

            # Filter valid points
            mask = (bors > 0) & (bors < 10) & (f1s > 0)
            bors = bors[mask]
            f1s = f1s[mask]

            all_bors.extend(bors)
            all_f1s.extend(f1s)

            # Scatter plot
            ax.scatter(bors, f1s,
                      c=METHOD_COLORS.get(method, "black"),
                      alpha=0.6, s=30,
                      label=METHOD_LABELS.get(method, method))

        # Compute correlation
        if len(all_bors) >= 3:
            r, p = stats.pearsonr(all_bors, all_f1s)

            # Fit line for visualization
            z = np.polyfit(all_bors, all_f1s, 1)
            poly = np.poly1d(z)
            x_line = np.linspace(min(all_bors), max(all_bors), 100)
            ax.plot(x_line, poly(x_line), 'k--', alpha=0.5, linewidth=1)
        else:
            r, p = 0, 1

        # Reference line at BOR = 1
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='BOR = 1')

        # Labels and title
        ax.set_xlabel("Density Ratio (|P| / |G|)", fontsize=12)
        ax.set_ylabel(f1_label, fontsize=12)
        ax.set_title(f"{dataset.upper()}: {f1_label}\nr = {r:.3f} (p = {p:.4f})",
                    fontsize=14)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = script_dir
    plt.savefig(output_dir / f"{output_prefix}_{dataset}.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"{output_prefix}_{dataset}.pdf", bbox_inches='tight')
    print(f"Saved: {output_prefix}_{dataset}.png/pdf")

    plt.close()


def plot_combined_scatter(
    datasets: list = ["dialseg711", "superseg"],
    methods: list = ["neural", "csm", "texttiling"],
    output_name: str = "dialogue_density_scatter_combined"
):
    """Create combined scatter plot for multiple datasets."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for row, dataset in enumerate(datasets):
        df = load_sweep_data(dataset, methods)
        if df.empty:
            continue

        for col, (f1_col, f1_label) in enumerate([
            ("wf1", "W-F1 (many-to-one)"),
            ("wf1_1to1", "W-F1 (one-to-one)")
        ]):
            ax = axes[row, col]

            all_bors = []
            all_f1s = []

            for method in methods:
                method_df = df[df["method"] == method]
                if method_df.empty:
                    continue

                bors = method_df["bor"].values
                f1s = method_df[f1_col].values

                # Filter valid points
                mask = (bors > 0) & (bors < 10) & (f1s > 0)
                bors = bors[mask]
                f1s = f1s[mask]

                all_bors.extend(bors)
                all_f1s.extend(f1s)

                ax.scatter(bors, f1s,
                          c=METHOD_COLORS.get(method, "black"),
                          alpha=0.6, s=30,
                          label=METHOD_LABELS.get(method, method))

            # Correlation
            if len(all_bors) >= 3:
                r, p = stats.pearsonr(all_bors, all_f1s)
                z = np.polyfit(all_bors, all_f1s, 1)
                poly = np.poly1d(z)
                x_line = np.linspace(min(all_bors), max(all_bors), 100)
                ax.plot(x_line, poly(x_line), 'k--', alpha=0.5, linewidth=1)
            else:
                r, p = 0, 1

            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_xlabel("Density Ratio (|P| / |G|)", fontsize=11)
            ax.set_ylabel(f1_label, fontsize=11)
            ax.set_title(f"{dataset.upper()}: {f1_label}\nr = {r:.3f}", fontsize=12)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1)
            if row == 0 and col == 1:
                ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = script_dir
    plt.savefig(output_dir / f"{output_name}.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"{output_name}.pdf", bbox_inches='tight')
    print(f"Saved: {output_name}.png/pdf")

    plt.close()


def print_correlation_summary(
    datasets: list = ["dialseg711", "superseg"],
    methods: list = ["neural", "csm", "texttiling"]
):
    """Print correlation summary for all datasets."""

    print("\n" + "=" * 70)
    print("DIALOGUE SEGMENTATION: Density-Quality Correlation Summary")
    print("=" * 70)

    for dataset in datasets:
        df = load_sweep_data(dataset, methods)
        if df.empty:
            continue

        print(f"\n{dataset.upper()}:")
        print("-" * 50)

        for f1_col, f1_label in [("wf1", "W-F1 (many-to-one)"), ("wf1_1to1", "W-F1 (one-to-one)")]:
            all_bors = []
            all_f1s = []

            for method in methods:
                method_df = df[df["method"] == method]
                bors = method_df["bor"].values
                f1s = method_df[f1_col].values
                mask = (bors > 0) & (bors < 10) & (f1s > 0)
                all_bors.extend(bors[mask])
                all_f1s.extend(f1s[mask])

            if len(all_bors) >= 3:
                r, p = stats.pearsonr(all_bors, all_f1s)
                best_idx = np.argmax(all_f1s)
                print(f"  {f1_label}:")
                print(f"    Pearson r = {r:.3f} (p = {p:.2e})")
                print(f"    Best F1 = {all_f1s[best_idx]:.3f} at BOR = {all_bors[best_idx]:.2f}")

    print("\n" + "=" * 70)
    print("Key finding: POSITIVE correlation (r > 0) indicates density confound.")
    print("In healthy evaluation, we expect NEGATIVE correlation with optimal at BOR ≈ 1.")
    print("=" * 70)


if __name__ == "__main__":
    # Print summary
    print_correlation_summary()

    # Generate plots for each dataset
    for dataset in ["dialseg711", "superseg"]:
        plot_mirex_style_scatter(dataset)

    # Generate combined plot
    plot_combined_scatter()

    print("\nAll plots saved to:", script_dir)
