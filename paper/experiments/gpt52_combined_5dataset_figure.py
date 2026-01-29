#!/usr/bin/env python3
"""
Generate combined 5-dataset Figure 4 with all GPT-5.2 results.
"""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper" / "experiments"

# Load all datasets
datasets = [
    ("DialSeg711", OUTPUT_DIR / "gpt52_dialseg711_figure4.json"),
    ("SuperSeg", OUTPUT_DIR / "gpt52_superseg_figure4.json"),
    ("DailyDialog", OUTPUT_DIR / "gpt52_dailydialog_figure4.json"),
    ("TopicalChat", OUTPUT_DIR / "gpt52_topicalchat_figure4.json"),
    ("TIAGE", OUTPUT_DIR / "gpt52_tiage_figure4.json"),
]

print("Loading data...")
data_all = {}
for name, path in datasets:
    if path.exists():
        with open(path) as f:
            data_all[name] = json.load(f)
        print(f"  Loaded {name}")
    else:
        print(f"  Missing: {name}")

print(f"\nGenerating combined figure with {len(data_all)} datasets...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create 5-row x 2-column figure
fig, axes = plt.subplots(5, 2, figsize=(14, 20))

colors = {
    "DialSeg711": "#8B5CF6",
    "SuperSeg": "#06B6D4",
    "DailyDialog": "#F59E0B",
    "TopicalChat": "#10B981",
    "TIAGE": "#EF4444",
}

metrics = [
    ("wf1_m2o", "W-F1 (many-to-one)"),
    ("wf1_1to1", "W-F1 (one-to-one)")
]

for row_idx, (name, _) in enumerate(datasets):
    if name not in data_all:
        continue

    data = data_all[name]
    sweep = data["sweep_points"]
    ci = data["bootstrap_ci"]
    color = colors.get(name, "#8B5CF6")

    # Sort by BOR
    bors = np.array([sp["bor"] for sp in sweep])
    idx = np.argsort(bors)
    bors_sorted = bors[idx]

    for col_idx, (metric, metric_label) in enumerate(metrics):
        ax = axes[row_idx, col_idx]

        # Get values sorted by BOR
        wf1s = np.array([sp[metric] for sp in sweep])[idx]

        # Get CIs
        ci_lo = []
        ci_hi = []
        for sp in sweep:
            pct = sp["percentile"]
            pct_key = str(pct) if str(pct) in ci else pct
            if pct_key in ci:
                ci_lo.append(ci[pct_key][f"{metric}_lo"])
                ci_hi.append(ci[pct_key][f"{metric}_hi"])
            else:
                ci_lo.append(wf1s[list(bors).index(sp["bor"])])
                ci_hi.append(wf1s[list(bors).index(sp["bor"])])

        ci_lo = np.array(ci_lo)[idx]
        ci_hi = np.array(ci_hi)[idx]

        # Add region shading (very light)
        ax.axvspan(0, 1.0, color='#FFE4E1', alpha=0.3, zorder=0)
        ax.axvspan(1.0, 10.0, color='#E6F3FF', alpha=0.3, zorder=0)

        # BOR=1 vertical line
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # 95% CI shaded band
        ax.fill_between(bors_sorted, ci_lo, ci_hi, color=color, alpha=0.25, zorder=2)

        # Main curve
        ax.plot(bors_sorted, wf1s, '-', color=color, linewidth=2.5,
                marker='o', markersize=4, label='GPT-5.2', zorder=3)

        # Find and mark peak
        peak_idx = np.argmax(wf1s)
        peak_bor = bors_sorted[peak_idx]
        peak_wf1 = wf1s[peak_idx]
        ax.scatter([peak_bor], [peak_wf1], s=100, c=color, edgecolors='white',
                   linewidth=2, zorder=4, marker='*')

        # Axes and labels
        max_bor = max(2.5, max(bors_sorted) * 1.1)
        ax.set_xlim(0, min(max_bor, 5.0))
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('BOR')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{name}: {metric_label}')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Add peak annotation
        ax.annotate(f'Peak: {peak_wf1:.3f}\nBOR={peak_bor:.2f}',
                    xy=(peak_bor, peak_wf1), xytext=(peak_bor + 0.3, peak_wf1 - 0.15),
                    fontsize=8, alpha=0.8,
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

# Main title
fig.suptitle('GPT-5.2 Boundary Scoring: Density–Quality Regime Plots (5 Datasets)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save outputs
output_png = OUTPUT_DIR / "gpt52_5dataset_combined.png"
output_pdf = OUTPUT_DIR / "gpt52_5dataset_combined.pdf"

print(f"Saving {output_png}...")
fig.savefig(output_png, dpi=200, bbox_inches='tight', facecolor='white')

print(f"Saving {output_pdf}...")
fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

plt.close(fig)

# Print summary table
print("\n" + "=" * 70)
print("GPT-5.2 RESULTS SUMMARY (All 5 Datasets)")
print("=" * 70)
print(f"{'Dataset':<14} | {'Peak W-F1 (m2o)':>15} | {'Peak BOR (m2o)':>14} | {'Peak W-F1 (1to1)':>16}")
print("-" * 70)

for name, _ in datasets:
    if name not in data_all:
        continue
    d = data_all[name]
    print(f"{name:<14} | {d['peak_wf1_m2o']:>15.4f} | {d['peak_bor_m2o']:>14.3f} | {d['peak_wf1_1to1']:>16.4f}")

print("=" * 70)
print(f"\nOutputs:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")
