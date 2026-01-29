#!/usr/bin/env python3
"""
Generate combined 4-panel Figure 4 replica for GPT-5.2 boundary scoring.
"""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DIALSEG_JSON = PROJECT_ROOT / "paper" / "experiments" / "gpt52_dialseg711_figure4.json"
SUPERSEG_JSON = PROJECT_ROOT / "paper" / "experiments" / "gpt52_superseg_figure4.json"
OUTPUT_PNG = PROJECT_ROOT / "paper" / "experiments" / "gpt52_figure4_combined.png"
OUTPUT_PDF = PROJECT_ROOT / "paper" / "experiments" / "gpt52_figure4_combined.pdf"

print("Loading data...")
with open(DIALSEG_JSON) as f:
    dialseg = json.load(f)
with open(SUPERSEG_JSON) as f:
    superseg = json.load(f)

print("Generating combined figure...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

datasets = [
    (dialseg, "DialSeg711", axes[0]),
    (superseg, "SuperSeg", axes[1])
]

metrics = [
    ("wf1_m2o", "W-F1 (many-to-one)"),
    ("wf1_1to1", "W-F1 (one-to-one)")
]

for data, dataset_name, ax_row in datasets:
    sweep = data["sweep_points"]
    ci = data["bootstrap_ci"]

    # Sort by BOR
    bors = np.array([sp["bor"] for sp in sweep])
    idx = np.argsort(bors)
    bors_sorted = bors[idx]

    for ax_idx, (metric, metric_label) in enumerate(metrics):
        ax = ax_row[ax_idx]

        # Get values sorted by BOR
        wf1s = np.array([sp[metric] for sp in sweep])[idx]
        ci_lo = np.array([ci[str(sp["percentile"])][f"{metric}_lo"] for sp in sweep])[idx]
        ci_hi = np.array([ci[str(sp["percentile"])][f"{metric}_hi"] for sp in sweep])[idx]

        # Add region shading (very light)
        ax.axvspan(0, 1.0, color='#FFE4E1', alpha=0.3, zorder=0)  # Pink for under-seg
        ax.axvspan(1.0, 3.0, color='#E6F3FF', alpha=0.3, zorder=0)  # Blue for over-seg

        # BOR=1 vertical line
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # 95% CI shaded band
        ax.fill_between(bors_sorted, ci_lo, ci_hi, color='#8B5CF6', alpha=0.25, zorder=2)

        # Main curve
        ax.plot(bors_sorted, wf1s, '-', color='#8B5CF6', linewidth=2.5,
                marker='o', markersize=4, label='GPT-5.2', zorder=3)

        # Find and mark peak
        peak_idx = np.argmax(wf1s)
        peak_bor = bors_sorted[peak_idx]
        peak_wf1 = wf1s[peak_idx]
        ax.scatter([peak_bor], [peak_wf1], s=100, c='#8B5CF6', edgecolors='white',
                   linewidth=2, zorder=4, marker='*')

        # Axes and labels
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('BOR (Boundary Oversegmentation Ratio)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{dataset_name}: {metric_label}')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='lower right', framealpha=0.9)

        # Add annotation for BOR=1 line
        if ax_idx == 0:
            ax.text(1.02, 0.05, 'BOR=1', rotation=90, fontsize=9,
                    color='gray', alpha=0.8, transform=ax.get_xaxis_transform())

# Main title
fig.suptitle('DialSeg711 and SuperSeg: Density–Quality Regime Plots with 95% CIs',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save outputs
print(f"Saving {OUTPUT_PNG}...")
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Saving {OUTPUT_PDF}...")
fig.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')

plt.close(fig)

print("\n" + "=" * 60)
print("COMBINED FIGURE 4 COMPLETE")
print("=" * 60)
print(f"PNG: {OUTPUT_PNG}")
print(f"PDF: {OUTPUT_PDF}")
print()
print("Summary:")
print(f"  DialSeg711 peak (m2o): {dialseg['peak_wf1_m2o']:.4f} at BOR={dialseg['peak_bor_m2o']:.3f}")
print(f"  DialSeg711 peak (1to1): {dialseg['peak_wf1_1to1']:.4f} at BOR={dialseg['peak_bor_1to1']:.3f}")
print(f"  SuperSeg peak (m2o): {superseg['peak_wf1_m2o']:.4f} at BOR={superseg['peak_bor_m2o']:.3f}")
print(f"  SuperSeg peak (1to1): {superseg['peak_wf1_1to1']:.4f} at BOR={superseg['peak_bor_1to1']:.3f}")
