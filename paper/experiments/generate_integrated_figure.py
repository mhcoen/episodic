#!/usr/bin/env python3
"""
Generate integrated TACL figure: All methods density-quality curves.

Layout: 2 rows × 3 columns
- Rows: DialSeg711, SuperSeg
- Cols: W-F1 (m2o), W-F1 (1to1), Exact F1 (w=0)

ALL 5 METHODS IN ALL 6 PANELS:
- GPT-5.2: Purple (#8B5CF6)
- DistilBERT: Blue (#0072B2)
- TextTiling: Green (#009E73)
- CSM: Orange (#E69F00)
- Random: Gray (#666666)

ALL SOLID LINES - NO DASHED
ALL METHODS HAVE 95% CI BANDS

DATA SOURCES:
- DistilBERT: paper/results/sweep_{dataset}_neural.csv
- DistilBERT CIs: paper/results/bootstrap_ci_{dataset}_neural.csv
- Baselines: paper/results/sweep_{dataset}_{method}.csv
- Baseline CIs: paper/results/bootstrap_ci_{dataset}_{method}.csv
- GPT-5.2: paper/experiments/gpt52_{dataset}_figure4.json
- GPT-5.2 scores: .gpt52_*_cache/cache.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "paper" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
DATASETS = ["dialseg711", "superseg"]
DATASET_LABELS = {"dialseg711": "DialSeg711", "superseg": "SuperSeg"}

# All methods to plot
METHODS = ["gpt52", "neural", "texttiling", "csm", "random"]
METHOD_LABELS = {
    "gpt52": "GPT-5.2",
    "neural": "DistilBERT",
    "texttiling": "TextTiling",
    "csm": "CSM",
    "random": "Random",
}

# Colors - ALL DISTINCT
COLORS = {
    "gpt52": "#8B5CF6",      # Purple
    "neural": "#0072B2",     # Blue
    "texttiling": "#009E73", # Green
    "csm": "#E69F00",        # Orange
    "random": "#CC79A7",     # Pink/Magenta
}

# Metrics
METRICS = [
    ("wf1", "W-F1 (many-to-one)"),
    ("wf1_1to1", "W-F1 (one-to-one)"),
    ("exact_f1", "Exact F1 (w=0)"),
]

# Axis limits per dataset
X_LIMITS = {"dialseg711": 2.2, "superseg": 1.35}

MIN_GAP = 2

print("=" * 70)
print("INTEGRATED FIGURE GENERATOR - COMPLETE REDO")
print("=" * 70)
print()

# ============================================================
# LOAD ALL SWEEP DATA
# ============================================================
print("Loading sweep data...")

sweep_data = {dataset: {} for dataset in DATASETS}
for dataset in DATASETS:
    for method in ["neural", "texttiling", "csm", "random"]:
        csv_path = RESULTS_DIR / f"sweep_{dataset}_{method}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sweep_data[dataset][method] = df
            print(f"  {dataset}/{method}: {len(df)} points")

# ============================================================
# LOAD ALL CI DATA
# ============================================================
print("\nLoading CI data...")

ci_data = {dataset: {} for dataset in DATASETS}
for dataset in DATASETS:
    for method in ["neural", "texttiling", "csm"]:  # random has no CIs
        ci_path = RESULTS_DIR / f"bootstrap_ci_{dataset}_{method}.csv"
        if ci_path.exists():
            df = pd.read_csv(ci_path)
            ci_data[dataset][method] = df
            print(f"  {dataset}/{method}: {len(df)} CI records")

# ============================================================
# LOAD GPT-5.2 DATA
# ============================================================
print("\nLoading GPT-5.2 data...")

gpt52_data = {}
for dataset in DATASETS:
    json_path = EXPERIMENTS_DIR / f"gpt52_{dataset}_figure4.json"
    if json_path.exists():
        with open(json_path) as f:
            gpt52_data[dataset] = json.load(f)
        print(f"  {dataset}: loaded")

# ============================================================
# COMPUTE GPT-5.2 EXACT F1 WITH BOOTSTRAP CIs
# ============================================================
print("\nComputing GPT-5.2 Exact F1 with bootstrap CIs...")

def greedy_nms(scores_by_pos, tau):
    """Greedy NMS prediction with min_gap=2."""
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

def compute_exact_f1(predicted, gold):
    """Exact F1 with window=0."""
    if not gold:
        return 0.0 if predicted else 1.0
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def load_gpt52_scores_and_gold(dataset):
    """Load GPT-5.2 scores and gold boundaries from cache."""
    cache_paths = [
        PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json",
        PROJECT_ROOT / f".gpt52_{dataset}_cache" / "cache.json",
        PROJECT_ROOT / ".gpt52_superseg_cache" / "cache.json",
    ]

    cache = {}
    for cache_path in cache_paths:
        if cache_path.exists():
            with open(cache_path) as f:
                cache.update(json.load(f))

    dataset_path = PROJECT_ROOT / "datasets" / dataset / "segmentation_file_test.json"
    with open(dataset_path) as f:
        data = json.load(f)

    scores_by_dialogue = defaultdict(dict)
    gold_by_dialogue = {}

    dial_data = data.get("dial_data", data)
    dialogue_id = 0
    for source_key, source_dialogs in dial_data.items():
        if not isinstance(source_dialogs, list):
            continue
        for dialog in source_dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

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

            gold_by_dialogue[dialogue_id] = boundaries
            dialogue_id += 1

    for key, entry in cache.items():
        if not key.startswith(dataset):
            continue
        parts = key.split("_")
        if len(parts) >= 4:
            try:
                did = int(parts[1])
                pos = int(parts[2])
                if did in gold_by_dialogue:
                    score = entry.get("score", 0)
                    if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
                        scores_by_dialogue[did][pos] = score
            except ValueError:
                pass

    return dict(scores_by_dialogue), gold_by_dialogue

def compute_gpt52_exact_f1_with_ci(dataset, n_bootstrap=1000):
    """Compute Exact F1 sweep with bootstrap CIs for GPT-5.2."""
    scores_by_dialogue, gold_by_dialogue = load_gpt52_scores_and_gold(dataset)

    if not scores_by_dialogue:
        return None

    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    percentiles = [1, 2, 3] + list(range(5, 100, 5))
    dialogue_ids = list(scores_by_dialogue.keys())
    n_dialogues = len(dialogue_ids)

    # Total gold from ALL dialogues (for BOR denominator)
    total_gold_all = sum(len(g) for g in gold_by_dialogue.values())

    # Point estimates
    sweep_points = []
    for pct in percentiles:
        tau = np.percentile(all_scores, pct)
        total_exact_f1 = 0.0
        total_pred = 0
        n_with_gold = 0

        for did in dialogue_ids:
            scores = scores_by_dialogue[did]
            gold = gold_by_dialogue.get(did, set())
            predicted = greedy_nms(scores, tau)

            # Always count predictions for BOR
            total_pred += len(predicted)

            # Only compute F1 for dialogues with gold
            if gold:
                ef1 = compute_exact_f1(predicted, gold)
                total_exact_f1 += ef1
                n_with_gold += 1

        if n_with_gold > 0:
            sweep_points.append({
                "percentile": pct,
                "tau": float(tau),
                "bor": total_pred / total_gold_all if total_gold_all > 0 else 0,
                "exact_f1": total_exact_f1 / n_with_gold,
            })

    # Bootstrap CIs
    print(f"    Running {n_bootstrap} bootstrap iterations for {dataset}...")
    bootstrap_exact_f1 = defaultdict(list)
    bootstrap_bor = defaultdict(list)

    np.random.seed(42)
    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"      {b + 1}/{n_bootstrap}")

        # Resample dialogues
        resampled_ids = np.random.choice(dialogue_ids, size=n_dialogues, replace=True)

        # Recompute total_gold for bootstrap sample (from resampled gold)
        resampled_gold_total = sum(len(gold_by_dialogue.get(did, set())) for did in resampled_ids)

        for pct in percentiles:
            tau = np.percentile(all_scores, pct)
            total_exact_f1 = 0.0
            total_pred = 0
            n_with_gold = 0

            for did in resampled_ids:
                scores = scores_by_dialogue[did]
                gold = gold_by_dialogue.get(did, set())
                predicted = greedy_nms(scores, tau)

                # Always count predictions for BOR
                total_pred += len(predicted)

                # Only compute F1 for dialogues with gold
                if gold:
                    ef1 = compute_exact_f1(predicted, gold)
                    total_exact_f1 += ef1
                    n_with_gold += 1

            if n_with_gold > 0:
                bootstrap_exact_f1[pct].append(total_exact_f1 / n_with_gold)
                bootstrap_bor[pct].append(total_pred / resampled_gold_total if resampled_gold_total > 0 else 0)

    # Compute CIs
    ci_data = {}
    for pct in percentiles:
        if pct in bootstrap_exact_f1:
            ef1_arr = np.array(bootstrap_exact_f1[pct])
            ci_data[pct] = {
                "exact_f1_lo": float(np.percentile(ef1_arr, 2.5)),
                "exact_f1_hi": float(np.percentile(ef1_arr, 97.5)),
            }

    return sweep_points, ci_data

gpt52_exact_f1 = {}
gpt52_exact_f1_ci = {}
for dataset in DATASETS:
    print(f"  {dataset}:")
    result = compute_gpt52_exact_f1_with_ci(dataset, n_bootstrap=1000)
    if result:
        gpt52_exact_f1[dataset], gpt52_exact_f1_ci[dataset] = result
        print(f"    Computed {len(gpt52_exact_f1[dataset])} points with CIs")

# ============================================================
# GENERATE FIGURE
# ============================================================
print("\nGenerating figure...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for row_idx, dataset in enumerate(DATASETS):
    dataset_label = DATASET_LABELS[dataset]
    x_max = X_LIMITS[dataset]

    for col_idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[row_idx, col_idx]

        # ----------------------------------------
        # REGIME SHADING
        # ----------------------------------------
        ax.axvspan(0, 1.0, color='#FFE4E1', alpha=0.4, zorder=0)  # Pink for BOR < 1
        ax.axvspan(1.0, x_max, color='#E6F3FF', alpha=0.4, zorder=0)  # Blue for BOR > 1
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # ----------------------------------------
        # PLOT ALL METHODS
        # ----------------------------------------

        # GPT-5.2 metric key mapping
        gpt_metric_key = "wf1_m2o" if metric_key == "wf1" else metric_key

        for method in METHODS:
            color = COLORS[method]
            label = METHOD_LABELS[method]

            if method == "gpt52":
                # GPT-5.2 from JSON
                gpt_data = gpt52_data.get(dataset)
                if gpt_data is None:
                    continue

                if metric_key == "exact_f1":
                    # Use computed exact_f1 with CIs
                    if dataset not in gpt52_exact_f1:
                        continue
                    sweep = gpt52_exact_f1[dataset]
                    ci = gpt52_exact_f1_ci.get(dataset, {})
                    metric_col = "exact_f1"
                else:
                    sweep = gpt_data.get("sweep_points", [])
                    ci = gpt_data.get("bootstrap_ci", {})
                    metric_col = gpt_metric_key

                if not sweep or metric_col not in sweep[0]:
                    continue

                bors = np.array([sp["bor"] for sp in sweep])
                vals = np.array([sp[metric_col] for sp in sweep])
                idx = np.argsort(bors)
                bors_sorted = bors[idx]
                vals_sorted = vals[idx]

                # Plot line - SOLID (same linewidth as all methods)
                ax.plot(bors_sorted, vals_sorted, '-', color=color, linewidth=2,
                        label=label, zorder=5)

                # Plot CI band
                if ci:
                    ci_lo = []
                    ci_hi = []
                    for sp in sweep:
                        pct = sp.get("percentile")
                        pct_key = str(pct) if str(pct) in ci else pct
                        if pct_key in ci:
                            lo_key = f"{metric_col}_lo"
                            hi_key = f"{metric_col}_hi"
                            if lo_key in ci[pct_key]:
                                ci_lo.append(ci[pct_key][lo_key])
                                ci_hi.append(ci[pct_key][hi_key])
                            else:
                                ci_lo.append(sp[metric_col])
                                ci_hi.append(sp[metric_col])
                        else:
                            ci_lo.append(sp[metric_col])
                            ci_hi.append(sp[metric_col])

                    ci_lo = np.array(ci_lo)[idx]
                    ci_hi = np.array(ci_hi)[idx]
                    ax.fill_between(bors_sorted, ci_lo, ci_hi, color=color, alpha=0.2, zorder=2)

                # GPT-5.2 ceiling annotation removed - max BOR now matches structural ceiling

            else:
                # Other methods from CSV
                df = sweep_data.get(dataset, {}).get(method)
                if df is None or metric_key not in df.columns:
                    continue

                # Random: aggregate across seeds by step (mean ± std)
                if method == "random":
                    grouped = df.groupby("step").agg({
                        "bor": "mean",
                        metric_key: ["mean", "std"],
                    }).reset_index()
                    grouped.columns = ["step", "bor", "val_mean", "val_std"]
                    grouped = grouped.sort_values("bor")
                    bors = grouped["bor"].values
                    vals = grouped["val_mean"].values
                    vals_std = grouped["val_std"].values

                    # Plot line - SOLID
                    ax.plot(bors, vals, '-', color=color, linewidth=2,
                            label=label, zorder=4)

                    # Plot CI band (mean ± 1.96*std for 95% CI)
                    ci_lo = vals - 1.96 * vals_std
                    ci_hi = vals + 1.96 * vals_std
                    ax.fill_between(bors, ci_lo, ci_hi, color=color, alpha=0.2, zorder=2)
                else:
                    df_sorted = df.sort_values("bor")
                    bors = df_sorted["bor"].values
                    vals = df_sorted[metric_key].values

                    # Plot line - SOLID
                    ax.plot(bors, vals, '-', color=color, linewidth=2,
                            label=label, zorder=4)

                # Plot CI band if available
                ci_df = ci_data.get(dataset, {}).get(method)
                if ci_df is not None:
                    metric_ci = ci_df[ci_df["metric"] == metric_key].copy()
                    if len(metric_ci) > 0:
                        metric_ci = metric_ci.sort_values("bor")
                        ax.fill_between(metric_ci["bor"], metric_ci["ci_low"], metric_ci["ci_high"],
                                        color=color, alpha=0.2, zorder=2)

        # ----------------------------------------
        # FORMATTING
        # ----------------------------------------
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("BOR", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f"{dataset_label}: {metric_label}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Legend on first panel only
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='lower right', fontsize=8)

fig.suptitle("Density-Quality Regime Comparison: All Methods",
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_png = OUTPUT_DIR / "fig_integrated_density_matching_scorer.png"
output_pdf = OUTPUT_DIR / "fig_integrated_density_matching_scorer.pdf"

print(f"\nSaving {output_png}...")
fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Saving {output_pdf}...")
fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

plt.close(fig)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Output: {output_png}")
print(f"Output: {output_pdf}")
