#!/usr/bin/env python3
"""
Generate Figure 4 panel for DialSeg711 using cached GPT-5.2 scores.
No API calls - uses existing cached data only.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_FIGURE4 = PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json"
CACHE_SANITY = PROJECT_ROOT / ".gpt52_cache" / "cache.json"
DATASET_FILE = PROJECT_ROOT / "datasets" / "dialseg711" / "segmentation_file_test.json"
OUTPUT_JSON = PROJECT_ROOT / "paper" / "experiments" / "gpt52_dialseg711_figure4.json"
OUTPUT_PNG = PROJECT_ROOT / "paper" / "experiments" / "gpt52_dialseg711_figure4.png"

# Config
TAU_PERCENTILES = list(range(5, 100, 5))  # 5, 10, 15, ..., 95
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
MIN_GAP = 2

print("=" * 60, flush=True)
print("GPT-5.2 DIALSEG711 FIGURE 4 GENERATION", flush=True)
print("=" * 60, flush=True)

# Load caches
print("\nLoading caches...", flush=True)

scores_by_dialogue = defaultdict(dict)

# Load figure4 cache (correct polarity)
if CACHE_FIGURE4.exists():
    with open(CACHE_FIGURE4) as f:
        cache = json.load(f)
    count = 0
    for key, entry in cache.items():
        if key.startswith("dialseg711_"):
            dialogue_id = entry["dialogue_id"]
            position = entry["position"]
            score = entry["score"]
            if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
                scores_by_dialogue[dialogue_id][position] = score
                count += 1
    print(f"  .gpt52_figure4_cache: {count} dialseg711 scores", flush=True)

# Load sanity check cache (FLIP polarity)
if CACHE_SANITY.exists():
    with open(CACHE_SANITY) as f:
        cache = json.load(f)
    count = 0
    for key, entry in cache.items():
        if key.startswith("dialseg711_"):
            dialogue_id = entry["dialogue_id"]
            position = entry["position"]
            # FLIP the score
            score = -entry["score"]
            if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
                # Only add if not already present from figure4 cache
                if position not in scores_by_dialogue[dialogue_id]:
                    scores_by_dialogue[dialogue_id][position] = score
                    count += 1
    print(f"  .gpt52_cache: {count} dialseg711 scores (FLIPPED)", flush=True)

n_dialogues = len(scores_by_dialogue)
n_boundaries = sum(len(s) for s in scores_by_dialogue.values())
print(f"  Total: {n_boundaries} scores across {n_dialogues} dialogues", flush=True)

# Load gold boundaries
print("\nLoading gold boundaries...", flush=True)

with open(DATASET_FILE) as f:
    data = json.load(f)

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

n_gold = sum(len(g) for g in gold_by_dialogue.values())
print(f"  {n_gold} gold boundaries across {len(gold_by_dialogue)} dialogues", flush=True)

# Metrics functions
def greedy_nms_predict(scores_by_pos: Dict[int, float], tau: float) -> Set[int]:
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

def compute_wf1(predicted: Set[int], gold: Set[int], one_to_one: bool = False, k: int = 3) -> float:
    if not gold:
        return 0.0 if predicted else 1.0

    if one_to_one:
        # Greedy one-to-one matching
        pred_list = sorted(predicted)
        gold_list = sorted(gold)
        pairs = []
        for p in pred_list:
            for g in gold_list:
                if abs(p - g) <= k:
                    pairs.append((abs(p - g), p, g))
        pairs.sort()
        matched_pred = set()
        matched_gold = set()
        for dist, p, g in pairs:
            if p not in matched_pred and g not in matched_gold:
                matched_pred.add(p)
                matched_gold.add(g)
        tp = len(matched_pred)
    else:
        # Many-to-one matching
        matched_pred = set()
        matched_gold = set()
        for p in predicted:
            for g in gold:
                if abs(p - g) <= k and g not in matched_gold:
                    matched_pred.add(p)
                    matched_gold.add(g)
                    break
        tp = len(matched_pred)

    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold) if one_to_one else len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_sweep(scores_by_dialogue: Dict, gold_by_dialogue: Dict) -> List[Dict]:
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    sweep_points = []
    for pct in TAU_PERCENTILES:
        tau = np.percentile(all_scores, pct)

        total_wf1_m2o = 0.0
        total_wf1_1to1 = 0.0
        total_pred = 0
        total_gold = 0
        n = 0
        per_dialogue_m2o = {}
        per_dialogue_1to1 = {}

        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            if not gold:
                continue

            predicted = greedy_nms_predict(scores_by_pos, tau)
            wf1_m2o = compute_wf1(predicted, gold, one_to_one=False)
            wf1_1to1 = compute_wf1(predicted, gold, one_to_one=True)

            per_dialogue_m2o[dialogue_id] = wf1_m2o
            per_dialogue_1to1[dialogue_id] = wf1_1to1
            total_wf1_m2o += wf1_m2o
            total_wf1_1to1 += wf1_1to1
            total_pred += len(predicted)
            total_gold += len(gold)
            n += 1

        if n > 0:
            sweep_points.append({
                "percentile": pct,
                "tau": float(tau),
                "bor": total_pred / total_gold if total_gold > 0 else 0,
                "wf1_m2o": total_wf1_m2o / n,
                "wf1_1to1": total_wf1_1to1 / n,
                "per_dialogue_m2o": per_dialogue_m2o,
                "per_dialogue_1to1": per_dialogue_1to1,
            })

    return sweep_points

# Compute main sweep
print("\nComputing threshold sweep...", flush=True)
sweep_points = compute_sweep(dict(scores_by_dialogue), gold_by_dialogue)

best_m2o = max(sweep_points, key=lambda x: x["wf1_m2o"])
best_1to1 = max(sweep_points, key=lambda x: x["wf1_1to1"])
print(f"  Peak W-F1 (m2o): {best_m2o['wf1_m2o']:.4f} at BOR={best_m2o['bor']:.3f}", flush=True)
print(f"  Peak W-F1 (1to1): {best_1to1['wf1_1to1']:.4f} at BOR={best_1to1['bor']:.3f}", flush=True)

# Bootstrap CIs
print(f"\nBootstrap ({N_BOOTSTRAP} iterations)...", flush=True)
rng = np.random.RandomState(BOOTSTRAP_SEED)
dialogue_ids = list(scores_by_dialogue.keys())
n_dial = len(dialogue_ids)

bootstrap_wf1_m2o = defaultdict(list)
bootstrap_wf1_1to1 = defaultdict(list)
bootstrap_bor = defaultdict(list)

for b in range(N_BOOTSTRAP):
    if (b + 1) % 100 == 0:
        print(f"  Bootstrap: {b + 1}/{N_BOOTSTRAP}", flush=True)

    sampled_ids = rng.choice(dialogue_ids, size=n_dial, replace=True)
    resampled_scores = {}
    resampled_gold = {}
    for new_id, orig_id in enumerate(sampled_ids):
        resampled_scores[new_id] = scores_by_dialogue[orig_id]
        resampled_gold[new_id] = gold_by_dialogue.get(orig_id, set())

    sweep = compute_sweep(resampled_scores, resampled_gold)
    for sp in sweep:
        pct = sp["percentile"]
        bootstrap_wf1_m2o[pct].append(sp["wf1_m2o"])
        bootstrap_wf1_1to1[pct].append(sp["wf1_1to1"])
        bootstrap_bor[pct].append(sp["bor"])

# Compute CIs
print("\nComputing CIs...", flush=True)
ci_data = {}
for pct in TAU_PERCENTILES:
    m2o = np.array(bootstrap_wf1_m2o[pct])
    o2o = np.array(bootstrap_wf1_1to1[pct])
    bor = np.array(bootstrap_bor[pct])

    ci_data[pct] = {
        "wf1_m2o_lo": float(np.percentile(m2o, 2.5)),
        "wf1_m2o_hi": float(np.percentile(m2o, 97.5)),
        "wf1_1to1_lo": float(np.percentile(o2o, 2.5)),
        "wf1_1to1_hi": float(np.percentile(o2o, 97.5)),
        "bor_lo": float(np.percentile(bor, 2.5)),
        "bor_hi": float(np.percentile(bor, 97.5)),
    }

# Save JSON
print(f"\nSaving results to {OUTPUT_JSON}...", flush=True)
results = {
    "description": "GPT-5.2 DialSeg711 Figure 4 (from cached scores)",
    "n_dialogues": n_dialogues,
    "n_boundaries": n_boundaries,
    "n_gold_boundaries": n_gold,
    "n_bootstrap": N_BOOTSTRAP,
    "peak_wf1_m2o": best_m2o["wf1_m2o"],
    "peak_bor_m2o": best_m2o["bor"],
    "peak_wf1_1to1": best_1to1["wf1_1to1"],
    "peak_bor_1to1": best_1to1["bor"],
    "sweep_points": [{k: v for k, v in sp.items() if not k.startswith("per_dialogue")} for sp in sweep_points],
    "bootstrap_ci": ci_data,
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

# Generate plot
print(f"\nGenerating plot {OUTPUT_PNG}...", flush=True)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bors = [sp["bor"] for sp in sweep_points]
    idx = np.argsort(bors)
    bors = np.array(bors)[idx]

    for ax_idx, (metric, title) in enumerate([("wf1_m2o", "W-F1 (many-to-one)"), ("wf1_1to1", "W-F1 (one-to-one)")]):
        ax = axes[ax_idx]

        wf1s = np.array([sp[metric] for sp in sweep_points])[idx]
        ci_lo = np.array([ci_data[sp["percentile"]][f"{metric}_lo"] for sp in sweep_points])[idx]
        ci_hi = np.array([ci_data[sp["percentile"]][f"{metric}_hi"] for sp in sweep_points])[idx]

        ax.fill_between(bors, ci_lo, ci_hi, color='#8B5CF6', alpha=0.2, label='95% CI')
        ax.plot(bors, wf1s, '-', color='#8B5CF6', linewidth=2.5, marker='o', markersize=4, label='GPT-5.2')

        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('BOR')
        ax.set_ylabel(title)
        ax.set_title(f'DialSeg711: {title}')
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

    fig.suptitle('GPT-5.2 Boundary Scoring: DialSeg711', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print("  Plot saved!", flush=True)

except Exception as e:
    print(f"  Plot error: {e}", flush=True)

print("\n" + "=" * 60, flush=True)
print("COMPLETE", flush=True)
print("=" * 60, flush=True)
print(f"Peak W-F1 (m2o): {best_m2o['wf1_m2o']:.4f} at BOR={best_m2o['bor']:.3f}", flush=True)
print(f"Peak W-F1 (1to1): {best_1to1['wf1_1to1']:.4f} at BOR={best_1to1['bor']:.3f}", flush=True)
print(f"CI width at peak (m2o): {ci_data[best_m2o['percentile']]['wf1_m2o_hi'] - ci_data[best_m2o['percentile']]['wf1_m2o_lo']:.4f}", flush=True)
