#!/usr/bin/env python3
"""
Fix SuperSeg GPT-5.2 BOR calculation.

Bug: The original script skipped dialogues without gold boundaries,
which excluded their predictions from the BOR calculation.

Fix: Count predictions from ALL dialogues for BOR, but only average
W-F1 over dialogues WITH gold boundaries.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = PROJECT_ROOT / ".gpt52_superseg_cache" / "cache.json"
DATASET_FILE = PROJECT_ROOT / "datasets" / "superseg" / "segmentation_file_test.json"
OUTPUT_JSON = PROJECT_ROOT / "paper" / "experiments" / "gpt52_superseg_figure4.json"

MIN_GAP = 2
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
TAU_PERCENTILES = [1, 2, 3] + list(range(5, 100, 5))
EXPLICIT_TAUS = [-30, -28, -26, -24, -22, -20, -18, -16, -15, -14]

print("=" * 60)
print("Fixing SuperSeg GPT-5.2 BOR Calculation")
print("=" * 60)

# Load cache
with open(CACHE_FILE) as f:
    cache = json.load(f)

# Load dataset
with open(DATASET_FILE) as f:
    data = json.load(f)

# Build dialogues
dialogues = []
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

        dialogues.append({
            "dialogue_id": dialogue_id,
            "gold_boundaries": boundaries,
        })
        dialogue_id += 1

# Build scores and gold dicts
scores_by_dialogue = defaultdict(dict)
gold_by_dialogue = {}

for dialogue in dialogues:
    gold_by_dialogue[dialogue["dialogue_id"]] = dialogue["gold_boundaries"]

for key, entry in cache.items():
    if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
        dialogue_id = entry["dialogue_id"]
        position = entry["position"]
        score = entry["score"]
        scores_by_dialogue[dialogue_id][position] = score

# Total gold (across ALL dialogues, for BOR denominator)
total_gold_all = sum(len(g) for g in gold_by_dialogue.values())
print(f"Total dialogues: {len(dialogues)}")
print(f"Dialogues with gold: {sum(1 for g in gold_by_dialogue.values() if g)}")
print(f"Total gold boundaries: {total_gold_all}")
print(f"Total scored positions: {sum(len(s) for s in scores_by_dialogue.values())}")

# NMS and metrics
def greedy_nms_predict(scores_by_pos, tau):
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

def compute_wf1(predicted, gold, one_to_one=False, k=3):
    if not gold:
        return 0.0 if predicted else 1.0

    if one_to_one:
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

def compute_sweep(scores_by_dialogue, gold_by_dialogue, total_gold, explicit_taus=None):
    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    tau_points = []
    for pct in TAU_PERCENTILES:
        tau_points.append((pct, np.percentile(all_scores, pct)))

    if explicit_taus:
        for i, tau in enumerate(explicit_taus):
            tau_points.append((-(i + 1), tau))

    sweep_points = []
    for pct, tau in tau_points:
        total_pred = 0  # Count predictions from ALL dialogues
        total_wf1_m2o = 0.0
        total_wf1_1to1 = 0.0
        n_with_gold = 0
        per_dialogue_m2o = {}
        per_dialogue_1to1 = {}

        for dialogue_id, scores_by_pos in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(dialogue_id, set())
            predicted = greedy_nms_predict(scores_by_pos, tau)

            # Always count predictions for BOR
            total_pred += len(predicted)

            # Only compute metrics for dialogues with gold
            if gold:
                wf1_m2o = compute_wf1(predicted, gold, one_to_one=False)
                wf1_1to1 = compute_wf1(predicted, gold, one_to_one=True)
                per_dialogue_m2o[dialogue_id] = wf1_m2o
                per_dialogue_1to1[dialogue_id] = wf1_1to1
                total_wf1_m2o += wf1_m2o
                total_wf1_1to1 += wf1_1to1
                n_with_gold += 1

        if n_with_gold > 0:
            sweep_points.append({
                "percentile": pct,
                "tau": float(tau),
                "bor": total_pred / total_gold if total_gold > 0 else 0,
                "wf1_m2o": total_wf1_m2o / n_with_gold,
                "wf1_1to1": total_wf1_1to1 / n_with_gold,
                "per_dialogue_m2o": per_dialogue_m2o,
                "per_dialogue_1to1": per_dialogue_1to1,
            })

    return sweep_points

# Compute main sweep
print("\nComputing threshold sweep...")
sweep_points = compute_sweep(dict(scores_by_dialogue), gold_by_dialogue, total_gold_all, explicit_taus=EXPLICIT_TAUS)

if sweep_points:
    best_m2o = max(sweep_points, key=lambda x: x["wf1_m2o"])
    best_1to1 = max(sweep_points, key=lambda x: x["wf1_1to1"])
    max_bor = max(sp["bor"] for sp in sweep_points)
    print(f"  Peak W-F1 (m2o): {best_m2o['wf1_m2o']:.4f} at BOR={best_m2o['bor']:.3f}")
    print(f"  Peak W-F1 (1to1): {best_1to1['wf1_1to1']:.4f} at BOR={best_1to1['bor']:.3f}")
    print(f"  Max BOR: {max_bor:.4f}")

# Bootstrap CIs
print(f"\nBootstrap ({N_BOOTSTRAP} iterations)...")
rng = np.random.RandomState(BOOTSTRAP_SEED)
dialogue_ids = list(scores_by_dialogue.keys())
n_dial = len(dialogue_ids)

bootstrap_wf1_m2o = defaultdict(list)
bootstrap_wf1_1to1 = defaultdict(list)
bootstrap_bor = defaultdict(list)

for b in range(N_BOOTSTRAP):
    if (b + 1) % 200 == 0:
        print(f"  Bootstrap: {b + 1}/{N_BOOTSTRAP}")

    sampled_ids = rng.choice(dialogue_ids, size=n_dial, replace=True)
    resampled_scores = {}
    resampled_gold = {}
    for new_id, orig_id in enumerate(sampled_ids):
        resampled_scores[new_id] = scores_by_dialogue[orig_id]
        resampled_gold[new_id] = gold_by_dialogue.get(orig_id, set())

    # Recompute total_gold for this bootstrap sample
    total_gold_sample = sum(len(g) for g in resampled_gold.values())

    sweep = compute_sweep(resampled_scores, resampled_gold, total_gold_sample, explicit_taus=EXPLICIT_TAUS)
    for sp in sweep:
        pct = sp["percentile"]
        bootstrap_wf1_m2o[pct].append(sp["wf1_m2o"])
        bootstrap_wf1_1to1[pct].append(sp["wf1_1to1"])
        bootstrap_bor[pct].append(sp["bor"])

# Compute CIs
ci_data = {}
all_pct_keys = list(TAU_PERCENTILES) + [-(i + 1) for i in range(len(EXPLICIT_TAUS))]
for pct in all_pct_keys:
    if pct in bootstrap_wf1_m2o:
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

# Remove per-dialogue data from sweep_points for smaller JSON
for sp in sweep_points:
    sp.pop("per_dialogue_m2o", None)
    sp.pop("per_dialogue_1to1", None)

# Build output
output = {
    "description": "GPT-5.2 SuperSeg Figure 4 (FIXED BOR)",
    "n_dialogues": len(dialogues),
    "n_dialogues_with_gold": sum(1 for g in gold_by_dialogue.values() if g),
    "n_boundaries": sum(len(s) for s in scores_by_dialogue.values()),
    "n_gold_boundaries": total_gold_all,
    "n_bootstrap": N_BOOTSTRAP,
    "peak_wf1_m2o": best_m2o["wf1_m2o"],
    "peak_bor_m2o": best_m2o["bor"],
    "peak_wf1_1to1": best_1to1["wf1_1to1"],
    "peak_bor_1to1": best_1to1["bor"],
    "max_bor": max_bor,
    "sweep_points": sweep_points,
    "bootstrap_ci": ci_data,
}

# Save
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {OUTPUT_JSON}")
print("\nComparison:")
print(f"  Old max BOR: 1.0773")
print(f"  New max BOR: {max_bor:.4f}")
print(f"  DistilBERT max BOR: 1.2576")
