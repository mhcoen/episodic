#!/usr/bin/env python3
"""
Regenerate DialSeg711 GPT-5.2 sweep with reconciled cache.
Uses the same logic as the original gpt52_dialseg711_figure4.py but with extended tau.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = PROJECT_ROOT / ".gpt52_figure4_cache" / "cache.json"
DATASET_FILE = PROJECT_ROOT / "datasets" / "dialseg711" / "segmentation_file_test.json"
OUTPUT_JSON = PROJECT_ROOT / "paper" / "experiments" / "gpt52_dialseg711_figure4.json"

MIN_GAP = 2
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
TAU_PERCENTILES = list(range(5, 100, 5))  # 5, 10, 15, ..., 95

print("=" * 60)
print("Regenerating DialSeg711 GPT-5.2 Sweep (Reconciled Cache)")
print("=" * 60)

# Load reconciled cache
with open(CACHE_FILE) as f:
    cache = json.load(f)

scores_by_dialogue = defaultdict(dict)
for key, entry in cache.items():
    if key.startswith("dialseg711_"):
        dialogue_id = entry["dialogue_id"]
        position = entry["position"]
        score = entry["score"]
        if not entry.get("missing_yn_in_toplogprobs") and not entry.get("invalid_first_token"):
            scores_by_dialogue[dialogue_id][position] = score

print(f"Loaded {sum(len(s) for s in scores_by_dialogue.values())} scores across {len(scores_by_dialogue)} dialogues")

# Load gold boundaries
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

total_gold = sum(len(g) for g in gold_by_dialogue.values())
print(f"Gold boundaries: {total_gold} across {len(gold_by_dialogue)} dialogues")

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
        matched_gold = set()
        tp = 0
        for p in predicted:
            for g in gold:
                if abs(p - g) <= k and g not in matched_gold:
                    matched_gold.add(g)
                    tp += 1
                    break

    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold) if one_to_one else len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# Compute sweep
all_scores = []
for scores in scores_by_dialogue.values():
    all_scores.extend(scores.values())
all_scores = np.array(all_scores)
print(f"Score range: [{all_scores.min():.2f}, {all_scores.max():.2f}]")

print("\nComputing threshold sweep...")
sweep_points = []
per_dialogue_data = {}

for pct in TAU_PERCENTILES:
    tau = np.percentile(all_scores, pct)

    total_pred = 0
    total_wf1_m2o = 0.0
    total_wf1_1to1 = 0.0
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
        total_pred += len(predicted)
        total_wf1_m2o += wf1_m2o
        total_wf1_1to1 += wf1_1to1
        n += 1

    if n > 0:
        bor = total_pred / total_gold
        sweep_points.append({
            "percentile": pct,
            "tau": float(tau),
            "bor": bor,
            "wf1_m2o": total_wf1_m2o / n,
            "wf1_1to1": total_wf1_1to1 / n,
        })
        per_dialogue_data[pct] = {
            "m2o": per_dialogue_m2o,
            "1to1": per_dialogue_1to1,
        }

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
        print(f"  {b + 1}/{N_BOOTSTRAP}")

    sampled_ids = rng.choice(dialogue_ids, size=n_dial, replace=True)

    for pct in TAU_PERCENTILES:
        if pct not in per_dialogue_data:
            continue

        m2o_vals = [per_dialogue_data[pct]["m2o"].get(did, 0) for did in sampled_ids]
        o2o_vals = [per_dialogue_data[pct]["1to1"].get(did, 0) for did in sampled_ids]

        bootstrap_wf1_m2o[pct].append(np.mean(m2o_vals))
        bootstrap_wf1_1to1[pct].append(np.mean(o2o_vals))

# Compute CIs
ci_data = {}
for pct in TAU_PERCENTILES:
    if pct in bootstrap_wf1_m2o:
        m2o = np.array(bootstrap_wf1_m2o[pct])
        o2o = np.array(bootstrap_wf1_1to1[pct])

        ci_data[str(pct)] = {
            "wf1_m2o_lo": float(np.percentile(m2o, 2.5)),
            "wf1_m2o_hi": float(np.percentile(m2o, 97.5)),
            "wf1_1to1_lo": float(np.percentile(o2o, 2.5)),
            "wf1_1to1_hi": float(np.percentile(o2o, 97.5)),
        }

# Build output
output = {
    "description": "GPT-5.2 DialSeg711 Figure 4 (reconciled cache)",
    "n_dialogues": len(scores_by_dialogue),
    "n_boundaries": sum(len(s) for s in scores_by_dialogue.values()),
    "n_gold_boundaries": total_gold,
    "n_bootstrap": N_BOOTSTRAP,
    "peak_wf1_m2o": best_m2o["wf1_m2o"],
    "peak_bor_m2o": best_m2o["bor"],
    "peak_wf1_1to1": best_1to1["wf1_1to1"],
    "peak_bor_1to1": best_1to1["bor"],
    "sweep_points": sweep_points,
    "bootstrap_ci": ci_data,
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {OUTPUT_JSON}")
