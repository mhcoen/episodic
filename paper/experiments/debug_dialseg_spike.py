#!/usr/bin/env python3
"""
Debug the W-F1 spike in DialSeg711 extended tau points.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
MIN_GAP = 2

def greedy_nms(scores_by_pos, tau):
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

def compute_wf1(predicted, gold, k=3):
    if not gold:
        return (0.0, 0.0) if predicted else (1.0, 1.0)

    # Many-to-one matching
    tp_m2o = sum(1 for p in predicted if any(abs(p - g) <= k for g in gold))
    fn_m2o = sum(1 for g in gold if not any(abs(p - g) <= k for p in predicted))

    precision_m2o = tp_m2o / len(predicted) if predicted else 0.0
    recall_m2o = (len(gold) - fn_m2o) / len(gold) if gold else 0.0
    wf1_m2o = 2 * precision_m2o * recall_m2o / (precision_m2o + recall_m2o) if (precision_m2o + recall_m2o) > 0 else 0.0

    # One-to-one matching
    used_gold = set()
    tp_1to1 = 0
    for p in sorted(predicted):
        for g in sorted(gold):
            if g not in used_gold and abs(p - g) <= k:
                tp_1to1 += 1
                used_gold.add(g)
                break

    precision_1to1 = tp_1to1 / len(predicted) if predicted else 0.0
    recall_1to1 = tp_1to1 / len(gold) if gold else 0.0
    wf1_1to1 = 2 * precision_1to1 * recall_1to1 / (precision_1to1 + recall_1to1) if (precision_1to1 + recall_1to1) > 0 else 0.0

    return wf1_m2o, wf1_1to1

# Load caches
cache_paths = [
    PROJECT_ROOT / '.gpt52_figure4_cache' / 'cache.json',
    PROJECT_ROOT / '.gpt52_dialseg711_cache' / 'cache.json',
]

cache = {}
for cache_path in cache_paths:
    if cache_path.exists():
        with open(cache_path) as f:
            cache.update(json.load(f))
        print(f"Loaded {cache_path.name}")

# Load dataset
dataset_path = PROJECT_ROOT / 'datasets' / 'dialseg711' / 'segmentation_file_test.json'
with open(dataset_path) as f:
    data = json.load(f)

# Extract dialogues and gold boundaries
scores_by_dialogue = defaultdict(dict)
gold_by_dialogue = {}

dial_data = data.get('dial_data', data)
dialogue_id = 0
for source_key, source_dialogs in dial_data.items():
    if not isinstance(source_dialogs, list):
        continue
    for dialog in source_dialogs:
        turns = dialog.get('turns', [])
        if len(turns) < 4:
            continue

        boundaries = set()
        prev_topic = None
        user_idx = 0
        for turn in turns:
            if turn.get('role') == 'user':
                topic = turn.get('topic_id') or turn.get('topic_name')
                if prev_topic is not None and topic != prev_topic:
                    boundaries.add(user_idx)
                prev_topic = topic
                user_idx += 1

        gold_by_dialogue[dialogue_id] = boundaries
        dialogue_id += 1

# Load scores from cache
for key, entry in cache.items():
    if not key.startswith('dialseg711'):
        continue
    parts = key.split('_')
    if len(parts) >= 4:
        try:
            did = int(parts[1])
            pos = int(parts[2])
            if did in gold_by_dialogue:
                score = entry.get('score', 0)
                if not entry.get('missing_yn_in_toplogprobs') and not entry.get('invalid_first_token'):
                    scores_by_dialogue[did][pos] = score
        except ValueError:
            pass

print(f"\nDialogues: {len(gold_by_dialogue)}")
print(f"Dialogues with scores: {len(scores_by_dialogue)}")

# Get all scores
all_scores = []
for scores in scores_by_dialogue.values():
    all_scores.extend(scores.values())
all_scores = np.array(all_scores)
print(f"Score range: [{all_scores.min():.2f}, {all_scores.max():.2f}]")

# Compare tau=-11.5 (5th percentile) vs tau=-14 (extended)
print("\n" + "=" * 70)
print("COMPARING TAU POINTS")
print("=" * 70)

for tau in [-11.5, -14.0, -22.0]:
    total_pred = 0
    total_gold = 0
    total_wf1_m2o = 0
    total_wf1_1to1 = 0
    n = 0

    for did, scores in scores_by_dialogue.items():
        gold = gold_by_dialogue.get(did, set())
        if not gold:
            continue
        predicted = greedy_nms(scores, tau)
        wf1_m2o, wf1_1to1 = compute_wf1(predicted, gold)
        total_pred += len(predicted)
        total_gold += len(gold)
        total_wf1_m2o += wf1_m2o
        total_wf1_1to1 += wf1_1to1
        n += 1

    if n > 0:
        bor = total_pred / total_gold
        avg_wf1_m2o = total_wf1_m2o / n
        avg_wf1_1to1 = total_wf1_1to1 / n
        print(f"tau={tau:.1f}: BOR={bor:.4f}, W-F1(m2o)={avg_wf1_m2o:.4f}, W-F1(1to1)={avg_wf1_1to1:.4f}, n={n}")

# Check a specific dialogue to see what's happening
print("\n" + "=" * 70)
print("SAMPLE DIALOGUE ANALYSIS")
print("=" * 70)

# Find a dialogue where W-F1 differs significantly between tau=-11.5 and tau=-14
for did in list(scores_by_dialogue.keys())[:10]:
    scores = scores_by_dialogue[did]
    gold = gold_by_dialogue.get(did, set())
    if not gold:
        continue

    pred_11 = greedy_nms(scores, -11.5)
    pred_14 = greedy_nms(scores, -14.0)

    wf1_11_m2o, wf1_11_1to1 = compute_wf1(pred_11, gold)
    wf1_14_m2o, wf1_14_1to1 = compute_wf1(pred_14, gold)

    if abs(wf1_14_m2o - wf1_11_m2o) > 0.1:
        print(f"\nDialogue {did}:")
        print(f"  Gold: {sorted(gold)}")
        print(f"  Scores: {sorted(scores.items(), key=lambda x: x[0])}")
        print(f"  tau=-11.5: pred={sorted(pred_11)}, W-F1(m2o)={wf1_11_m2o:.4f}")
        print(f"  tau=-14.0: pred={sorted(pred_14)}, W-F1(m2o)={wf1_14_m2o:.4f}")
        break
