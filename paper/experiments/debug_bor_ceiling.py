#!/usr/bin/env python3
"""
Debug BOR ceiling contradiction between GPT-5.2 and DistilBERT on SuperSeg
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
MIN_GAP = 2

def greedy_nms(scores_by_pos, tau):
    """Greedy NMS with MIN_GAP suppression."""
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

# Load GPT-5.2 scores for SuperSeg
cache_path = PROJECT_ROOT / '.gpt52_superseg_cache' / 'cache.json'
with open(cache_path) as f:
    cache = json.load(f)

# Load SuperSeg dataset
dataset_path = PROJECT_ROOT / 'datasets' / 'superseg' / 'segmentation_file_test.json'
with open(dataset_path) as f:
    data = json.load(f)

# Extract gold boundaries and scoreable positions
scores_by_dialogue = defaultdict(dict)
gold_by_dialogue = {}
total_gold = 0
total_positions = 0

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
        total_gold += len(boundaries)
        dialogue_id += 1

# Load GPT-5.2 scores
for key, entry in cache.items():
    if not key.startswith('superseg'):
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

# Count total positions with scores
for did, scores in scores_by_dialogue.items():
    total_positions += len(scores)

print("=" * 70)
print("GPT-5.2 SuperSeg Analysis")
print("=" * 70)
print(f"Dialogues with gold: {len(gold_by_dialogue)}")
print(f"Dialogues with scores: {len(scores_by_dialogue)}")
print(f"Total gold boundaries: {total_gold}")
print(f"Total scored positions: {total_positions}")

# Get all scores
all_scores = []
for scores in scores_by_dialogue.values():
    all_scores.extend(scores.values())
all_scores = np.array(all_scores)
print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")

# At tau = -25 (very low threshold)
tau = -25
total_before_nms = 0
total_after_nms = 0
for did, scores in scores_by_dialogue.items():
    candidates_before = [(pos, score) for pos, score in scores.items() if score > tau]
    total_before_nms += len(candidates_before)
    predicted = greedy_nms(scores, tau)
    total_after_nms += len(predicted)

print(f"\nAt tau = {tau}:")
print(f"  Candidates BEFORE NMS: {total_before_nms}")
print(f"  Candidates AFTER NMS: {total_after_nms}")
print(f"  BOR = {total_after_nms} / {total_gold} = {total_after_nms / total_gold:.4f}")

# At tau = min - 1 (ALL positions pass threshold)
tau_min = all_scores.min() - 1
total_before_nms = 0
total_after_nms = 0
for did, scores in scores_by_dialogue.items():
    candidates_before = [(pos, score) for pos, score in scores.items() if score > tau_min]
    total_before_nms += len(candidates_before)
    predicted = greedy_nms(scores, tau_min)
    total_after_nms += len(predicted)

print(f"\nAt tau = {tau_min:.4f} (min_score - 1, all pass):")
print(f"  Candidates BEFORE NMS: {total_before_nms}")
print(f"  Candidates AFTER NMS: {total_after_nms}")
print(f"  BOR = {total_after_nms} / {total_gold} = {total_after_nms / total_gold:.4f}")

# Now compare with DistilBERT
print("\n" + "=" * 70)
print("DistilBERT SuperSeg Analysis")
print("=" * 70)

distilbert_path = PROJECT_ROOT / 'paper' / 'results' / 'sweep_superseg_neural.csv'
df = pd.read_csv(distilbert_path)
print(f"Sweep points: {len(df)}")
print(f"BOR range: [{df['bor'].min():.4f}, {df['bor'].max():.4f}]")

# Find the row with max BOR
max_bor_row = df.loc[df['bor'].idxmax()]
print(f"\nAt max BOR:")
print(f"  tau: {max_bor_row['tau']:.6f}")
print(f"  BOR: {max_bor_row['bor']:.4f}")
print(f"  MIN_GAP (g): {max_bor_row['g']}")

# Check what's happening - is there per-dialogue data?
per_dial_path = PROJECT_ROOT / 'paper' / 'results' / 'sweep_superseg_neural_per_dialogue.json'
if per_dial_path.exists():
    with open(per_dial_path) as f:
        per_dial_data = json.load(f)
    # Count predictions at lowest tau
    first_step = per_dial_data.get('0', {})
    total_pred = sum(d.get('n_pred', 0) for d in first_step.values() if isinstance(d, dict))
    total_gold_check = sum(d.get('n_gold', 0) for d in first_step.values() if isinstance(d, dict))
    print(f"\nDistilBERT per-dialogue (step 0 = lowest tau):")
    print(f"  Total predictions: {total_pred}")
    print(f"  Total gold: {total_gold_check}")
    if total_gold_check > 0:
        print(f"  BOR: {total_pred / total_gold_check:.4f}")
else:
    print(f"\nNo per-dialogue data at {per_dial_path}")

# KEY QUESTION: How many scored positions does DistilBERT have?
print("\n" + "=" * 70)
print("KEY COMPARISON")
print("=" * 70)
print(f"GPT-5.2 scored positions per dialogue: {total_positions / len(scores_by_dialogue):.1f}")
print(f"GPT-5.2 max predictions (all pass tau, after NMS): {total_after_nms}")
print(f"GPT-5.2 max BOR (NMS ceiling): {total_after_nms / total_gold:.4f}")
print(f"DistilBERT max BOR: {df['bor'].max():.4f}")

# The difference must be in the number of scored positions!
print("\n*** HYPOTHESIS: Different coverage of scoreable positions ***")
print("   GPT-5.2 only scores user turns")
print("   DistilBERT may score ALL turns (user + assistant)")

# Check what the JSON file reports vs what we computed
print("\n" + "=" * 70)
print("JSON FILE DISCREPANCY CHECK")
print("=" * 70)

json_path = PROJECT_ROOT / 'paper' / 'experiments' / 'gpt52_superseg_figure4.json'
with open(json_path) as f:
    json_data = json.load(f)

print(f"JSON n_boundaries: {json_data['n_boundaries']}")
print(f"Our total_positions: {total_positions}")
print(f"JSON peak_bor_m2o: {json_data['peak_bor_m2o']:.4f}")
print(f"Our max BOR: {total_after_nms / total_gold:.4f}")

# The JSON is reporting a DIFFERENT gold count
# BOR = pred / gold
# If JSON BOR = 1.077 and our BOR = 1.208, then:
# JSON_gold = pred / 1.077 = 3530 / 1.077 = 3277
# Our_gold = pred / 1.208 = 3530 / 1.208 = 2923
print(f"\nJSON implied gold count: {total_after_nms / json_data['peak_bor_m2o']:.0f}")
print(f"Our gold count: {total_gold}")

# Check n_gold_boundaries in JSON if present
if 'n_gold_boundaries' in json_data:
    print(f"JSON n_gold_boundaries: {json_data['n_gold_boundaries']}")

# PROBLEM: "if not gold: continue" skips dialogues without boundaries
# Let me recompute using the SAME logic as the overnight script
print("\n" + "=" * 70)
print("RECOMPUTE WITH SAME LOGIC AS OVERNIGHT SCRIPT")
print("=" * 70)

# Recount with the exact same skip logic
total_gold_consistent = 0
skipped_dialogues = 0
dialogues_with_gold = 0
for did, gold in gold_by_dialogue.items():
    if not gold:
        skipped_dialogues += 1
        continue
    total_gold_consistent += len(gold)
    dialogues_with_gold += 1

print(f"Dialogues with gold: {dialogues_with_gold}")
print(f"Dialogues skipped (no gold): {skipped_dialogues}")
print(f"Total gold (consistent): {total_gold_consistent}")

# Now use this for BOR
bor_consistent = total_after_nms / total_gold_consistent
print(f"BOR (consistent): {bor_consistent:.4f}")

# Check the max sweep point in JSON
max_sweep = max(json_data['sweep_points'], key=lambda x: x['bor'])
print(f"\nJSON max sweep point:")
print(f"  tau: {max_sweep['tau']}")
print(f"  BOR: {max_sweep['bor']:.4f}")
print(f"  predictions at this BOR: {max_sweep['bor'] * total_gold_consistent:.0f}")

# Verify by loading the overnight script's way
print("\n" + "=" * 70)
print("VERIFY CACHE PARSING")
print("=" * 70)

# Load cache entries matching superseg
superseg_entries = {k: v for k, v in cache.items() if k.startswith('superseg_')}
print(f"Total superseg cache entries: {len(superseg_entries)}")

# Parse entries
overnight_scores = defaultdict(dict)
for key, entry in superseg_entries.items():
    parts = key.split('_')
    if len(parts) >= 4:
        try:
            did = int(parts[1])
            pos = int(parts[2])
            score = entry.get('score', 0)
            # Apply same filters as overnight script
            if not entry.get('missing_yn_in_toplogprobs') and not entry.get('invalid_first_token'):
                overnight_scores[did][pos] = score
        except ValueError:
            pass

overnight_total_positions = sum(len(s) for s in overnight_scores.values())
print(f"Overnight-parsed positions: {overnight_total_positions}")
print(f"Debug-parsed positions: {total_positions}")

# Check if scores differ
diff_dialogues = 0
for did in set(overnight_scores.keys()) | set(scores_by_dialogue.keys()):
    ov = overnight_scores.get(did, {})
    db = scores_by_dialogue.get(did, {})
    if set(ov.keys()) != set(db.keys()):
        diff_dialogues += 1

print(f"Dialogues with different positions: {diff_dialogues}")

# Recompute using overnight parsing
tau_test = -30
total_after_nms_overnight = 0
for did, scores in overnight_scores.items():
    predicted = greedy_nms(scores, tau_test)
    total_after_nms_overnight += len(predicted)

print(f"\nUsing overnight parsing at tau={tau_test}:")
print(f"  Total predictions: {total_after_nms_overnight}")
print(f"  BOR: {total_after_nms_overnight / total_gold_consistent:.4f}")
