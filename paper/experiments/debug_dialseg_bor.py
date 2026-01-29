#!/usr/bin/env python3
"""Debug DialSeg711 BOR computation."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path('.')
MIN_GAP = 2

def greedy_nms(scores_by_pos, tau):
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

# Load cache
with open('.gpt52_figure4_cache/cache.json') as f:
    cache = json.load(f)

# Load dataset
with open('datasets/dialseg711/segmentation_file_test.json') as f:
    data = json.load(f)

# Build gold_by_dialogue
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

# Build scores_by_dialogue
scores_by_dialogue = defaultdict(dict)
for key, entry in cache.items():
    if key.startswith('dialseg711_'):
        dialogue_id = entry['dialogue_id']
        position = entry['position']
        score = entry['score']
        if not entry.get('missing_yn_in_toplogprobs') and not entry.get('invalid_first_token'):
            scores_by_dialogue[dialogue_id][position] = score

total_gold = sum(len(g) for g in gold_by_dialogue.values())
print(f'Total gold: {total_gold}')
print(f'Dialogues with scores: {len(scores_by_dialogue)}')
print(f'Dialogues with gold: {len(gold_by_dialogue)}')

# Check dialogues with gold but no scores
dids_with_gold_no_scores = set(gold_by_dialogue.keys()) - set(scores_by_dialogue.keys())
gold_in_missing = sum(len(gold_by_dialogue[did]) for did in dids_with_gold_no_scores)
print(f'Dialogues with gold but no scores: {len(dids_with_gold_no_scores)}')
print(f'Gold boundaries in those: {gold_in_missing}')

# Compute at 5th percentile
all_scores = []
for scores in scores_by_dialogue.values():
    all_scores.extend(scores.values())
all_scores = np.array(all_scores)
tau = np.percentile(all_scores, 5)
print(f'\nTau at 5th percentile: {tau:.2f}')

# Count predictions (only from dialogues WITH scores)
total_pred = 0
for did, scores in scores_by_dialogue.items():
    predicted = greedy_nms(scores, tau)
    total_pred += len(predicted)

print(f'Total predictions at tau={tau:.2f}: {total_pred}')
print(f'BOR (using total_gold={total_gold}): {total_pred/total_gold:.4f}')

# What if we only count gold from dialogues WITH scores?
gold_in_scored = sum(len(gold_by_dialogue.get(did, set())) for did in scores_by_dialogue.keys())
print(f'\nGold in scored dialogues: {gold_in_scored}')
print(f'BOR (using gold_in_scored): {total_pred/gold_in_scored:.4f}')
