#!/usr/bin/env python3
"""
Exactly replicate overnight script logic to find BOR discrepancy.
"""
import json
from collections import defaultdict

# Load cache
with open('.gpt52_superseg_cache/cache.json') as f:
    cache = json.load(f)

# Load dataset
with open('datasets/superseg/segmentation_file_test.json') as f:
    data = json.load(f)

# Build dialogues exactly as overnight script
dialogues = []
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
        messages = []

        for turn in turns:
            role = turn.get('role')
            content = turn.get('utterance', turn.get('text', ''))
            messages.append({'role': role, 'content': content})

            if role == 'user':
                topic = turn.get('topic_id') or turn.get('topic_name')
                if prev_topic is not None and topic != prev_topic:
                    boundaries.add(user_idx)
                prev_topic = topic
                user_idx += 1

        dialogues.append({
            'dialogue_id': dialogue_id,
            'messages': messages,
            'gold_boundaries': boundaries,
            'num_user_turns': user_idx
        })
        dialogue_id += 1

print(f'Dialogues: {len(dialogues)}')

# Build scores_by_dialogue and gold_by_dialogue EXACTLY as overnight script
scores_by_dialogue = defaultdict(dict)
gold_by_dialogue = {}

for dialogue in dialogues:
    gold_by_dialogue[dialogue['dialogue_id']] = dialogue['gold_boundaries']

for key, entry in cache.items():
    if not entry.get('missing_yn_in_toplogprobs') and not entry.get('invalid_first_token'):
        dialogue_id = entry['dialogue_id']
        position = entry['position']
        score = entry['score']
        scores_by_dialogue[dialogue_id][position] = score

n_dialogues = len(scores_by_dialogue)
n_boundaries = sum(len(s) for s in scores_by_dialogue.values())
print(f'Loaded {n_boundaries} scores across {n_dialogues} dialogues')

# Greedy NMS
MIN_GAP = 2
def greedy_nms_predict(scores_by_pos, tau):
    candidates = [(pos, score) for pos, score in scores_by_pos.items() if score > tau]
    candidates.sort(key=lambda x: -x[1])
    predicted = set()
    for pos, score in candidates:
        if not any(abs(pos - p) < MIN_GAP for p in predicted):
            predicted.add(pos)
    return predicted

# Test at tau=-30 (what the JSON reports as max)
tau = -30
total_pred = 0
total_gold = 0
n = 0

for dialogue_id, scores_by_pos in scores_by_dialogue.items():
    gold = gold_by_dialogue.get(dialogue_id, set())
    if not gold:
        continue
    predicted = greedy_nms_predict(scores_by_pos, tau)
    total_pred += len(predicted)
    total_gold += len(gold)
    n += 1

print(f'\nAt tau={tau}:')
print(f'  Dialogues processed: {n}')
print(f'  Total predictions: {total_pred}')
print(f'  Total gold: {total_gold}')
print(f'  BOR = {total_pred}/{total_gold} = {total_pred/total_gold:.4f}')

# The JSON says BOR=1.0773, which implies:
implied_pred = int(1.0773 * total_gold)
print(f'\nJSON implies {implied_pred} predictions')
print(f'Difference: {total_pred - implied_pred}')

# Check: are there dialogues with scores but no gold?
dialogues_with_scores_no_gold = 0
extra_pred_from_no_gold = 0
for dialogue_id, scores_by_pos in scores_by_dialogue.items():
    gold = gold_by_dialogue.get(dialogue_id, set())
    if not gold:
        dialogues_with_scores_no_gold += 1
        predicted = greedy_nms_predict(scores_by_pos, tau)
        extra_pred_from_no_gold += len(predicted)

print(f'\nDialogues with scores but no gold: {dialogues_with_scores_no_gold}')
print(f'Extra predictions from those: {extra_pred_from_no_gold}')

# If we add those predictions to total_pred, what BOR do we get?
total_pred_all = total_pred + extra_pred_from_no_gold
bor_all = total_pred_all / total_gold
print(f'\nIf we count predictions from ALL dialogues:')
print(f'  Total predictions: {total_pred_all}')
print(f'  BOR = {total_pred_all}/{total_gold} = {bor_all:.4f}')
print(f'  (This should match DistilBERT approach)')

# But wait - DistilBERT may have DIFFERENT total_gold
# Let me check what total_gold DistilBERT uses
import pandas as pd
df = pd.read_csv('paper/results/sweep_superseg_neural.csv')
print(f'\nDistilBERT sweep data:')
print(f'  Max BOR: {df["bor"].max():.4f}')
# At max BOR, how many predictions and gold?
max_bor_row = df.loc[df['bor'].idxmax()]
n_pred_max = max_bor_row.get('n_pred_boundaries', 'N/A')
n_gold_max = max_bor_row.get('n_gold_boundaries', 'N/A')
print(f'  At max BOR: pred={n_pred_max}, gold={n_gold_max}')
if n_gold_max != 'N/A':
    print(f'  Implied BOR from pred/gold: {n_pred_max / n_gold_max:.4f}')
