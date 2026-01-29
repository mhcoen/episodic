#!/usr/bin/env python3
"""Verify BOR ceiling is structural, not a bug."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ========== SUPERSEG ==========
print('=' * 60)
print('SUPERSEG')
print('=' * 60)

# Load dataset
superseg_file = PROJECT_ROOT / 'datasets' / 'superseg' / 'segmentation_file_test.json'
with open(superseg_file) as f:
    data = json.load(f)

ss_num_candidates = 0
ss_num_gold = 0
ss_dialogues = 0

dial_data = data.get('dial_data', data)
for source_key, source_dialogs in dial_data.items():
    if not isinstance(source_dialogs, list):
        continue
    for dialog in source_dialogs:
        turns = dialog.get('turns', [])
        if len(turns) < 4:
            continue

        # Count user turns
        user_turns = [t for t in turns if t.get('role') == 'user']
        num_user = len(user_turns)

        # Candidates = user_turns - 1
        ss_num_candidates += max(0, num_user - 1)

        # Gold boundaries
        prev_topic = None
        for i, t in enumerate(user_turns):
            topic = t.get('topic_id') or t.get('topic_name')
            if prev_topic is not None and topic != prev_topic:
                ss_num_gold += 1
            prev_topic = topic

        ss_dialogues += 1

ss_structural_max = ss_num_candidates / ss_num_gold if ss_num_gold > 0 else 0

# Load observed max from JSON
ss_json = json.load(open(PROJECT_ROOT / 'paper' / 'experiments' / 'gpt52_superseg_figure4.json'))
ss_observed_max = max(sp['bor'] for sp in ss_json['sweep_points'])

print(f'  Dialogues: {ss_dialogues}')
print(f'  Candidates: {ss_num_candidates}')
print(f'  Gold boundaries: {ss_num_gold}')
print(f'  Structural BOR_max: {ss_structural_max:.3f}')
print(f'  Observed BOR_max (GPT-5.2): {ss_observed_max:.3f}')
ss_match = abs(ss_structural_max - ss_observed_max) < 0.01
print(f'  Match: {"YES" if ss_match else "NO - BUG!"}')

# ========== DIALSEG711 ==========
print()
print('=' * 60)
print('DIALSEG711')
print('=' * 60)

# Load dataset (same format as SuperSeg)
dialseg_file = PROJECT_ROOT / 'datasets' / 'dialseg711' / 'segmentation_file_test.json'
with open(dialseg_file) as f:
    data = json.load(f)

ds_num_candidates = 0
ds_num_gold = 0
ds_dialogues = 0

dial_data = data.get('dial_data', data)
for source_key, source_dialogs in dial_data.items():
    if not isinstance(source_dialogs, list):
        continue
    for dialog in source_dialogs:
        turns = dialog.get('turns', [])
        if len(turns) < 4:
            continue

        # Count user turns
        user_turns = [t for t in turns if t.get('role') == 'user']
        num_user = len(user_turns)

        # Candidates = user_turns - 1
        ds_num_candidates += max(0, num_user - 1)

        # Gold boundaries
        prev_topic = None
        for i, t in enumerate(user_turns):
            topic = t.get('topic_id') or t.get('topic_name') or t.get('segment_id')
            if prev_topic is not None and topic != prev_topic:
                ds_num_gold += 1
            prev_topic = topic

        ds_dialogues += 1

ds_structural_max = ds_num_candidates / ds_num_gold if ds_num_gold > 0 else 0

# Load observed max from JSON
ds_json = json.load(open(PROJECT_ROOT / 'paper' / 'experiments' / 'gpt52_dialseg711_figure4.json'))
ds_observed_max = max(sp['bor'] for sp in ds_json['sweep_points'])

print(f'  Dialogues: {ds_dialogues}')
print(f'  Candidates: {ds_num_candidates}')
print(f'  Gold boundaries: {ds_num_gold}')
print(f'  Structural BOR_max: {ds_structural_max:.3f}')
print(f'  Observed BOR_max (GPT-5.2): {ds_observed_max:.3f}')
ds_match = abs(ds_structural_max - ds_observed_max) < 0.01
print(f'  Match: {"YES" if ds_match else "NO - BUG!"}')

# ========== SUMMARY TABLE ==========
print()
print('=' * 60)
print('SUMMARY TABLE')
print('=' * 60)
print('Dataset      | #Candidates | #Gold | Structural Max | Observed Max | Match?')
print('-------------|-------------|-------|----------------|--------------|-------')
print(f'DialSeg711   | {ds_num_candidates:11d} | {ds_num_gold:5d} | {ds_structural_max:14.3f} | {ds_observed_max:12.3f} | {"YES" if ds_match else "NO"}')
print(f'SuperSeg     | {ss_num_candidates:11d} | {ss_num_gold:5d} | {ss_structural_max:14.3f} | {ss_observed_max:12.3f} | {"YES" if ss_match else "NO"}')

# ========== CONCLUSION ==========
print()
print('=' * 60)
print('CONCLUSION')
print('=' * 60)
if ss_match and ds_match:
    print('BOR ceiling is STRUCTURAL (dataset property), not model-specific.')
    print('GPT-5.2 is correctly selecting all candidate boundaries at τ=min_score.')
else:
    print('BUG DETECTED: Observed max does not match structural ceiling!')
    print('Check selection logic or BOR computation.')
