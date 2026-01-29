#!/usr/bin/env python3
"""
Screen all datasets for feasible BOR support.
Compute structural BOR ceiling to identify which can probe oversegmentation.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / 'datasets'

def count_boundaries(data, format_type='dial_data'):
    """Count candidates and gold boundaries."""
    candidates = 0
    gold = 0
    dialogues = 0

    if format_type == 'dial_data':
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
                candidates += max(0, num_user - 1)

                # Gold boundaries (topic changes)
                prev_topic = None
                for t in user_turns:
                    topic = t.get('topic_id') or t.get('topic_name') or t.get('segment_id')
                    if prev_topic is not None and topic != prev_topic:
                        gold += 1
                    prev_topic = topic

                dialogues += 1

    return candidates, gold, dialogues

# Define datasets and their test files
datasets = [
    ('DialSeg711', DATASETS_DIR / 'dialseg711' / 'segmentation_file_test.json'),
    ('SuperSeg', DATASETS_DIR / 'superseg' / 'segmentation_file_test.json'),
    ('DailyDialog', DATASETS_DIR / 'dailydialog' / 'segmentation_file_test.json'),
    ('TIAGE', DATASETS_DIR / 'tiage' / 'segmentation_file_test.json'),
    ('TopicalChat', DATASETS_DIR / 'topical_chat' / 'segmentation_file_test.json'),
]

results = []

for name, path in datasets:
    if not path.exists():
        print(f"  Skipping {name}: file not found")
        continue

    with open(path) as f:
        data = json.load(f)

    C, G, D = count_boundaries(data, 'dial_data')

    if G == 0:
        print(f"  Skipping {name}: no gold boundaries found")
        continue

    structural_max = C / G
    nms_max = structural_max / 2  # Approximate with min_gap=2

    # Determine recommendation
    if name in ('DialSeg711', 'SuperSeg'):
        status = 'DONE'
    elif nms_max > 1.5:
        status = 'YES (>1.5)'
    elif nms_max > 1.2:
        status = 'MAYBE (1.2-1.5)'
    else:
        status = 'NO (<1.2)'

    results.append({
        'name': name,
        'candidates': C,
        'gold': G,
        'dialogues': D,
        'structural_max': structural_max,
        'nms_max': nms_max,
        'status': status,
    })

# Sort by nms_max descending
results.sort(key=lambda x: -x['nms_max'])

# Print table
print("=" * 90)
print("DATASET BOR CEILING SCREEN")
print("=" * 90)
print()
print(f"{'Dataset':<14} | {'#Candidates':>11} | {'#Gold':>6} | {'#Dial':>5} | {'C/G':>5} | {'NMS Max':>8} | Worth running?")
print("-" * 90)

for r in results:
    print(f"{r['name']:<14} | {r['candidates']:>11,} | {r['gold']:>6,} | {r['dialogues']:>5,} | {r['structural_max']:>5.2f} | {r['nms_max']:>8.2f} | {r['status']}")

print()
print("=" * 90)
print("RECOMMENDATIONS")
print("=" * 90)
print()
print("Priority 1 (NMS Max > 1.5 - can probe oversegmentation):")
for r in results:
    if r['status'] == 'YES (>1.5)':
        print(f"  - {r['name']}: NMS Max = {r['nms_max']:.2f}")

print()
print("Priority 2 (NMS Max 1.2-1.5 - limited overseg range):")
for r in results:
    if r['status'] == 'MAYBE (1.2-1.5)':
        print(f"  - {r['name']}: NMS Max = {r['nms_max']:.2f}")

print()
print("Already done:")
for r in results:
    if r['status'] == 'DONE':
        print(f"  - {r['name']}: NMS Max = {r['nms_max']:.2f}")

print()
print("Skip (NMS Max < 1.2 - ceiling too tight):")
for r in results:
    if r['status'] == 'NO (<1.2)':
        print(f"  - {r['name']}: NMS Max = {r['nms_max']:.2f}")
