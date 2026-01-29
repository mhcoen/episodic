#!/usr/bin/env python3
"""Check if DialSeg711 has dialogues without gold boundaries."""
import json

with open('datasets/dialseg711/segmentation_file_test.json') as f:
    data = json.load(f)

dialogues_with_gold = 0
dialogues_without_gold = 0
dial_data = data.get('dial_data', data)
for source_key, source_dialogs in dial_data.items():
    if not isinstance(source_dialogs, list):
        continue
    for dialog in source_dialogs:
        turns = dialog.get('turns', [])
        if len(turns) < 4:
            continue

        has_boundary = False
        prev_topic = None
        for turn in turns:
            if turn.get('role') == 'user':
                topic = turn.get('topic_id') or turn.get('topic_name')
                if prev_topic is not None and topic != prev_topic:
                    has_boundary = True
                    break
                prev_topic = topic

        if has_boundary:
            dialogues_with_gold += 1
        else:
            dialogues_without_gold += 1

print(f'DialSeg711:')
print(f'  Dialogues with gold: {dialogues_with_gold}')
print(f'  Dialogues without gold: {dialogues_without_gold}')
print(f'  Need fix: {"YES" if dialogues_without_gold > 0 else "NO"}')
