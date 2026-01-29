#!/usr/bin/env python3
"""
Extend GPT-5.2 tau sweep to match DistilBERT BOR range.
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

def compute_exact_f1(predicted, gold):
    if not gold:
        return 0.0 if predicted else 1.0
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def load_scores_and_gold(dataset):
    cache_paths = [
        PROJECT_ROOT / '.gpt52_figure4_cache' / 'cache.json',
        PROJECT_ROOT / f'.gpt52_{dataset}_cache' / 'cache.json',
        PROJECT_ROOT / '.gpt52_superseg_cache' / 'cache.json',
    ]

    cache = {}
    for cache_path in cache_paths:
        if cache_path.exists():
            with open(cache_path) as f:
                cache.update(json.load(f))

    dataset_path = PROJECT_ROOT / 'datasets' / dataset / 'segmentation_file_test.json'
    with open(dataset_path) as f:
        data = json.load(f)

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

    for key, entry in cache.items():
        if not key.startswith(dataset):
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

    return dict(scores_by_dialogue), gold_by_dialogue

# Extend sweep and update JSON files
for dataset in ['superseg', 'dialseg711']:
    print(f'\n{"=" * 60}')
    print(f'{dataset.upper()}')
    print('=' * 60)

    scores_by_dialogue, gold_by_dialogue = load_scores_and_gold(dataset)

    all_scores = []
    for scores in scores_by_dialogue.values():
        all_scores.extend(scores.values())
    all_scores = np.array(all_scores)

    print(f'Score range: [{all_scores.min():.2f}, {all_scores.max():.2f}]')

    # Load existing JSON
    json_path = PROJECT_ROOT / 'paper' / 'experiments' / f'gpt52_{dataset}_figure4.json'
    with open(json_path) as f:
        data = json.load(f)

    existing_sweep = data.get('sweep_points', [])
    existing_bors = {sp['bor'] for sp in existing_sweep}
    print(f'Existing sweep points: {len(existing_sweep)}')
    print(f'Existing max BOR: {max(sp["bor"] for sp in existing_sweep):.4f}')

    # Extended tau values (below 1st percentile)
    extended_taus = [-12, -13, -14, -15, -16, -17, -18, -19, -20, -22, -25, all_scores.min() - 1]

    new_points = []
    for tau in extended_taus:
        total_pred = 0
        total_gold = 0
        total_wf1_m2o = 0
        total_wf1_1to1 = 0
        total_exact_f1 = 0
        n = 0

        for did, scores in scores_by_dialogue.items():
            gold = gold_by_dialogue.get(did, set())
            if not gold:
                continue
            predicted = greedy_nms(scores, tau)
            wf1_m2o, wf1_1to1 = compute_wf1(predicted, gold)
            ef1 = compute_exact_f1(predicted, gold)
            total_pred += len(predicted)
            total_gold += len(gold)
            total_wf1_m2o += wf1_m2o
            total_wf1_1to1 += wf1_1to1
            total_exact_f1 += ef1
            n += 1

        if n > 0:
            bor = total_pred / total_gold
            # Only add if BOR is new (not already in existing sweep)
            if not any(abs(bor - eb) < 0.01 for eb in existing_bors):
                new_points.append({
                    'percentile': None,  # Extended point
                    'tau': float(tau),
                    'bor': bor,
                    'wf1_m2o': total_wf1_m2o / n,
                    'wf1_1to1': total_wf1_1to1 / n,
                    'exact_f1': total_exact_f1 / n,
                })
                existing_bors.add(bor)
                print(f'  Added: tau={tau:.1f}, BOR={bor:.4f}')

    # Merge and sort
    all_points = existing_sweep + new_points
    all_points.sort(key=lambda x: x['bor'])

    data['sweep_points'] = all_points
    print(f'\nTotal sweep points after extension: {len(all_points)}')
    print(f'New max BOR: {max(sp["bor"] for sp in all_points):.4f}')

    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Updated: {json_path}')
