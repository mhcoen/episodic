#!/usr/bin/env python3
"""
Benchmark topic detection strategies on SuperDialseg.

Usage:
    python benchmark_strategies.py [--full] [--strategies neural,ensemble]

Options:
    --full          Run on full test set (1322 dialogues) instead of 100
    --strategies    Comma-separated list of strategies to test
    --threshold     Neural confidence threshold (default: 0.5)
"""

import json
import time
import argparse
from typing import Dict, List, Any


def load_superdialseg(limit: int = None) -> List[Dict]:
    """Load SuperDialseg test set."""
    import os

    # Try different paths
    paths = [
        'datasets/superseg/segmentation_file_test.json',
        '../datasets/superseg/segmentation_file_test.json',
        os.path.expanduser('~/proj/episodic/datasets/superseg/segmentation_file_test.json'),
    ]

    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            break
    else:
        raise FileNotFoundError(f"Could not find SuperDialseg test file in: {paths}")

    dialogues = []
    for dataset_name, dlgs in data['dial_data'].items():
        dialogues.extend(dlgs)

    if limit:
        dialogues = dialogues[:limit]

    return dialogues


def get_topic_boundaries(turns: List[Dict]) -> set:
    """Get positions where topic changes."""
    boundaries = set()
    prev_topic = turns[0].get('topic_id')
    for i, turn in enumerate(turns[1:], 1):
        if turn.get('topic_id') != prev_topic:
            boundaries.add(i)
        prev_topic = turn.get('topic_id')
    return boundaries


def evaluate_strategy(strategy, dialogues: List[Dict], min_history: int = 5) -> Dict:
    """Evaluate a strategy on dialogues."""
    tp, fp, tn, fn = 0, 0, 0, 0
    total_time = 0
    keyword_triggers = 0

    for dlg in dialogues:
        turns = dlg['turns']
        boundaries = get_topic_boundaries(turns)

        messages = []
        for i, turn in enumerate(turns):
            role = 'assistant' if turn['role'] == 'agent' else turn['role']

            if len(messages) >= min_history:
                expected = i in boundaries

                start = time.time()
                try:
                    decision = strategy.get_decision(turn['utterance'], messages)
                    predicted = decision.topic_changed

                    # Track keyword triggers for ensemble
                    if hasattr(decision, 'signals'):
                        if decision.signals.get('decision_source') == 'keyword_explicit':
                            keyword_triggers += 1
                except Exception as e:
                    print(f"Error: {e}")
                    predicted = False
                total_time += time.time() - start

                if expected and predicted: tp += 1
                elif expected and not predicted: fn += 1
                elif not expected and predicted: fp += 1
                else: tn += 1

            messages.append({'role': role, 'content': turn['utterance']})

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_positions': total,
        'total_time_s': total_time,
        'avg_time_ms': (total_time / total * 1000) if total > 0 else 0,
        'keyword_triggers': keyword_triggers,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark topic detection strategies')
    parser.add_argument('--full', action='store_true', help='Run on full test set')
    parser.add_argument('--strategies', type=str, default='neural,ensemble,relative_embedding,keyword',
                       help='Comma-separated list of strategies')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Neural confidence threshold')
    args = parser.parse_args()

    # Load data
    limit = None if args.full else 100
    print(f"Loading SuperDialseg test set (limit={limit})...")
    dialogues = load_superdialseg(limit)
    print(f"Loaded {len(dialogues)} dialogues\n")

    # Import strategy registry
    from episodic.topics.strategy_registry import get_strategy, reset_strategy
    reset_strategy()

    strategies_to_test = [s.strip() for s in args.strategies.split(',')]
    results = {}

    for name in strategies_to_test:
        print(f"Testing {name}...")

        # Get strategy with appropriate params
        if name == 'neural':
            strategy = get_strategy(name, {'confidence_threshold': args.threshold})
        elif name == 'ensemble':
            strategy = get_strategy(name, {
                'use_keyword': True,
                'use_neural': True,
                'use_embedding': False,
                'neural_threshold': args.threshold
            })
        else:
            strategy = get_strategy(name)

        result = evaluate_strategy(strategy, dialogues)
        results[name] = result

        print(f"  F1={result['f1']:.3f}, P={result['precision']:.3f}, R={result['recall']:.3f}")
        print(f"  Time: {result['avg_time_ms']:.1f}ms/decision")
        if result['keyword_triggers'] > 0:
            print(f"  Keyword triggers: {result['keyword_triggers']}")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time(ms)':>10}")
    print("-" * 70)

    for name in sorted(results.keys(), key=lambda x: results[x]['f1'], reverse=True):
        r = results[name]
        print(f"{name:<20} {r['f1']:>8.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['avg_time_ms']:>10.1f}")

    print("-" * 70)
    print(f"Dataset: SuperDialseg test ({len(dialogues)} dialogues)")
    print(f"Neural threshold: {args.threshold}")


if __name__ == '__main__':
    main()
