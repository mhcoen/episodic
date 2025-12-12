#!/usr/bin/env python3
"""
Evaluate all topic detection strategies on cross-dataset benchmarks.

Tests all available strategies and compares their performance on:
- W-F1 (Windowed F1 with ±1 tolerance)
- BOR (Boundary Oversegmentation Ratio)
- Precision/Recall
- Speed
"""

import json
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from episodic.topics.evaluation import (
    Message,
    TestCase,
    BoundaryAlignment,
    ALIGNMENT_PRESETS,
    normalize_strategy_output,
    compute_operational_metrics,
    aggregate_operational_metrics,
    OperationalMetrics,
)


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset."""
    name: str
    path: str
    alignment: BoundaryAlignment
    max_dialogues: int = 100
    role_map: Dict[str, str] = None

    def __post_init__(self):
        if self.role_map is None:
            self.role_map = {'user': 'user', 'agent': 'assistant', 'assistant': 'assistant'}


# Dataset configurations
DATASETS = [
    DatasetConfig(
        name="SuperSeg",
        path="datasets/superseg/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=200,
    ),
    DatasetConfig(
        name="DialSeg711",
        path="datasets/dialseg711/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=200,
    ),
    DatasetConfig(
        name="TIAGE",
        path="datasets/tiage/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=100,
    ),
    DatasetConfig(
        name="DailyDialog",
        path="datasets/dailydialog/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=100,
    ),
    DatasetConfig(
        name="Topical_chat",
        path="datasets/topical_chat/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=100,
    ),
]


def load_segmentation_dataset(config: DatasetConfig) -> List[TestCase]:
    """Load a segmentation dataset into TestCase format."""
    path = Path(config.path)
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    test_cases = []
    dial_data = data.get('dial_data', {})

    for dataset_key, dialogues in dial_data.items():
        for i, dialogue in enumerate(dialogues[:config.max_dialogues]):
            dial_id = dialogue.get('dial_id', f'{dataset_key}_{i}')
            turns = dialogue.get('turns', [])

            if len(turns) < 4:
                continue

            messages = []
            boundaries = []

            for turn in turns:
                role = config.role_map.get(turn.get('role', 'user'), 'user')
                content = turn.get('utterance', '')
                seg_label = turn.get('segmentation_label', 0)

                messages.append(Message(
                    role=role,
                    content=content,
                    node_id=str(turn.get('turn_id', len(messages)))
                ))

                if seg_label == 1 and len(messages) < len(turns):
                    boundaries.append(len(messages))

            if messages and boundaries:
                test_cases.append(TestCase(
                    id=dial_id,
                    name=f"{config.name}_{dial_id}",
                    description=f"Dialogue from {config.name}",
                    messages=messages,
                    expected_boundaries=boundaries,
                    boundary_alignment=config.alignment,
                    source=config.name,
                ))

    return test_cases


def evaluate_strategy_on_testcase(
    strategy,
    test_case: TestCase,
    strategy_alignment: BoundaryAlignment = None
) -> Tuple[Set[int], Set[int], OperationalMetrics]:
    """Evaluate a strategy on a single test case."""
    if strategy_alignment is None:
        strategy_alignment = ALIGNMENT_PRESETS['user_starts_topic']

    gold_canonical = test_case.get_canonical_boundaries()

    message_history = []
    predicted_indices = []

    if hasattr(strategy, 'reset'):
        strategy.reset()

    for i, message in enumerate(test_case.messages):
        if message.role == 'user' and len(message_history) >= 2:
            decision = strategy.get_decision(
                query=message.content,
                messages=message_history,
                current_thread=None
            )

            if decision.topic_changed:
                predicted_indices.append(i)

        message_history.append(message.to_dict())

    pred_canonical = normalize_strategy_output(
        predicted_indices,
        test_case.messages,
        strategy_alignment
    )

    metrics = compute_operational_metrics(
        gold_canonical,
        pred_canonical,
        len(test_case.messages),
        messages=[m.to_dict() for m in test_case.messages]
    )

    return gold_canonical, pred_canonical, metrics


def create_all_strategies() -> List[Tuple[str, Any]]:
    """Create instances of all available strategies."""
    strategies = []

    # Neural strategies (baseline)
    try:
        from episodic.topics.strategies.neural_strategy import NeuralStrategy
        strategies.append(("Neural(fine)", NeuralStrategy({'granularity': 'fine'})))
        strategies.append(("Neural(coarse)", NeuralStrategy({'granularity': 'coarse'})))
    except ImportError as e:
        print(f"  Skip Neural: {e}")

    # Commitment wrapper
    try:
        from episodic.topics.strategies.neural_strategy import NeuralStrategy
        from episodic.topics.strategies.commitment_strategy import (
            CommitmentPolicyStrategy,
            CommitmentPolicy,
        )
        strategies.append(("Commit(medium)", CommitmentPolicyStrategy(
            NeuralStrategy({'granularity': 'fine'}),
            CommitmentPolicy(
                min_gap=2,
                evidence_window=2,
                min_evidence=0.7,
                evidence_decay=0.85,
            )
        )))
    except ImportError as e:
        print(f"  Skip Commitment: {e}")

    # CUSUM strategy
    try:
        from episodic.topics.strategies.cusum_strategy import CUSUMStrategy
        strategies.append(("CUSUM", CUSUMStrategy()))
    except ImportError as e:
        print(f"  Skip CUSUM: {e}")

    # Delta strategy
    try:
        from episodic.topics.strategies.delta_strategy import DeltaStrategy
        strategies.append(("Delta", DeltaStrategy()))
    except ImportError as e:
        print(f"  Skip Delta: {e}")

    # Speech Act strategy
    try:
        from episodic.topics.strategies.speech_act_strategy import SpeechActStrategy
        strategies.append(("SpeechAct", SpeechActStrategy()))
    except ImportError as e:
        print(f"  Skip SpeechAct: {e}")

    # Time Aware strategy
    try:
        from episodic.topics.strategies.time_aware_strategy import TimeAwareStrategy
        strategies.append(("TimeAware", TimeAwareStrategy()))
    except ImportError as e:
        print(f"  Skip TimeAware: {e}")

    # Summary Probe strategy
    try:
        from episodic.topics.strategies.summary_probe_strategy import SummaryProbeStrategy
        strategies.append(("SummaryProbe", SummaryProbeStrategy({'mode': 'embedding'})))
    except ImportError as e:
        print(f"  Skip SummaryProbe: {e}")

    # Dual Window strategy
    try:
        from episodic.topics.strategies.dual_window_strategy import DualWindowStrategy
        strategies.append(("DualWindow", DualWindowStrategy()))
    except ImportError as e:
        print(f"  Skip DualWindow: {e}")

    # Ensemble strategy
    try:
        from episodic.topics.strategies.ensemble_strategy import EnsembleStrategy
        strategies.append(("Ensemble", EnsembleStrategy()))
    except ImportError as e:
        print(f"  Skip Ensemble: {e}")

    return strategies


def run_evaluation(
    strategies: List[Tuple[str, Any]] = None,
    datasets: List[DatasetConfig] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run evaluation on all strategies and datasets.

    Returns:
        Dict[strategy_name][dataset_name] = {w_f1, bor, precision, recall, ...}
    """
    if strategies is None:
        print("Initializing strategies...")
        strategies = create_all_strategies()
        print(f"  Loaded {len(strategies)} strategies")

    if datasets is None:
        datasets = DATASETS

    all_results = {name: {} for name, _ in strategies}

    for config in datasets:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Dataset: {config.name}")
            print("=" * 70)

        test_cases = load_segmentation_dataset(config)
        if not test_cases:
            if verbose:
                print(f"  Dataset not found or empty: {config.path}")
            continue

        if verbose:
            print(f"  Loaded {len(test_cases)} dialogues")

        for strategy_name, strategy in strategies:
            if verbose:
                print(f"\n  {strategy_name}...", end=" ", flush=True)

            metrics_list = []
            total_gold = 0
            total_pred = 0
            errors = 0

            start_time = time.time()

            for tc in test_cases:
                try:
                    gold, pred, metrics = evaluate_strategy_on_testcase(strategy, tc)
                    metrics_list.append(metrics)
                    total_gold += len(gold)
                    total_pred += len(pred)
                except Exception as e:
                    errors += 1
                    if verbose and errors <= 3:
                        print(f"\n    Error on {tc.id}: {e}")

            elapsed = time.time() - start_time

            if metrics_list:
                agg = aggregate_operational_metrics(metrics_list)
                all_results[strategy_name][config.name] = {
                    'w_f1': agg.windowed_f1_w1,
                    'bor': agg.bor,
                    'precision': agg.precision,
                    'recall': agg.recall,
                    'num_dialogues': len(metrics_list),
                    'total_gold': total_gold,
                    'total_pred': total_pred,
                    'errors': errors,
                    'time': elapsed,
                    'speed': len(metrics_list) / elapsed if elapsed > 0 else 0,
                }

                if verbose:
                    print(f"W-F1={agg.windowed_f1_w1:.3f} BOR={agg.bor:.2f} "
                          f"P/R={agg.precision:.2f}/{agg.recall:.2f} "
                          f"({elapsed:.1f}s)")

    return all_results


def print_summary_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    strategies: List[Tuple[str, Any]],
    datasets: List[DatasetConfig]
):
    """Print summary comparison tables."""
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)

    strategy_names = [name for name, _ in strategies]

    # Determine column width based on strategy names
    col_width = max(12, max(len(n) for n in strategy_names) + 2)

    # Header
    print(f"\n{'Dataset':<14}", end="")
    for name in strategy_names:
        print(f"{name:>{col_width}}", end="")
    print()
    print("-" * (14 + col_width * len(strategy_names)))

    # W-F1 comparison
    print("\nW-F1 (Windowed F1, ±1 tolerance):")
    for config in datasets:
        print(f"  {config.name:<12}", end="")
        for name in strategy_names:
            results = all_results[name].get(config.name, {})
            w_f1 = results.get('w_f1', 0)
            if w_f1 > 0:
                print(f"{w_f1:>{col_width}.3f}", end="")
            else:
                print(f"{'--':>{col_width}}", end="")
        print()

    # BOR comparison
    print("\nBOR (Boundary Oversegmentation Ratio, ideal=1.0):")
    for config in datasets:
        print(f"  {config.name:<12}", end="")
        for name in strategy_names:
            results = all_results[name].get(config.name, {})
            bor = results.get('bor', 0)
            if bor > 0:
                print(f"{bor:>{col_width}.2f}", end="")
            else:
                print(f"{'--':>{col_width}}", end="")
        print()

    # Speed comparison
    print("\nSpeed (dialogues/second):")
    for config in datasets:
        print(f"  {config.name:<12}", end="")
        for name in strategy_names:
            results = all_results[name].get(config.name, {})
            speed = results.get('speed', 0)
            if speed > 0:
                print(f"{speed:>{col_width}.1f}", end="")
            else:
                print(f"{'--':>{col_width}}", end="")
        print()

    # Overall rankings
    print("\n" + "=" * 100)
    print("OVERALL RANKINGS (averaged across datasets)")
    print("=" * 100)

    # Calculate averages
    avg_scores = {}
    for name in strategy_names:
        w_f1_vals = [r['w_f1'] for r in all_results[name].values() if r.get('w_f1', 0) > 0]
        bor_vals = [r['bor'] for r in all_results[name].values() if r.get('bor', 0) > 0]
        speed_vals = [r['speed'] for r in all_results[name].values() if r.get('speed', 0) > 0]

        if w_f1_vals:
            avg_w_f1 = sum(w_f1_vals) / len(w_f1_vals)
            avg_bor = sum(bor_vals) / len(bor_vals) if bor_vals else 0
            avg_speed = sum(speed_vals) / len(speed_vals) if speed_vals else 0
            # Combined score: W-F1 with BOR penalty
            bor_penalty = 1.0 - abs(1.0 - avg_bor) * 0.3
            combined = avg_w_f1 * max(0.5, bor_penalty)
            avg_scores[name] = {
                'w_f1': avg_w_f1,
                'bor': avg_bor,
                'speed': avg_speed,
                'combined': combined,
            }

    # Sort by combined score
    ranked = sorted(avg_scores.items(), key=lambda x: x[1]['combined'], reverse=True)

    print(f"\n{'Rank':<6}{'Strategy':<18}{'Avg W-F1':>10}{'Avg BOR':>10}{'Speed':>10}{'Combined':>10}")
    print("-" * 64)
    for i, (name, scores) in enumerate(ranked, 1):
        print(f"{i:<6}{name:<18}{scores['w_f1']:>10.3f}{scores['bor']:>10.2f}"
              f"{scores['speed']:>10.1f}{scores['combined']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate topic detection strategies")
    parser.add_argument('--strategies', nargs='+', help="Specific strategies to test")
    parser.add_argument('--datasets', nargs='+', help="Specific datasets to test")
    parser.add_argument('--quick', action='store_true', help="Quick mode: fewer dialogues")
    parser.add_argument('--quiet', action='store_true', help="Less verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("Topic Detection Strategy Evaluation")
    print("=" * 70)

    # Create strategies
    all_strategies = create_all_strategies()

    # Filter strategies if specified
    if args.strategies:
        strategies = [(n, s) for n, s in all_strategies if n in args.strategies]
        if not strategies:
            print(f"No matching strategies. Available: {[n for n, _ in all_strategies]}")
            return
    else:
        strategies = all_strategies

    # Filter datasets if specified
    if args.datasets:
        datasets = [d for d in DATASETS if d.name in args.datasets]
        if not datasets:
            print(f"No matching datasets. Available: {[d.name for d in DATASETS]}")
            return
    else:
        datasets = DATASETS

    # Quick mode: reduce dialogues
    if args.quick:
        for d in datasets:
            d.max_dialogues = min(20, d.max_dialogues)

    print(f"\nStrategies: {[n for n, _ in strategies]}")
    print(f"Datasets: {[d.name for d in datasets]}")

    # Run evaluation
    results = run_evaluation(strategies, datasets, verbose=not args.quiet)

    # Print summary
    print_summary_table(results, strategies, datasets)


if __name__ == "__main__":
    main()
