#!/usr/bin/env python3
"""
Evaluate CommitmentPolicyStrategy on cross-dataset benchmarks.

Compares NeuralStrategy with and without commitment wrapper to measure
the effect of hysteresis on BOR (Boundary Oversegmentation Ratio).
"""

import json
import sys
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
    to_canonical_boundaries,
    normalize_strategy_output,
    compute_windowed_metrics,
    compute_bor,
    compute_operational_metrics,
    aggregate_operational_metrics,
    OperationalMetrics,
)
from episodic.topics.strategies.commitment_strategy import (
    CommitmentPolicyStrategy,
    CommitmentPolicy,
    AdaptiveCommitmentStrategy,
    AdaptivePolicy,
)


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset."""
    name: str
    path: str
    alignment: BoundaryAlignment
    max_dialogues: int = 100
    role_map: Dict[str, str] = None  # Map dataset roles to user/assistant

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
        print(f"  Dataset not found: {path}")
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

            # Convert to messages
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

                # segmentation_label=1 means this is the last message of a segment
                # So boundary is AFTER this message (next message starts new topic)
                if seg_label == 1 and len(messages) < len(turns):
                    boundaries.append(len(messages))  # Boundary at next position

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
    """
    Evaluate a strategy on a single test case.

    Returns:
        (gold_boundaries, predicted_boundaries, metrics)
    """
    if strategy_alignment is None:
        strategy_alignment = ALIGNMENT_PRESETS['user_starts_topic']

    # Get canonical gold boundaries
    gold_canonical = test_case.get_canonical_boundaries()

    # Run strategy on each message
    message_history = []
    predicted_indices = []

    # Reset strategy state if it has a reset method
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

    # Convert to canonical
    pred_canonical = normalize_strategy_output(
        predicted_indices,
        test_case.messages,
        strategy_alignment
    )

    # Compute metrics
    metrics = compute_operational_metrics(
        gold_canonical,
        pred_canonical,
        len(test_case.messages),
        messages=[m.to_dict() for m in test_case.messages]
    )

    return gold_canonical, pred_canonical, metrics


def run_evaluation():
    """Run full evaluation comparing strategies."""
    print("=" * 70)
    print("CommitmentPolicyStrategy Evaluation")
    print("=" * 70)

    # Try to import neural strategy
    try:
        from episodic.topics.strategies.neural_strategy import NeuralStrategy
        neural_available = True
    except ImportError as e:
        print(f"NeuralStrategy not available: {e}")
        neural_available = False

    if not neural_available:
        print("Cannot run evaluation without NeuralStrategy")
        return

    # Create strategies
    print("\nInitializing strategies...")

    # Base neural strategy with fine granularity
    base_strategy = NeuralStrategy({'granularity': 'fine'})

    # Wrapped with different commitment policy configurations
    # Light commitment: just min_gap, low evidence threshold
    light_commit = CommitmentPolicyStrategy(
        NeuralStrategy({'granularity': 'fine'}),
        CommitmentPolicy(
            min_gap=2,
            evidence_window=1,
            min_evidence=0.5,
            evidence_decay=0.9,
        )
    )

    # Medium commitment
    medium_commit = CommitmentPolicyStrategy(
        NeuralStrategy({'granularity': 'fine'}),
        CommitmentPolicy(
            min_gap=2,
            evidence_window=2,
            min_evidence=0.7,
            evidence_decay=0.85,
        )
    )

    # Heavy commitment (original settings)
    heavy_commit = CommitmentPolicyStrategy(
        NeuralStrategy({'granularity': 'fine'}),
        CommitmentPolicy(
            min_gap=3,
            evidence_window=2,
            min_evidence=1.0,
            evidence_decay=0.8,
        )
    )

    # Also test with coarse granularity for comparison
    coarse_strategy = NeuralStrategy({'granularity': 'coarse'})

    # Adaptive v2: single-knob control with warmup calibration
    # Uses fixed min_gap, adapts only min_evidence for stability
    adaptive_v2 = AdaptiveCommitmentStrategy(
        NeuralStrategy({'granularity': 'fine'}),
        AdaptivePolicy(
            target_rate=0.12,  # ~1 boundary per 8 messages
            rate_window=40,
            adaptation_rate=0.15,  # Conservative for stability
            tolerance=0.25,
            fixed_min_gap=2,
            warmup_messages=10,
            warmup_calibrate=True,
        )
    )

    strategies = [
        ("Neural(fine)", base_strategy),
        ("Commit(medium)", medium_commit),
        ("Adaptive(v2)", adaptive_v2),
        ("Neural(coarse)", coarse_strategy),
    ]

    # Results storage
    all_results = {name: {} for name, _ in strategies}

    # Evaluate on each dataset
    for config in DATASETS:
        print(f"\n{'='*70}")
        print(f"Dataset: {config.name}")
        print("=" * 70)

        test_cases = load_segmentation_dataset(config)
        if not test_cases:
            print(f"  No test cases loaded")
            continue

        print(f"  Loaded {len(test_cases)} dialogues")

        for strategy_name, strategy in strategies:
            print(f"\n  Strategy: {strategy_name}")

            metrics_list = []
            total_gold = 0
            total_pred = 0

            start_time = time.time()

            for tc in test_cases:
                try:
                    gold, pred, metrics = evaluate_strategy_on_testcase(strategy, tc)
                    metrics_list.append(metrics)
                    total_gold += len(gold)
                    total_pred += len(pred)
                except Exception as e:
                    print(f"    Error on {tc.id}: {e}")
                    continue

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
                }

                print(f"    W-F1: {agg.windowed_f1_w1:.3f}")
                print(f"    BOR:  {agg.bor:.2f}")
                print(f"    P/R:  {agg.precision:.2f} / {agg.recall:.2f}")
                print(f"    Boundaries: {total_pred} pred / {total_gold} gold")
                print(f"    Time: {elapsed:.1f}s ({len(metrics_list)/elapsed:.1f} dial/s)")

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    # Header
    print(f"\n{'Dataset':<15}", end="")
    for name, _ in strategies:
        print(f"  {name:<25}", end="")
    print()
    print("-" * 95)

    # W-F1 comparison
    print("\nW-F1 (Windowed F1, ±1 tolerance):")
    for config in DATASETS:
        print(f"  {config.name:<13}", end="")
        for strategy_name, _ in strategies:
            results = all_results[strategy_name].get(config.name, {})
            w_f1 = results.get('w_f1', 0)
            print(f"  {w_f1:>6.3f}                   ", end="")
        print()

    # BOR comparison
    print("\nBOR (Boundary Oversegmentation Ratio, ideal=1.0):")
    for config in DATASETS:
        print(f"  {config.name:<13}", end="")
        for strategy_name, _ in strategies:
            results = all_results[strategy_name].get(config.name, {})
            bor = results.get('bor', 0)
            print(f"  {bor:>6.2f}                   ", end="")
        print()

    # Calculate and show best configuration
    print("\nBest Configuration Per Dataset (highest W-F1 with BOR closest to 1.0):")
    for config in DATASETS:
        best_name = None
        best_score = -1

        for strategy_name, _ in strategies:
            results = all_results[strategy_name].get(config.name, {})
            w_f1 = results.get('w_f1', 0)
            bor = results.get('bor', 0)

            if w_f1 > 0:
                # Score: W-F1 weighted by how close BOR is to 1.0
                bor_penalty = 1.0 - abs(1.0 - bor) * 0.3  # Small penalty for BOR deviation
                score = w_f1 * max(0.5, bor_penalty)

                if score > best_score:
                    best_score = score
                    best_name = strategy_name

        if best_name:
            results = all_results[best_name].get(config.name, {})
            print(f"  {config.name:<13}: {best_name} (W-F1={results.get('w_f1', 0):.3f}, BOR={results.get('bor', 0):.2f})")


if __name__ == "__main__":
    run_evaluation()
