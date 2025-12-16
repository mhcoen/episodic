#!/usr/bin/env python3
"""
Online evaluation of topic detection strategies.

Replays dialogues turn-by-turn, traces state machine, computes delay/churn metrics.

Usage:
    # Single dataset evaluation
    python scripts/evaluate_online.py --dataset superseg --output traces.jsonl

    # Full ablation with threshold sweep
    python scripts/evaluate_online.py --ablation --datasets superseg dialseg tiage \
        --drift-thresholds 0.90,0.93,0.95,0.97,0.99 \
        --delay-tolerance 2

    # Run synthetic tests
    python scripts/evaluate_online.py --synthetic
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from episodic.topics.evaluation import (
    Message,
    TestCase,
    BoundaryAlignment,
    ALIGNMENT_PRESETS,
)
from episodic.topics.online_evaluation import (
    OnlineReplayHarness,
    DialogueTrace,
    write_traces_jsonl,
)
from episodic.topics.online_metrics import (
    OnlineMetrics,
    compute_online_metrics,
    aggregate_online_metrics,
)


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

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


DATASETS = {
    "superseg": DatasetConfig(
        name="SuperSeg",
        path="datasets/superseg/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=200,
    ),
    "dialseg": DatasetConfig(
        name="DialSeg711",
        path="datasets/dialseg711/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=200,
    ),
    "tiage": DatasetConfig(
        name="TIAGE",
        path="datasets/tiage/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=100,
    ),
    "dailydialog": DatasetConfig(
        name="DailyDialog",
        path="datasets/dailydialog/segmentation_file_test.json",
        alignment=ALIGNMENT_PRESETS['segment_start'],
        max_dialogues=100,
    ),
}


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


# ============================================================================
# SYNTHETIC TESTS
# ============================================================================

def create_synthetic_tests() -> List[TestCase]:
    """
    Create synthetic test dialogues targeting specific failure modes.

    Includes:
    - Sharp immediate shift (should detect at boundary)
    - Gradual transition (2-5 turns to shift)
    - Short interruption (should NOT trigger)
    - Paraphrase shift (should NOT trigger)
    - Hard negatives: new entity same topic, one-sentence digression
    """
    tests = []

    # 1. Sharp immediate shift (AI -> coffee)
    tests.append(TestCase(
        id="sharp_shift_1",
        name="Sharp immediate shift",
        description="Topic A then sudden switch to Topic B",
        messages=[
            Message(role="user", content="Tell me about artificial intelligence", node_id="0"),
            Message(role="assistant", content="AI is the simulation of human intelligence by machines.", node_id="1"),
            Message(role="user", content="What are some applications of machine learning?", node_id="2"),
            Message(role="assistant", content="Machine learning is used in image recognition, NLP, and recommendation systems.", node_id="3"),
            Message(role="user", content="How does deep learning work?", node_id="4"),
            Message(role="assistant", content="Deep learning uses neural networks with many layers to learn from data.", node_id="5"),
            # Sharp shift here
            Message(role="user", content="What's the ideal temperature for brewing coffee?", node_id="6"),
            Message(role="assistant", content="The ideal brewing temperature is between 195-205°F.", node_id="7"),
        ],
        expected_boundaries=[6],  # Boundary at coffee question
        boundary_alignment=ALIGNMENT_PRESETS['segment_start'],
        tags=["sharp", "positive"],
        source="synthetic",
    ))

    # 2. Gradual transition
    tests.append(TestCase(
        id="gradual_shift_1",
        name="Gradual transition",
        description="Topic gradually shifts over 2-3 turns",
        messages=[
            Message(role="user", content="I'm learning Python programming", node_id="0"),
            Message(role="assistant", content="Python is great for beginners!", node_id="1"),
            Message(role="user", content="What libraries should I learn?", node_id="2"),
            Message(role="assistant", content="NumPy, Pandas, and Matplotlib are essential.", node_id="3"),
            Message(role="user", content="NumPy is good for data analysis right?", node_id="4"),
            Message(role="assistant", content="Yes, it's fundamental for numerical computing.", node_id="5"),
            # Gradual shift: data analysis -> statistics
            Message(role="user", content="Speaking of data, what's a good way to understand statistics?", node_id="6"),
            Message(role="assistant", content="Start with descriptive statistics and probability.", node_id="7"),
            Message(role="user", content="How do I calculate standard deviation?", node_id="8"),
            Message(role="assistant", content="It measures the spread of data from the mean.", node_id="9"),
        ],
        expected_boundaries=[6],  # Boundary at statistics shift
        boundary_alignment=ALIGNMENT_PRESETS['segment_start'],
        tags=["gradual", "positive"],
        source="synthetic",
    ))

    # 3. Short interruption (should NOT trigger)
    tests.append(TestCase(
        id="interruption_1",
        name="Short interruption - no boundary",
        description="One off-topic message then back to original topic",
        messages=[
            Message(role="user", content="Let's discuss database design", node_id="0"),
            Message(role="assistant", content="Sure, what aspect interests you?", node_id="1"),
            Message(role="user", content="What's the difference between SQL and NoSQL?", node_id="2"),
            Message(role="assistant", content="SQL uses structured schemas, NoSQL is more flexible.", node_id="3"),
            # Brief interruption
            Message(role="user", content="Oh, nice weather today isn't it?", node_id="4"),
            Message(role="assistant", content="Yes, it's pleasant outside.", node_id="5"),
            # Back to databases
            Message(role="user", content="Anyway, when should I use NoSQL?", node_id="6"),
            Message(role="assistant", content="NoSQL is good for unstructured data and scalability.", node_id="7"),
        ],
        expected_boundaries=[],  # NO boundary - just a brief interruption
        boundary_alignment=ALIGNMENT_PRESETS['segment_start'],
        tags=["interruption", "negative"],
        source="synthetic",
    ))

    # 4. Paraphrase shift (should NOT trigger)
    tests.append(TestCase(
        id="paraphrase_1",
        name="Paraphrase - no boundary",
        description="Same topic with different wording",
        messages=[
            Message(role="user", content="How do I make a website?", node_id="0"),
            Message(role="assistant", content="You can use HTML, CSS, and JavaScript.", node_id="1"),
            Message(role="user", content="What are the steps to build a web page?", node_id="2"),
            Message(role="assistant", content="Start with structure (HTML), then styling (CSS).", node_id="3"),
            Message(role="user", content="So creating websites involves those three technologies?", node_id="4"),
            Message(role="assistant", content="Yes, they form the foundation of web development.", node_id="5"),
        ],
        expected_boundaries=[],  # NO boundary - same topic
        boundary_alignment=ALIGNMENT_PRESETS['segment_start'],
        tags=["paraphrase", "negative"],
        source="synthetic",
    ))

    # 5. HARD NEGATIVE: New entity same topic
    tests.append(TestCase(
        id="new_entity_same_topic",
        name="Hard negative: new entity same topic",
        description="Topic A with suddenly introduced named entity (high drift, same topic)",
        messages=[
            Message(role="user", content="I want to learn Python decorators", node_id="0"),
            Message(role="assistant", content="Decorators wrap functions to add functionality.", node_id="1"),
            Message(role="user", content="Can you give me an example?", node_id="2"),
            Message(role="assistant", content="@property is a common built-in decorator.", node_id="3"),
            # New entity introduced but SAME TOPIC
            Message(role="user", content="My colleague Bob mentioned Python decorators are used for caching too", node_id="4"),
            Message(role="assistant", content="Yes, functools.lru_cache is a caching decorator.", node_id="5"),
            Message(role="user", content="How does Bob's approach with memoization work?", node_id="6"),
            Message(role="assistant", content="Memoization stores function results for repeated calls.", node_id="7"),
        ],
        expected_boundaries=[],  # NO boundary - still Python decorators
        boundary_alignment=ALIGNMENT_PRESETS['segment_start'],
        tags=["hard_negative", "new_entity"],
        source="synthetic",
    ))

    # 6. HARD NEGATIVE: One-sentence digression
    tests.append(TestCase(
        id="one_sentence_digression",
        name="Hard negative: one-sentence digression",
        description="Semantically distant one-liner that immediately returns to topic A",
        messages=[
            Message(role="user", content="Let's talk about database optimization", node_id="0"),
            Message(role="assistant", content="Indexing is key for query performance.", node_id="1"),
            Message(role="user", content="What types of indexes should I use?", node_id="2"),
            Message(role="assistant", content="B-tree for range queries, hash for equality.", node_id="3"),
            Message(role="user", content="How do I identify slow queries?", node_id="4"),
            Message(role="assistant", content="Use EXPLAIN ANALYZE to see query plans.", node_id="5"),
            # Semantically distant one-liner
            Message(role="user", content="My cat just knocked over my coffee", node_id="6"),
            Message(role="assistant", content="Oh no! Hope it didn't spill on anything important.", node_id="7"),
            # Immediately back to databases
            Message(role="user", content="Anyway, should I add indexes to all columns?", node_id="8"),
            Message(role="assistant", content="No, indexes add overhead for writes.", node_id="9"),
        ],
        expected_boundaries=[],  # NO boundary - just a digression
        boundary_alignment=ALIGNMENT_PRESETS['segment_start'],
        tags=["hard_negative", "digression"],
        source="synthetic",
    ))

    return tests


# ============================================================================
# STRATEGY CREATION
# ============================================================================

def create_strategy(
    strategy_type: str,
    drift_threshold: Optional[float] = None,
    neural_threshold: float = 0.5,
    min_gap: int = 4,
):
    """
    Create a strategy variant for ablation.

    Strategy types:
    - neural_only: NeuralStrategy with threshold+min_gap post-processing
    - commitment_neural_only: CommitmentPolicy without drift trigger
    - commitment_hybrid: CommitmentPolicy with drift trigger at given threshold
    """
    from episodic.topics.strategies.neural_strategy import NeuralStrategy
    from episodic.topics.strategies.commitment_strategy import (
        CommitmentPolicyStrategy,
        CommitmentPolicy,
    )

    neural_base = NeuralStrategy({'granularity': 'fine'})

    if strategy_type == "neural_only":
        # Neural with same threshold + min_gap as commitment (fair baseline)
        policy = CommitmentPolicy(
            min_gap=min_gap,
            suspect_threshold=neural_threshold,
            abort_threshold=0.0,  # Never abort
            abort_streak=999,     # Never abort
            min_evidence=0.0,     # Commit immediately (no evidence accumulation)
            evidence_decay=1.0,   # No decay
            drift_suspect_threshold=None,  # No drift trigger
        )
        return CommitmentPolicyStrategy(neural_base, policy)

    elif strategy_type == "commitment_neural_only":
        # Full commitment state machine, no drift trigger
        policy = CommitmentPolicy(
            min_gap=min_gap,
            suspect_threshold=neural_threshold,
            abort_threshold=0.3,
            abort_streak=3,
            min_evidence=1.2,
            evidence_decay=0.7,
            drift_suspect_threshold=None,  # No drift trigger
        )
        return CommitmentPolicyStrategy(neural_base, policy)

    elif strategy_type == "commitment_hybrid":
        # Full commitment + drift trigger at specified threshold
        policy = CommitmentPolicy(
            min_gap=min_gap,
            suspect_threshold=neural_threshold,
            abort_threshold=0.3,
            abort_streak=3,
            min_evidence=1.2,
            evidence_decay=0.7,
            drift_suspect_threshold=drift_threshold,
        )
        return CommitmentPolicyStrategy(neural_base, policy)

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_strategy_on_dataset(
    strategy,
    test_cases: List[TestCase],
    compute_drift: bool = True,
    delay_tolerance: int = 2,
) -> Tuple[List[DialogueTrace], OnlineMetrics]:
    """
    Evaluate a strategy on a dataset using online replay.

    Returns per-dialogue traces and aggregate metrics.
    """
    harness = OnlineReplayHarness(strategy, compute_drift=compute_drift)
    traces = []

    for test_case in test_cases:
        try:
            trace = harness.replay_dialogue(test_case)
            traces.append(trace)
        except Exception as e:
            print(f"  Error on {test_case.id}: {e}")
            continue

    metrics = compute_online_metrics(traces, tolerance=delay_tolerance)
    return traces, metrics


def run_ablation(
    datasets: List[str],
    drift_thresholds: List[float],
    delay_tolerance: int = 2,
    output_dir: Path = None,
) -> Dict[str, Dict[str, OnlineMetrics]]:
    """
    Run full ablation study across datasets and drift thresholds.

    Returns metrics indexed by (strategy, dataset).
    """
    results: Dict[str, Dict[str, OnlineMetrics]] = {}
    strategy_types = ["neural_only", "commitment_neural_only", "commitment_hybrid"]

    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            continue

        print(f"\n=== Loading {dataset_name} ===")
        config = DATASETS[dataset_name]
        test_cases = load_segmentation_dataset(config)
        print(f"  Loaded {len(test_cases)} dialogues")

        if not test_cases:
            continue

        for strategy_type in strategy_types:
            if strategy_type == "commitment_hybrid":
                # Sweep thresholds for hybrid
                for threshold in drift_thresholds:
                    key = f"{strategy_type}_{threshold:.2f}"
                    print(f"\n--- {key} on {dataset_name} ---")

                    strategy = create_strategy(strategy_type, drift_threshold=threshold)
                    traces, metrics = evaluate_strategy_on_dataset(
                        strategy, test_cases, delay_tolerance=delay_tolerance
                    )

                    if key not in results:
                        results[key] = {}
                    results[key][dataset_name] = metrics

                    # Write traces if output dir specified
                    if output_dir:
                        trace_file = output_dir / f"{dataset_name}_{key}_traces.jsonl"
                        write_traces_jsonl(traces, trace_file)

                    print(f"  W-F1: {metrics.w_f1:.3f}  BOR: {metrics.bor:.2f}  "
                          f"Delay: {metrics.delay_mean:.2f}±{metrics.delay_std:.2f}  "
                          f"Coverage: {metrics.delay_coverage:.2f}")
                    print(f"  Churn: {metrics.suspect_abort_rate:.2%} abort rate  "
                          f"({metrics.abort_count} abort / {metrics.commit_count} commit)")
            else:
                # Non-hybrid strategies (no threshold sweep)
                print(f"\n--- {strategy_type} on {dataset_name} ---")
                strategy = create_strategy(strategy_type)
                traces, metrics = evaluate_strategy_on_dataset(
                    strategy, test_cases, delay_tolerance=delay_tolerance
                )

                if strategy_type not in results:
                    results[strategy_type] = {}
                results[strategy_type][dataset_name] = metrics

                if output_dir:
                    trace_file = output_dir / f"{dataset_name}_{strategy_type}_traces.jsonl"
                    write_traces_jsonl(traces, trace_file)

                print(f"  W-F1: {metrics.w_f1:.3f}  BOR: {metrics.bor:.2f}  "
                      f"Delay: {metrics.delay_mean:.2f}±{metrics.delay_std:.2f}  "
                      f"Coverage: {metrics.delay_coverage:.2f}")

    return results


def run_synthetic_tests(
    drift_thresholds: List[float] = [0.95],
) -> Dict[str, Dict[str, Any]]:
    """
    Run synthetic tests and report hard negative behavior.
    """
    print("\n=== Running Synthetic Tests ===")
    test_cases = create_synthetic_tests()
    print(f"Created {len(test_cases)} synthetic tests")

    results: Dict[str, Dict[str, Any]] = {}

    for threshold in drift_thresholds:
        key = f"hybrid_{threshold:.2f}"
        strategy = create_strategy("commitment_hybrid", drift_threshold=threshold)
        harness = OnlineReplayHarness(strategy, compute_drift=True)

        positive_tests = [t for t in test_cases if "positive" in t.tags]
        negative_tests = [t for t in test_cases if "negative" in t.tags]
        hard_negative_tests = [t for t in test_cases if "hard_negative" in t.tags]

        print(f"\n--- {key} ---")

        # Positive tests: should detect boundaries
        correct_positive = 0
        for tc in positive_tests:
            trace = harness.replay_dialogue(tc)
            gold = tc.get_canonical_boundaries()
            pred = trace.predicted_boundaries
            # Check if we detected something near the gold boundary
            if any(any(abs(p - g) <= 2 for g in gold) for p in pred):
                correct_positive += 1

        # Negative tests: should NOT detect boundaries
        correct_negative = 0
        for tc in negative_tests:
            trace = harness.replay_dialogue(tc)
            if not trace.predicted_boundaries:
                correct_negative += 1

        # Hard negative tests: should trigger SUSPECT but ABORT
        hard_neg_suspect_rate = 0
        hard_neg_abort_rate = 0
        hard_neg_spurious = 0

        for tc in hard_negative_tests:
            trace = harness.replay_dialogue(tc)
            # Check if any SUSPECT entry
            suspect_entries = [t for t in trace.turns if t.state == "SUSPECT"]
            if suspect_entries:
                hard_neg_suspect_rate += 1
            # Check abort events
            if trace.abort_events:
                hard_neg_abort_rate += 1
            # Check for spurious boundaries
            if trace.predicted_boundaries:
                hard_neg_spurious += 1

        results[key] = {
            'positive_accuracy': correct_positive / len(positive_tests) if positive_tests else 0,
            'negative_accuracy': correct_negative / len(negative_tests) if negative_tests else 0,
            'hard_neg_suspect_rate': hard_neg_suspect_rate / len(hard_negative_tests) if hard_negative_tests else 0,
            'hard_neg_abort_rate': hard_neg_abort_rate / len(hard_negative_tests) if hard_negative_tests else 0,
            'hard_neg_spurious_rate': hard_neg_spurious / len(hard_negative_tests) if hard_negative_tests else 0,
        }

        print(f"  Positive (boundary detected): {correct_positive}/{len(positive_tests)}")
        print(f"  Negative (no boundary): {correct_negative}/{len(negative_tests)}")
        print(f"  Hard negative SUSPECT entry rate: {results[key]['hard_neg_suspect_rate']:.1%}")
        print(f"  Hard negative ABORT rate: {results[key]['hard_neg_abort_rate']:.1%}")
        print(f"  Hard negative spurious boundary rate: {results[key]['hard_neg_spurious_rate']:.1%}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Online topic detection evaluation")
    parser.add_argument("--dataset", type=str, help="Single dataset to evaluate")
    parser.add_argument("--datasets", type=str, nargs="+",
                        help="Datasets for ablation (e.g., superseg dialseg)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation study")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run synthetic tests")
    parser.add_argument("--drift-thresholds", type=str, default="0.90,0.93,0.95,0.97,0.99",
                        help="Comma-separated drift thresholds to sweep")
    parser.add_argument("--delay-tolerance", type=int, default=2,
                        help="Tolerance window for delay matching")
    parser.add_argument("--output", type=str, help="Output directory for traces")
    args = parser.parse_args()

    # Parse thresholds
    thresholds = [float(t) for t in args.drift_thresholds.split(",")]

    # Output directory
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    if args.synthetic:
        results = run_synthetic_tests(drift_thresholds=thresholds)
        if output_dir:
            with open(output_dir / "synthetic_results.json", "w") as f:
                json.dump(results, f, indent=2)

    elif args.ablation or args.datasets:
        datasets = args.datasets or ["superseg", "dialseg", "tiage"]
        results = run_ablation(
            datasets=datasets,
            drift_thresholds=thresholds,
            delay_tolerance=args.delay_tolerance,
            output_dir=output_dir,
        )

        # Write summary
        if output_dir:
            summary = {
                strategy: {
                    dataset: metrics.to_dict()
                    for dataset, metrics in dataset_metrics.items()
                }
                for strategy, dataset_metrics in results.items()
            }
            with open(output_dir / "metrics_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Print comparison table
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Strategy':<30} {'Dataset':<12} {'W-F1':>8} {'BOR':>8} {'Delay':>8} {'Cov':>6}")
        print("-" * 80)
        for strategy, dataset_metrics in sorted(results.items()):
            for dataset, metrics in sorted(dataset_metrics.items()):
                print(f"{strategy:<30} {dataset:<12} {metrics.w_f1:>8.3f} "
                      f"{metrics.bor:>8.2f} {metrics.delay_mean:>8.2f} {metrics.delay_coverage:>6.2f}")

    elif args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            sys.exit(1)

        config = DATASETS[args.dataset]
        test_cases = load_segmentation_dataset(config)
        print(f"Loaded {len(test_cases)} dialogues from {args.dataset}")

        # Run default strategy
        strategy = create_strategy("commitment_hybrid", drift_threshold=0.95)
        traces, metrics = evaluate_strategy_on_dataset(
            strategy, test_cases, delay_tolerance=args.delay_tolerance
        )

        print(f"\nResults:")
        print(f"  W-F1: {metrics.w_f1:.3f}")
        print(f"  BOR: {metrics.bor:.2f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  Delay mean: {metrics.delay_mean:.2f} ± {metrics.delay_std:.2f}")
        print(f"  Delay coverage: {metrics.delay_coverage:.2f}")
        print(f"  Abort rate: {metrics.suspect_abort_rate:.2%}")

        if output_dir:
            write_traces_jsonl(traces, output_dir / f"{args.dataset}_traces.jsonl")
            print(f"\nTraces written to {output_dir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
