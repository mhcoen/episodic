"""
Memory eviction proxy for downstream evaluation of topic segmentation.

Tests whether boundaries cause correct context retention/eviction.
Uses deterministic string-match verification with injected facts.

NO LLM EVALUATION - fact retention is determined by:
1. Whether fact's turn is within context_budget of current position
2. Whether a committed boundary exists between fact and current position
"""

import uuid
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

from episodic.topics.evaluation import TestCase, Message
from episodic.topics.online_evaluation import DialogueTrace

logger = logging.getLogger(__name__)


# Fact templates for injection
FACT_TEMPLATES = [
    ("passport", "My passport number is {value}.", "PXFQ-{num:04d}"),
    ("codename", "The project codename is {value}.", "ORCHID-{num:d}"),
    ("phone", "My phone number is {value}.", "555-{num:04d}"),
    ("date", "The deadline is {value}.", "March {num:d}th"),
    ("account", "My account ID is {value}.", "ACC-{num:06d}"),
    ("reference", "The reference code is {value}.", "REF-{num:04d}"),
]


@dataclass
class InjectedFact:
    """A fact injected into a dialogue for memory testing."""
    turn_idx: int                # Index of message containing the fact
    fact_type: str               # "passport", "codename", etc.
    fact_value: str              # The unique fact value (e.g., "PXFQ-2917")
    original_text: str           # Full message text with fact


@dataclass
class MemoryProxyResult:
    """Result of checking a single fact's retention."""
    fact: InjectedFact
    check_turn_idx: int          # Turn index where retention was checked
    distance_from_boundary: int  # Turns from nearest committed boundary
    in_retained_context: bool    # Is fact's turn still in context?
    same_segment_as_current: bool  # Is fact in same segment as check turn?


class MemoryEvictionProxy:
    """
    Evaluate segmentation quality via deterministic memory retention.

    Deterministic retention rule (v1 - no summaries):
    A fact is retained iff its turn_idx is within context_budget of
    the current position AND no committed boundary exists between
    fact.turn_idx and current position.

    Example with context_budget=10, boundary at idx 15, current at idx 20:
    - Fact at idx 12: NOT retained (boundary between fact and current)
    - Fact at idx 16: retained (same segment, within budget)
    - Fact at idx 8: NOT retained (different segment + > budget)

    Usage:
        proxy = MemoryEvictionProxy(context_budget=10)
        modified_case, facts = proxy.inject_facts(test_case)
        trace = harness.replay_dialogue(modified_case)
        results = proxy.simulate_retention(trace, facts)
        metrics = compute_memory_metrics(results)
    """

    def __init__(self, context_budget: int = 10, seed: int = 42):
        """
        Initialize memory proxy.

        Args:
            context_budget: Number of turns retained in context window
            seed: Random seed for reproducible fact generation
        """
        self.context_budget = context_budget
        self.random = random.Random(seed)

    def inject_facts(
        self,
        test_case: TestCase,
        injection_density: float = 0.2
    ) -> Tuple[TestCase, List[InjectedFact]]:
        """
        Insert unique, greppable facts at strategic locations.

        Injects facts near expected gold boundaries, mid-segment, and at edges.

        Args:
            test_case: Original TestCase
            injection_density: Fraction of turns to inject facts into

        Returns:
            (modified_test_case, list_of_injected_facts)
        """
        facts: List[InjectedFact] = []
        messages = list(test_case.messages)  # Copy
        gold_boundaries = set(test_case.expected_boundaries)

        # Identify injection points
        injection_points = self._select_injection_points(
            len(messages), gold_boundaries, injection_density
        )

        # Inject facts
        fact_counter = 1000
        for turn_idx in injection_points:
            if turn_idx >= len(messages):
                continue

            msg = messages[turn_idx]
            fact_type, template, value_template = self.random.choice(FACT_TEMPLATES)
            fact_value = value_template.format(num=fact_counter)
            fact_counter += 1

            # Append fact to message content
            fact_sentence = template.format(value=fact_value)
            new_content = f"{msg.content} {fact_sentence}"

            # Create modified message
            messages[turn_idx] = Message(
                role=msg.role,
                content=new_content,
                node_id=msg.node_id,
                metadata=msg.metadata,
            )

            facts.append(InjectedFact(
                turn_idx=turn_idx,
                fact_type=fact_type,
                fact_value=fact_value,
                original_text=new_content,
            ))

        # Create modified test case
        modified_case = TestCase(
            id=test_case.id,
            name=test_case.name,
            description=test_case.description,
            messages=messages,
            expected_boundaries=test_case.expected_boundaries,
            boundary_alignment=test_case.boundary_alignment,
            expected_topic_names=test_case.expected_topic_names,
            retrieval_tests=test_case.retrieval_tests,
            tags=test_case.tags,
            difficulty=test_case.difficulty,
            source=test_case.source,
        )

        return modified_case, facts

    def _select_injection_points(
        self,
        num_messages: int,
        gold_boundaries: Set[int],
        density: float
    ) -> List[int]:
        """
        Select strategic injection points.

        Prioritizes:
        1. Just before gold boundaries (1-2 turns before)
        2. Just after gold boundaries (1-2 turns after)
        3. Mid-segment positions
        4. Dialogue edges
        """
        points: List[int] = []

        # Near gold boundaries (before and after)
        for boundary in gold_boundaries:
            if boundary - 2 >= 0:
                points.append(boundary - 2)
            if boundary - 1 >= 0:
                points.append(boundary - 1)
            if boundary + 1 < num_messages:
                points.append(boundary + 1)
            if boundary + 2 < num_messages:
                points.append(boundary + 2)

        # Edges
        if num_messages > 0:
            points.append(0)
        if num_messages > 1:
            points.append(num_messages - 1)

        # Fill to density with random mid-segment points
        target_count = int(num_messages * density)
        all_indices = set(range(num_messages))
        available = list(all_indices - set(points))
        self.random.shuffle(available)

        while len(points) < target_count and available:
            points.append(available.pop())

        return sorted(set(points))

    def simulate_retention(
        self,
        trace: DialogueTrace,
        facts: List[InjectedFact],
    ) -> List[MemoryProxyResult]:
        """
        Simulate memory eviction based on committed boundaries.

        Deterministic retention rule (v1 - no summaries):
        A fact is retained iff its turn_idx is within context_budget of
        the current position AND no committed boundary exists between
        fact.turn_idx and current position.

        Args:
            trace: DialogueTrace with committed boundaries
            facts: List of injected facts to check

        Returns:
            List of MemoryProxyResult for each fact
        """
        results: List[MemoryProxyResult] = []
        boundaries = sorted(trace.predicted_boundaries)

        # Check each fact at the end of dialogue
        check_turn_idx = len(trace.turns) - 1

        for fact in facts:
            # Find segment containing the fact
            fact_segment = self._get_segment(fact.turn_idx, boundaries)
            current_segment = self._get_segment(check_turn_idx, boundaries)

            same_segment = (fact_segment == current_segment)

            # Distance from nearest boundary
            distance = self._distance_from_boundary(fact.turn_idx, boundaries)

            # Check if any boundary between fact and current
            boundaries_between = [
                b for b in boundaries
                if fact.turn_idx < b <= check_turn_idx
            ]

            # Retention rule: within budget AND no boundary between
            within_budget = (check_turn_idx - fact.turn_idx) <= self.context_budget
            no_boundary_between = len(boundaries_between) == 0
            in_retained = within_budget and no_boundary_between

            results.append(MemoryProxyResult(
                fact=fact,
                check_turn_idx=check_turn_idx,
                distance_from_boundary=distance,
                in_retained_context=in_retained,
                same_segment_as_current=same_segment,
            ))

        return results

    def _get_segment(self, turn_idx: int, boundaries: List[int]) -> int:
        """Get segment number for a turn index."""
        segment = 0
        for b in boundaries:
            if turn_idx >= b:
                segment += 1
            else:
                break
        return segment

    def _distance_from_boundary(self, turn_idx: int, boundaries: List[int]) -> int:
        """Get distance to nearest boundary."""
        if not boundaries:
            return -1  # No boundaries

        distances = [abs(turn_idx - b) for b in boundaries]
        return min(distances)

    def check_fact_present(self, context: List[str], fact: InjectedFact) -> bool:
        """
        Check if fact value is present in context via string match.

        Args:
            context: List of message content strings
            fact: Fact to check

        Returns:
            True if fact_value found in any context string
        """
        return any(fact.fact_value in turn for turn in context)


def compute_memory_metrics(results: List[MemoryProxyResult]) -> Dict[str, float]:
    """
    Compute memory retention metrics from proxy results.

    Args:
        results: List of MemoryProxyResult from simulate_retention

    Returns:
        Dict with retention metrics
    """
    if not results:
        return {
            'retention_rate': 0.0,
            'same_segment_retention': 0.0,
            'cross_segment_retention': 0.0,
            'retention_by_distance': {},
        }

    total = len(results)
    retained = sum(1 for r in results if r.in_retained_context)

    # Retention by segment relationship
    same_segment = [r for r in results if r.same_segment_as_current]
    cross_segment = [r for r in results if not r.same_segment_as_current]

    same_retained = sum(1 for r in same_segment if r.in_retained_context)
    cross_retained = sum(1 for r in cross_segment if r.in_retained_context)

    # Retention by distance
    retention_by_distance: Dict[int, Dict[str, int]] = {}
    for r in results:
        d = r.distance_from_boundary
        if d not in retention_by_distance:
            retention_by_distance[d] = {'total': 0, 'retained': 0}
        retention_by_distance[d]['total'] += 1
        if r.in_retained_context:
            retention_by_distance[d]['retained'] += 1

    distance_rates = {
        d: v['retained'] / v['total'] if v['total'] > 0 else 0.0
        for d, v in retention_by_distance.items()
    }

    return {
        'retention_rate': retained / total if total > 0 else 0.0,
        'same_segment_retention': same_retained / len(same_segment) if same_segment else 0.0,
        'cross_segment_retention': cross_retained / len(cross_segment) if cross_segment else 0.0,
        'retention_by_distance': distance_rates,
        'total_facts': total,
        'retained_facts': retained,
    }
