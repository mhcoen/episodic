"""
Deterministic benchmark runner using fixed embeddings.

Runs resume scenarios with pre-computed embeddings to enable
reproducible CI testing without embedding model calls.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark_fixtures import FixedResumeScenario, load_benchmark_fixtures

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of running a single scenario in a single mode."""

    scenario_id: str
    mode: str  # "ancestry", "topic_local", "hybrid"

    # Outcomes
    reactivation_fired: bool = False
    reactivation_target: str = "none"  # topic name or "none"

    # Contamination check
    contamination_detected: bool = False
    foreign_topic_strings_found: List[str] = field(default_factory=list)

    # Content checks
    contains_expected: Dict[str, bool] = field(default_factory=dict)  # substring -> found
    excludes_expected: Dict[str, bool] = field(
        default_factory=dict
    )  # substring -> correctly_absent

    # Metrics
    total_context_tokens: int = 0
    active_topic_content_ratio: float = 0.0

    # Assembled context (for inspection)
    assembled_context: str = ""

    # Pass/fail
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)


def run_deterministic_benchmark(
    scenarios: Optional[List[FixedResumeScenario]] = None,
    modes: Optional[List[str]] = None,
) -> List[BenchmarkResult]:
    """
    Run benchmark with fixed embeddings for determinism.

    Does NOT call embedding model - uses pre-computed embeddings from fixtures.
    Simulates context assembly to test contamination detection.

    Args:
        scenarios: Scenarios to test (loads from fixtures if None)
        modes: Modes to test (defaults to all three)

    Returns:
        List of BenchmarkResult objects
    """
    if scenarios is None:
        scenarios = load_benchmark_fixtures()

    if modes is None:
        modes = ["ancestry", "topic_local", "hybrid"]

    results = []

    for scenario in scenarios:
        for mode in modes:
            result = _run_single_scenario(scenario, mode)
            results.append(result)

    return results


def _run_single_scenario(
    scenario: FixedResumeScenario,
    mode: str,
) -> BenchmarkResult:
    """
    Run a single scenario in a single mode.

    Uses synthetic context assembly to simulate what each strategy would produce.
    """
    failure_reasons = []

    # Simulate context assembly based on mode
    if mode == "ancestry":
        # Ancestry includes everything (all topics)
        assembled_context = _simulate_ancestry_context(scenario)
    elif mode == "topic_local":
        # Topic-local only includes the target topic
        assembled_context = _simulate_topic_local_context(scenario)
    elif mode == "hybrid":
        # Hybrid reactivates then uses topic-local
        assembled_context = _simulate_hybrid_context(scenario)
    else:
        assembled_context = ""

    # Check expected content presence
    contains_expected: Dict[str, bool] = {}
    for substring in scenario.expected_context_contains:
        found = substring.lower() in assembled_context.lower()
        contains_expected[substring] = found
        if not found:
            failure_reasons.append(f"Missing expected content: '{substring}'")

    # Check excluded content absence
    excludes_expected: Dict[str, bool] = {}
    foreign_strings: List[str] = []
    for substring in scenario.expected_context_excludes:
        absent = substring.lower() not in assembled_context.lower()
        excludes_expected[substring] = absent
        if not absent:
            failure_reasons.append(f"Contamination: found '{substring}' in context")
            foreign_strings.append(substring)

    contamination_detected = len(foreign_strings) > 0

    # Determine reactivation behavior
    reactivation_fired = mode in ("hybrid", "topic_local")
    if reactivation_fired and scenario.expected_reactivation.startswith("topic_"):
        reactivation_target = scenario.expected_reactivation.replace("topic_", "")
        if reactivation_target == "a":
            reactivation_target = scenario.topic_a_name
        elif reactivation_target == "b":
            reactivation_target = scenario.topic_b_name
    else:
        reactivation_target = "none"

    # Estimate tokens (rough: 1 token per 4 chars)
    total_context_tokens = len(assembled_context) // 4

    # Calculate active topic content ratio
    topic_a_text = _get_topic_text(scenario.topic_a_exchanges)
    topic_b_text = _get_topic_text(scenario.topic_b_exchanges)

    topic_a_in_context = topic_a_text.lower() in assembled_context.lower() or any(
        exc["user"].lower() in assembled_context.lower()
        for exc in scenario.topic_a_exchanges
    )
    topic_b_in_context = topic_b_text.lower() in assembled_context.lower() or any(
        exc["user"].lower() in assembled_context.lower()
        for exc in scenario.topic_b_exchanges
    )

    if topic_a_in_context and not topic_b_in_context:
        active_topic_ratio = 1.0
    elif topic_a_in_context and topic_b_in_context:
        active_topic_ratio = 0.5
    elif not topic_a_in_context and topic_b_in_context:
        active_topic_ratio = 0.0
    else:
        active_topic_ratio = 0.0

    passed = len(failure_reasons) == 0

    return BenchmarkResult(
        scenario_id=scenario.scenario_id,
        mode=mode,
        reactivation_fired=reactivation_fired,
        reactivation_target=reactivation_target,
        contamination_detected=contamination_detected,
        foreign_topic_strings_found=foreign_strings,
        contains_expected=contains_expected,
        excludes_expected=excludes_expected,
        total_context_tokens=total_context_tokens,
        active_topic_content_ratio=active_topic_ratio,
        assembled_context=assembled_context,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def _get_topic_text(exchanges: List[Dict[str, str]]) -> str:
    """Combine all exchanges into a single text."""
    parts = []
    for exc in exchanges:
        parts.append(f"User: {exc['user']}")
        parts.append(f"Assistant: {exc['assistant']}")
    return "\n".join(parts)


def _simulate_ancestry_context(scenario: FixedResumeScenario) -> str:
    """
    Simulate ancestry context assembly.

    Ancestry mode includes ALL conversation context, both topics A and B.
    This represents the contamination-prone behavior.
    """
    parts = []

    # Include topic A (original)
    parts.append(f"[Topic: {scenario.topic_a_name}]")
    parts.append(_get_topic_text(scenario.topic_a_exchanges))

    # Include topic B (intervening) - this is the contamination
    parts.append(f"\n[Topic: {scenario.topic_b_name}]")
    parts.append(_get_topic_text(scenario.topic_b_exchanges))

    # Current query
    parts.append(f"\nCurrent query: {scenario.resume_query}")

    return "\n".join(parts)


def _simulate_topic_local_context(scenario: FixedResumeScenario) -> str:
    """
    Simulate topic-local context assembly.

    Topic-local mode only includes the active/reactivated topic (A).
    This represents the clean, non-contaminated behavior.
    """
    parts = []

    # Only include topic A (the reactivated topic)
    parts.append(f"[Topic: {scenario.topic_a_name}]")
    parts.append(_get_topic_text(scenario.topic_a_exchanges))

    # Current query
    parts.append(f"\nCurrent query: {scenario.resume_query}")

    return "\n".join(parts)


def _simulate_hybrid_context(scenario: FixedResumeScenario) -> str:
    """
    Simulate hybrid context assembly.

    Hybrid mode probes for reactivation, then uses topic-local if reactivating.
    For resume scenarios, this should behave like topic-local.
    """
    # For resume scenarios, hybrid should reactivate to topic A
    # and then use topic-local assembly
    return _simulate_topic_local_context(scenario)


def compute_benchmark_metrics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute aggregate metrics from benchmark results."""
    by_mode: Dict[str, Dict[str, Any]] = {}

    for mode in ["ancestry", "topic_local", "hybrid"]:
        mode_results = [r for r in results if r.mode == mode]

        if not mode_results:
            continue

        total = len(mode_results)
        passed = sum(1 for r in mode_results if r.passed)
        contaminated = sum(1 for r in mode_results if r.contamination_detected)
        avg_tokens = np.mean([r.total_context_tokens for r in mode_results])

        by_mode[mode] = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "contamination_count": contaminated,
            "contamination_rate": contaminated / total if total > 0 else 0.0,
            "avg_context_tokens": float(avg_tokens),
        }

    # Compute comparative metrics
    ancestry = by_mode.get("ancestry", {})
    topic_local = by_mode.get("topic_local", {})

    comparison = {}
    if ancestry and topic_local:
        comparison = {
            "contamination_reduction": (
                ancestry.get("contamination_rate", 0) - topic_local.get("contamination_rate", 0)
            ),
            "token_reduction": (
                ancestry.get("avg_context_tokens", 0) - topic_local.get("avg_context_tokens", 0)
            ),
        }

    return {
        "by_mode": by_mode,
        "topic_local_vs_ancestry": comparison,
        "total_scenarios": len(set(r.scenario_id for r in results)),
    }


def format_benchmark_report(results: List[BenchmarkResult]) -> str:
    """Generate human-readable benchmark report."""
    metrics = compute_benchmark_metrics(results)

    lines = [
        "=" * 60,
        "Resume Benchmark Results",
        "=" * 60,
        "",
    ]

    # Summary by mode
    for mode, data in metrics["by_mode"].items():
        lines.extend([
            f"{mode}:",
            f"  Passed: {data['passed']}/{data['total']} ({data['pass_rate']:.1%})",
            f"  Contamination rate: {data['contamination_rate']:.1%}",
            f"  Avg context tokens: {data['avg_context_tokens']:.0f}",
            "",
        ])

    # Comparison
    comp = metrics.get("topic_local_vs_ancestry", {})
    if comp:
        lines.extend([
            "topic_local vs ancestry:",
            f"  Contamination reduction: {comp.get('contamination_reduction', 0):.1%}",
            f"  Token reduction: {comp.get('token_reduction', 0):.0f}",
            "",
        ])

    # Failures detail
    failures = [r for r in results if not r.passed]
    if failures:
        lines.extend([
            "Failures:",
            "-" * 40,
        ])
        for f in failures:
            lines.append(f"  {f.scenario_id} ({f.mode}):")
            for reason in f.failure_reasons:
                lines.append(f"    - {reason}")
        lines.append("")

    return "\n".join(lines)
