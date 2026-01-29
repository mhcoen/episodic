"""
Resume benchmark harness for comparing context recovery strategies.

Compares ancestry vs hybrid vs topic_local modes on scenarios requiring
historical topic context.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from episodic.context_recovery import (
    ContextRecoveryMode,
    ContextAssemblyResult,
    select_strategy,
)
from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)


class BenchmarkScenarioType(Enum):
    """Types of resume scenarios."""
    IMMEDIATE_RESUME = "immediate_resume"  # Resume after 1-2 turns away
    SHORT_GAP = "short_gap"  # Resume after 3-10 turns
    LONG_GAP = "long_gap"  # Resume after 10+ turns
    CROSS_TOPIC_IMPORT = "cross_topic_import"  # Explicit import from another topic


@dataclass
class BenchmarkScenario:
    """A single benchmark scenario."""
    scenario_id: str
    scenario_type: BenchmarkScenarioType
    user_turn_text: str
    user_node_id: str
    target_topic_start_node_id: str
    target_topic_name: str
    gap_turns: int  # Number of turns since last topic activity
    expected_context_node_ids: List[str]  # Nodes that should be in context
    forbidden_context_node_ids: List[str] = field(default_factory=list)  # Nodes that shouldn't be in context


@dataclass
class StrategyResult:
    """Result of running a strategy on a scenario."""
    mode: ContextRecoveryMode
    scenario_id: str
    included_expected: int  # How many expected nodes were included
    total_expected: int  # Total expected nodes
    included_forbidden: int  # How many forbidden nodes were included
    recall: float  # included_expected / total_expected
    precision: float  # (included_nodes - included_forbidden) / included_nodes
    assembly_time_ms: float
    included_node_ids: List[str]
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    total_scenarios: int
    by_mode: Dict[str, Dict[str, float]]  # mode -> metric -> value
    by_scenario_type: Dict[str, Dict[str, Dict[str, float]]]  # type -> mode -> metric -> value
    details: List[StrategyResult]


def find_resume_scenarios(
    min_gap_turns: int = 3,
    max_scenarios: int = 50,
    conn: Optional[sqlite3.Connection] = None
) -> List[BenchmarkScenario]:
    """
    Find natural resume scenarios from conversation history.

    Looks for:
    1. User turns that triggered reactivation decisions
    2. Topics that were dormant for at least min_gap_turns
    3. Subsequent activity in the target topic

    Args:
        min_gap_turns: Minimum turns of gap to qualify as resume
        max_scenarios: Maximum scenarios to return
        conn: Optional database connection

    Returns:
        List of BenchmarkScenario objects
    """
    def _find(c: sqlite3.Connection) -> List[BenchmarkScenario]:
        scenarios = []

        # Find reactivation decisions where user resumed a topic
        cursor = c.execute("""
            SELECT
                rd.user_node_id,
                rd.decision,
                rd.topic_name,
                rd.topic_start_node_id,
                rd.dormancy_turns,
                n.content
            FROM reactivation_decisions rd
            JOIN nodes n ON rd.user_node_id = n.id
            WHERE rd.decision = 'REACTIVATE'
            AND rd.dormancy_turns >= ?
            ORDER BY rd.created_at DESC
            LIMIT ?
        """, (min_gap_turns, max_scenarios * 2))  # Fetch extra to filter

        for row in cursor.fetchall():
            user_node_id = row[0]
            topic_name = row[2]
            topic_start_node_id = row[3]
            dormancy_turns = row[4] or 0
            user_content = row[5] or ""

            # Get nodes that should be in context (from target topic)
            topic_cursor = c.execute("""
                SELECT node_id FROM topic_nodes
                WHERE topic_start_node_id = ?
                ORDER BY turn_idx DESC
                LIMIT 10
            """, (topic_start_node_id,))
            expected_nodes = [r[0] for r in topic_cursor.fetchall()]

            if not expected_nodes:
                continue

            # Get nodes that should NOT be in context (from other topics)
            forbidden_cursor = c.execute("""
                SELECT tn.node_id
                FROM topic_nodes tn
                JOIN topics t ON tn.topic_start_node_id = t.start_node_id
                WHERE t.start_node_id != ?
                ORDER BY RANDOM()
                LIMIT 5
            """, (topic_start_node_id,))
            forbidden_nodes = [r[0] for r in forbidden_cursor.fetchall()]

            # Determine scenario type
            if dormancy_turns < 3:
                scenario_type = BenchmarkScenarioType.IMMEDIATE_RESUME
            elif dormancy_turns < 10:
                scenario_type = BenchmarkScenarioType.SHORT_GAP
            else:
                scenario_type = BenchmarkScenarioType.LONG_GAP

            scenarios.append(BenchmarkScenario(
                scenario_id=f"resume_{user_node_id}",
                scenario_type=scenario_type,
                user_turn_text=user_content,
                user_node_id=user_node_id,
                target_topic_start_node_id=topic_start_node_id,
                target_topic_name=topic_name,
                gap_turns=dormancy_turns,
                expected_context_node_ids=expected_nodes,
                forbidden_context_node_ids=forbidden_nodes,
            ))

            if len(scenarios) >= max_scenarios:
                break

        return scenarios

    if conn is not None:
        return _find(conn)

    with get_connection() as c:
        return _find(c)


def run_scenario(
    scenario: BenchmarkScenario,
    mode: ContextRecoveryMode,
    token_budget: int = 4000,
    conn: Optional[sqlite3.Connection] = None,
    chroma_collection: Optional[Any] = None,
) -> StrategyResult:
    """
    Run a single scenario with a specific context recovery mode.

    Args:
        scenario: The benchmark scenario to run
        mode: The context recovery mode to test
        token_budget: Token budget for context assembly
        conn: Optional database connection
        chroma_collection: Optional Chroma collection

    Returns:
        StrategyResult with metrics
    """
    from episodic.recall.reactivation import ReactivationDecision

    # For hybrid mode, we need to simulate the reactivation decision
    if mode == ContextRecoveryMode.HYBRID:
        reactivation_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name=scenario.target_topic_name,
            topic_start_node_id=scenario.target_topic_start_node_id,
        )
    else:
        reactivation_decision = None

    strategy = select_strategy(mode, reactivation_decision)

    start_time = time.perf_counter()

    result = strategy.assemble(
        user_turn_text=scenario.user_turn_text,
        user_node_id=scenario.user_node_id,
        active_topic_start_node_id=scenario.target_topic_start_node_id,
        user_embedding=None,
        token_budget=token_budget,
        conn=conn,
        chroma_collection=chroma_collection,
    )

    assembly_time_ms = (time.perf_counter() - start_time) * 1000

    # Extract included node IDs from result
    included_node_ids = result.debug.get("included_node_ids", [])

    # Count expected nodes that were included
    included_expected = sum(
        1 for node_id in scenario.expected_context_node_ids
        if node_id in included_node_ids
    )
    total_expected = len(scenario.expected_context_node_ids)

    # Count forbidden nodes that were included
    included_forbidden = sum(
        1 for node_id in scenario.forbidden_context_node_ids
        if node_id in included_node_ids
    )

    # Compute metrics
    recall = included_expected / total_expected if total_expected > 0 else 0.0

    total_included = len(included_node_ids)
    correct_included = total_included - included_forbidden
    precision = correct_included / total_included if total_included > 0 else 0.0

    return StrategyResult(
        mode=mode,
        scenario_id=scenario.scenario_id,
        included_expected=included_expected,
        total_expected=total_expected,
        included_forbidden=included_forbidden,
        recall=recall,
        precision=precision,
        assembly_time_ms=assembly_time_ms,
        included_node_ids=included_node_ids,
        debug=result.debug,
    )


def run_benchmark(
    scenarios: Optional[List[BenchmarkScenario]] = None,
    modes: Optional[List[ContextRecoveryMode]] = None,
    token_budget: int = 4000,
    conn: Optional[sqlite3.Connection] = None,
    chroma_collection: Optional[Any] = None,
) -> BenchmarkResult:
    """
    Run the full benchmark comparing all modes.

    Args:
        scenarios: Scenarios to test (auto-discovers if None)
        modes: Modes to compare (defaults to ancestry, topic_local, hybrid)
        token_budget: Token budget for context assembly
        conn: Optional database connection
        chroma_collection: Optional Chroma collection

    Returns:
        BenchmarkResult with aggregated metrics
    """
    if modes is None:
        modes = [
            ContextRecoveryMode.ANCESTRY,
            ContextRecoveryMode.TOPIC_LOCAL,
            ContextRecoveryMode.HYBRID,
        ]

    if scenarios is None:
        scenarios = find_resume_scenarios(conn=conn)

    if not scenarios:
        logger.warning("No benchmark scenarios found")
        return BenchmarkResult(
            total_scenarios=0,
            by_mode={},
            by_scenario_type={},
            details=[],
        )

    all_results: List[StrategyResult] = []

    for scenario in scenarios:
        for mode in modes:
            result = run_scenario(
                scenario=scenario,
                mode=mode,
                token_budget=token_budget,
                conn=conn,
                chroma_collection=chroma_collection,
            )
            all_results.append(result)

    # Aggregate by mode
    by_mode: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        mode_results = [r for r in all_results if r.mode == mode]
        if mode_results:
            by_mode[mode.value] = {
                "recall": np.mean([r.recall for r in mode_results]),
                "precision": np.mean([r.precision for r in mode_results]),
                "avg_assembly_ms": np.mean([r.assembly_time_ms for r in mode_results]),
                "avg_included_expected": np.mean([r.included_expected for r in mode_results]),
                "avg_included_forbidden": np.mean([r.included_forbidden for r in mode_results]),
            }

    # Aggregate by scenario type
    by_scenario_type: Dict[str, Dict[str, Dict[str, float]]] = {}
    scenario_map = {s.scenario_id: s for s in scenarios}

    for scenario_type in BenchmarkScenarioType:
        type_scenarios = [s for s in scenarios if s.scenario_type == scenario_type]
        if not type_scenarios:
            continue

        by_scenario_type[scenario_type.value] = {}

        for mode in modes:
            type_mode_results = [
                r for r in all_results
                if r.mode == mode
                and scenario_map.get(r.scenario_id, BenchmarkScenario("", BenchmarkScenarioType.IMMEDIATE_RESUME, "", "", "", "", 0, [])).scenario_type == scenario_type
            ]
            if type_mode_results:
                by_scenario_type[scenario_type.value][mode.value] = {
                    "recall": np.mean([r.recall for r in type_mode_results]),
                    "precision": np.mean([r.precision for r in type_mode_results]),
                    "count": len(type_mode_results),
                }

    return BenchmarkResult(
        total_scenarios=len(scenarios),
        by_mode=by_mode,
        by_scenario_type=by_scenario_type,
        details=all_results,
    )


def get_benchmark_summary(result: BenchmarkResult) -> str:
    """Generate human-readable summary of benchmark results."""
    lines = [
        "Resume Benchmark Summary",
        "=" * 50,
        f"Total scenarios: {result.total_scenarios}",
        "",
    ]

    if not result.by_mode:
        lines.append("No results to display.")
        return "\n".join(lines)

    # Overall by mode
    lines.append("Overall Results by Mode:")
    lines.append("-" * 40)
    lines.append(f"{'Mode':<15} {'Recall':<10} {'Precision':<10} {'Time (ms)':<10}")
    lines.append("-" * 40)

    for mode, metrics in sorted(result.by_mode.items()):
        lines.append(
            f"{mode:<15} {metrics['recall']:.1%}      {metrics['precision']:.1%}       {metrics['avg_assembly_ms']:.1f}"
        )

    # By scenario type
    if result.by_scenario_type:
        lines.append("")
        lines.append("Results by Scenario Type:")
        lines.append("-" * 50)

        for scenario_type, mode_metrics in sorted(result.by_scenario_type.items()):
            lines.append(f"\n{scenario_type.replace('_', ' ').title()}:")
            for mode, metrics in sorted(mode_metrics.items()):
                lines.append(
                    f"  {mode:<15} recall={metrics['recall']:.1%}  precision={metrics['precision']:.1%}  (n={int(metrics['count'])})"
                )

    return "\n".join(lines)


def export_benchmark_results(result: BenchmarkResult, output_path: str) -> None:
    """Export benchmark results to JSON."""
    export_data = {
        "total_scenarios": result.total_scenarios,
        "by_mode": result.by_mode,
        "by_scenario_type": result.by_scenario_type,
        "details": [
            {
                "mode": r.mode.value,
                "scenario_id": r.scenario_id,
                "recall": r.recall,
                "precision": r.precision,
                "included_expected": r.included_expected,
                "total_expected": r.total_expected,
                "included_forbidden": r.included_forbidden,
                "assembly_time_ms": r.assembly_time_ms,
            }
            for r in result.details
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Exported benchmark results to {output_path}")
