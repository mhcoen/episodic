#!/usr/bin/env python3
"""
Latency and token stability benchmark for topic_local context assembly.

Creates a synthetic workload with 20 topics and long working sets,
then measures assembly performance and validates token budgets.

Run: python scripts/latency_benchmark.py
Output: episodic/evaluation/reports/latency_benchmark.json
"""

import json
import os
import sqlite3
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test mode to allow force_no_recency
os.environ["EPISODIC_TEST_MODE"] = "1"


@dataclass
class BenchmarkRun:
    """Single benchmark run result."""

    topic_id: str
    topic_name: str
    assembly_ms: float
    sqlite_ms: float
    chroma_ms: float
    total_tokens: int
    token_breakdown: Dict[str, int]
    fallback_triggered: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    timestamp: str
    config: Dict[str, Any]
    runs: List[BenchmarkRun]
    statistics: Dict[str, Any] = field(default_factory=dict)
    assertions: Dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False


def create_synthetic_topic(
    conn: sqlite3.Connection,
    topic_name: str,
    exchange_count: int = 20,
) -> str:
    """
    Create a synthetic topic with exchanges in the database.

    Args:
        conn: SQLite connection
        topic_name: Name for the topic
        exchange_count: Number of user/assistant exchange pairs

    Returns:
        topic_start_node_id
    """
    cursor = conn.cursor()

    # Create the first node as topic start
    topic_start_id = f"bench_{uuid.uuid4().hex[:12]}"
    parent_id = None

    for i in range(exchange_count):
        # User node
        user_id = topic_start_id if i == 0 else f"bench_{uuid.uuid4().hex[:12]}"
        user_content = f"User message {i+1} for topic {topic_name}. " * 10  # ~100 chars

        cursor.execute(
            """
            INSERT INTO conversation_nodes (id, role, content, parent_id, created_at)
            VALUES (?, 'user', ?, ?, CURRENT_TIMESTAMP)
        """,
            (user_id, user_content, parent_id),
        )

        # Assistant node
        asst_id = f"bench_{uuid.uuid4().hex[:12]}"
        asst_content = f"Assistant response {i+1} for {topic_name}. " * 15  # ~150 chars

        cursor.execute(
            """
            INSERT INTO conversation_nodes (id, role, content, parent_id, created_at)
            VALUES (?, 'assistant', ?, ?, CURRENT_TIMESTAMP)
        """,
            (asst_id, asst_content, user_id),
        )

        # Add to topic_nodes
        cursor.execute(
            """
            INSERT OR IGNORE INTO topic_nodes (node_id, topic_start_node_id, turn_idx)
            VALUES (?, ?, ?)
        """,
            (user_id, topic_start_id, i * 2),
        )
        cursor.execute(
            """
            INSERT OR IGNORE INTO topic_nodes (node_id, topic_start_node_id, turn_idx)
            VALUES (?, ?, ?)
        """,
            (asst_id, topic_start_id, i * 2 + 1),
        )

        parent_id = asst_id

    # Create working set entry
    cursor.execute(
        """
        INSERT OR REPLACE INTO topic_working_set
        (topic_start_node_id, topic_name, last_updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """,
        (topic_start_id, topic_name),
    )

    conn.commit()
    return topic_start_id


def run_assembly_benchmark(
    topic_start_id: str,
    topic_name: str,
    conn: sqlite3.Connection,
    token_budget: int = 4000,
) -> BenchmarkRun:
    """Run a single assembly benchmark for a topic."""
    from episodic.context_recovery.topic_local import TopicLocalStrategy

    strategy = TopicLocalStrategy()

    # Create a fake embedding (normalized random vector)
    np.random.seed(hash(topic_start_id) % (2**32))
    fake_embedding = np.random.randn(384).astype(np.float32)
    fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)

    user_text = "What were we discussing about this topic?"

    try:
        result = strategy.assemble(
            user_turn_text=user_text,
            user_node_id=None,
            active_topic_start_node_id=topic_start_id,
            user_embedding=fake_embedding,
            token_budget=token_budget,
            conn=conn,
            chroma_collection=None,  # No Chroma for benchmark
            force_no_recency=False,
        )

        debug = result.debug
        timing = debug.get("timing", {})
        token_breakdown = debug.get("token_breakdown", {})

        return BenchmarkRun(
            topic_id=topic_start_id[:8],
            topic_name=topic_name,
            assembly_ms=timing.get("context_assembly_ms", 0.0),
            sqlite_ms=timing.get("sqlite_ops_ms", 0.0),
            chroma_ms=timing.get("chroma_query_ms", 0.0),
            total_tokens=token_breakdown.get("total_tokens", 0),
            token_breakdown=token_breakdown,
            fallback_triggered=debug.get("fallback_reason") is not None,
        )

    except Exception as e:
        return BenchmarkRun(
            topic_id=topic_start_id[:8],
            topic_name=topic_name,
            assembly_ms=0.0,
            sqlite_ms=0.0,
            chroma_ms=0.0,
            total_tokens=0,
            token_breakdown={},
            error=str(e),
        )


def run_benchmark(
    num_topics: int = 20,
    exchanges_per_topic: int = 30,
    iterations_per_topic: int = 3,
    token_budget: int = 4000,
) -> BenchmarkReport:
    """Run the full latency benchmark."""
    from episodic.config import config
    from episodic.db_connection import get_connection

    print(f"Running latency benchmark: {num_topics} topics, {exchanges_per_topic} exchanges each")

    # Capture config
    benchmark_config = {
        "num_topics": num_topics,
        "exchanges_per_topic": exchanges_per_topic,
        "iterations_per_topic": iterations_per_topic,
        "token_budget": token_budget,
        "min_anchors_for_topic_local": config.get("min_anchors_for_topic_local", 2),
        "min_tokens_for_topic_local": config.get("min_tokens_for_topic_local", 500),
    }

    runs: List[BenchmarkRun] = []

    with get_connection() as conn:
        # Create synthetic topics
        topic_ids = []
        print("Creating synthetic topics...")
        for i in range(num_topics):
            topic_name = f"benchmark-topic-{i+1}"
            topic_id = create_synthetic_topic(conn, topic_name, exchanges_per_topic)
            topic_ids.append((topic_id, topic_name))
            print(f"  Created {topic_name} ({topic_id[:8]})")

        # Run benchmarks
        print(f"\nRunning {iterations_per_topic} iterations per topic...")
        for iteration in range(iterations_per_topic):
            print(f"  Iteration {iteration + 1}/{iterations_per_topic}")
            for topic_id, topic_name in topic_ids:
                result = run_assembly_benchmark(
                    topic_id, topic_name, conn, token_budget
                )
                runs.append(result)

        # Cleanup synthetic data
        print("\nCleaning up synthetic data...")
        cursor = conn.cursor()
        for topic_id, _ in topic_ids:
            cursor.execute(
                "DELETE FROM conversation_nodes WHERE id LIKE 'bench_%'"
            )
            cursor.execute(
                "DELETE FROM topic_nodes WHERE topic_start_node_id = ?",
                (topic_id,),
            )
            cursor.execute(
                "DELETE FROM topic_working_set WHERE topic_start_node_id = ?",
                (topic_id,),
            )
        conn.commit()

    # Compute statistics
    successful_runs = [r for r in runs if r.error is None]
    assembly_times = [r.assembly_ms for r in successful_runs]
    total_tokens_list = [r.total_tokens for r in successful_runs]

    if assembly_times:
        assembly_times_sorted = sorted(assembly_times)
        p50_idx = int(len(assembly_times_sorted) * 0.5)
        p95_idx = int(len(assembly_times_sorted) * 0.95)
        p99_idx = int(len(assembly_times_sorted) * 0.99)

        stats = {
            "run_count": len(runs),
            "successful_runs": len(successful_runs),
            "error_count": len(runs) - len(successful_runs),
            "assembly_ms": {
                "min": min(assembly_times),
                "max": max(assembly_times),
                "mean": statistics.mean(assembly_times),
                "median": statistics.median(assembly_times),
                "stdev": statistics.stdev(assembly_times) if len(assembly_times) > 1 else 0,
                "p50": assembly_times_sorted[p50_idx] if p50_idx < len(assembly_times_sorted) else 0,
                "p95": assembly_times_sorted[p95_idx] if p95_idx < len(assembly_times_sorted) else 0,
                "p99": assembly_times_sorted[min(p99_idx, len(assembly_times_sorted) - 1)],
            },
            "tokens": {
                "min": min(total_tokens_list) if total_tokens_list else 0,
                "max": max(total_tokens_list) if total_tokens_list else 0,
                "mean": statistics.mean(total_tokens_list) if total_tokens_list else 0,
                "budget": token_budget,
                "over_budget_count": sum(1 for t in total_tokens_list if t > token_budget),
            },
            "fallback_rate": sum(1 for r in successful_runs if r.fallback_triggered) / len(successful_runs) if successful_runs else 0,
        }
    else:
        stats = {"error": "No successful runs"}

    # Assertions
    assertions = {}
    if "assembly_ms" in stats:
        # p95 assembly < 50ms
        assertions["p95_assembly_under_50ms"] = stats["assembly_ms"]["p95"] < 50.0
        # p99 tokens <= budget
        assertions["p99_tokens_within_budget"] = stats["tokens"]["over_budget_count"] == 0
        # No errors
        assertions["no_errors"] = stats["error_count"] == 0

    overall_pass = all(assertions.values()) if assertions else False

    report = BenchmarkReport(
        timestamp=datetime.utcnow().isoformat(),
        config=benchmark_config,
        runs=[],  # Don't include full runs in report (too large)
        statistics=stats,
        assertions=assertions,
        overall_pass=overall_pass,
    )

    return report


def main():
    """Run benchmark and save report."""
    print("=" * 60)
    print("Latency & Token Stability Benchmark")
    print("=" * 60)

    report = run_benchmark(
        num_topics=20,
        exchanges_per_topic=30,
        iterations_per_topic=3,
        token_budget=4000,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    stats = report.statistics
    if "assembly_ms" in stats:
        print(f"\nAssembly Time (ms):")
        print(f"  Min:    {stats['assembly_ms']['min']:.2f}")
        print(f"  Mean:   {stats['assembly_ms']['mean']:.2f}")
        print(f"  Median: {stats['assembly_ms']['median']:.2f}")
        print(f"  p95:    {stats['assembly_ms']['p95']:.2f}")
        print(f"  p99:    {stats['assembly_ms']['p99']:.2f}")
        print(f"  Max:    {stats['assembly_ms']['max']:.2f}")

        print(f"\nToken Usage:")
        print(f"  Min:    {stats['tokens']['min']}")
        print(f"  Mean:   {stats['tokens']['mean']:.1f}")
        print(f"  Max:    {stats['tokens']['max']}")
        print(f"  Budget: {stats['tokens']['budget']}")
        print(f"  Over budget: {stats['tokens']['over_budget_count']}")

        print(f"\nFallback rate: {stats['fallback_rate']:.1%}")

    print("\n" + "=" * 60)
    print("ASSERTIONS")
    print("=" * 60)
    for name, passed in report.assertions.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if report.overall_pass else 'FAIL'}")
    print("=" * 60)

    # Save report
    reports_dir = Path(__file__).parent.parent / "episodic" / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "latency_benchmark.json"

    with open(report_path, "w") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "config": report.config,
                "statistics": report.statistics,
                "assertions": report.assertions,
                "overall_pass": report.overall_pass,
            },
            f,
            indent=2,
        )

    print(f"\nReport saved to: {report_path}")

    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
