"""
Replay harness for reactivation probe calibration.

Replays probe decisions on historical conversations and computes
metrics for evaluation and calibration.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result of replaying a single turn."""
    turn_id: str
    user_content: str
    ground_truth: str  # "reactivate:{topic}", "continue", "new_topic"
    probe_decision: str  # "REACTIVATE", "CONTINUE", "DISAMBIGUATE"
    probe_topic: Optional[str]
    correct: bool
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayMetrics:
    """Aggregated metrics from replay."""
    total: int
    correct: int
    accuracy: float

    # By decision type
    true_positives: int  # Correctly reactivated
    false_positives: int  # Reactivated but shouldn't have
    true_negatives: int  # Correctly continued
    false_negatives: int  # Should have reactivated but didn't

    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1: float

    # Additional metrics
    thrash_rate: float  # Rapid switches within W turns
    missed_resumes: int  # Actual topic returns not detected

    # Details
    by_topic: Dict[str, Dict[str, int]] = field(default_factory=dict)


def replay_conversation(
    node_ids: Optional[List[str]] = None,
    limit: int = 100,
    use_ground_truth: bool = True,
    conn: Optional[sqlite3.Connection] = None
) -> List[ReplayResult]:
    """
    Replay probe decisions on historical conversation.

    Args:
        node_ids: Specific node IDs to replay (None = use recent)
        limit: Maximum turns to replay
        use_ground_truth: Use labeled ground truth if available
        conn: Optional database connection

    Returns:
        List of ReplayResult objects
    """
    def _replay(c: sqlite3.Connection) -> List[ReplayResult]:
        results = []

        if node_ids:
            # Use specific node IDs
            query_node_ids = node_ids
        else:
            # Get recent user nodes
            cursor = c.execute("""
                SELECT id FROM nodes
                WHERE role = 'user'
                AND COALESCE(is_meta_query, 0) = 0
                ORDER BY rowid DESC
                LIMIT ?
            """, (limit,))
            query_node_ids = [row[0] for row in cursor.fetchall()]

        # Get stored decisions
        cursor = c.execute("""
            SELECT
                rd.user_node_id,
                rd.decision,
                rd.topic_name,
                rd.confidence,
                rd.candidates_json,
                rd.support_counts_json,
                rd.gates_json,
                rd.best_similarity,
                rd.best_support_count,
                rd.dormancy_turns,
                rl.ground_truth
            FROM reactivation_decisions rd
            LEFT JOIN reactivation_labels rl ON rd.user_node_id = rl.user_node_id
            WHERE rd.user_node_id IN ({})
        """.format(','.join('?' * len(query_node_ids))), query_node_ids)

        decision_map = {}
        for row in cursor.fetchall():
            decision_map[row[0]] = {
                'decision': row[1],
                'topic_name': row[2],
                'confidence': row[3],
                'candidates': json.loads(row[4]) if row[4] else [],
                'support_counts': json.loads(row[5]) if row[5] else {},
                'gates': json.loads(row[6]) if row[6] else {},
                'best_similarity': row[7],
                'best_support_count': row[8],
                'dormancy_turns': row[9],
                'ground_truth': row[10],
            }

        # Get node contents
        cursor = c.execute("""
            SELECT id, content FROM nodes
            WHERE id IN ({})
        """.format(','.join('?' * len(query_node_ids))), query_node_ids)

        content_map = {row[0]: row[1] for row in cursor.fetchall()}

        # Build results
        for node_id in query_node_ids:
            if node_id not in decision_map:
                continue

            decision_info = decision_map[node_id]
            ground_truth = decision_info.get('ground_truth')

            if use_ground_truth and not ground_truth:
                continue  # Skip unlabeled if using ground truth

            probe_decision = decision_info['decision']
            probe_topic = decision_info.get('topic_name')

            # Determine if correct
            correct = _evaluate_correctness(probe_decision, probe_topic, ground_truth)

            results.append(ReplayResult(
                turn_id=node_id,
                user_content=content_map.get(node_id, ''),
                ground_truth=ground_truth or 'unknown',
                probe_decision=probe_decision,
                probe_topic=probe_topic,
                correct=correct,
                features={
                    'confidence': decision_info.get('confidence'),
                    'best_similarity': decision_info.get('best_similarity'),
                    'best_support_count': decision_info.get('best_support_count'),
                    'dormancy_turns': decision_info.get('dormancy_turns'),
                    'candidates': decision_info.get('candidates', []),
                    'support_counts': decision_info.get('support_counts', {}),
                    'gates': decision_info.get('gates', {}),
                }
            ))

        return results

    if conn is not None:
        return _replay(conn)

    with get_connection() as c:
        return _replay(c)


def _evaluate_correctness(
    probe_decision: str,
    probe_topic: Optional[str],
    ground_truth: Optional[str]
) -> bool:
    """Evaluate if a probe decision matches ground truth."""
    if not ground_truth:
        return True  # Can't evaluate without ground truth

    ground_truth_lower = ground_truth.lower()

    if ground_truth_lower == 'continue':
        return probe_decision == 'CONTINUE'

    if ground_truth_lower == 'new_topic':
        # New topic is correct if probe didn't reactivate
        return probe_decision == 'CONTINUE'

    if ground_truth_lower.startswith('reactivate:'):
        expected_topic = ground_truth_lower.split(':', 1)[1].strip()
        if probe_decision != 'REACTIVATE':
            return False
        # Check if topic matches (fuzzy)
        if probe_topic and expected_topic:
            return expected_topic.lower() in probe_topic.lower() or \
                   probe_topic.lower() in expected_topic.lower()
        return probe_decision == 'REACTIVATE'

    return False


def compute_metrics(results: List[ReplayResult]) -> ReplayMetrics:
    """
    Compute evaluation metrics from replay results.

    Args:
        results: List of ReplayResult objects

    Returns:
        ReplayMetrics with aggregated metrics
    """
    total = len(results)
    correct = sum(1 for r in results if r.correct)

    # Classification counts
    true_positives = 0  # Correctly reactivated
    false_positives = 0  # Reactivated but shouldn't have
    true_negatives = 0  # Correctly continued
    false_negatives = 0  # Should have reactivated but didn't

    missed_resumes = 0
    by_topic: Dict[str, Dict[str, int]] = {}

    for r in results:
        ground_truth = r.ground_truth.lower()
        is_reactivate_ground_truth = ground_truth.startswith('reactivate:')
        did_reactivate = r.probe_decision == 'REACTIVATE'

        if is_reactivate_ground_truth:
            if did_reactivate:
                if r.correct:
                    true_positives += 1
                else:
                    # Reactivated to wrong topic - count as FP
                    false_positives += 1
            else:
                false_negatives += 1
                missed_resumes += 1
        else:
            # Ground truth is continue or new_topic
            if did_reactivate:
                false_positives += 1
            else:
                true_negatives += 1

        # Track by topic
        topic_key = r.probe_topic or 'none'
        if topic_key not in by_topic:
            by_topic[topic_key] = {'tp': 0, 'fp': 0, 'fn': 0}
        if is_reactivate_ground_truth and did_reactivate and r.correct:
            by_topic[topic_key]['tp'] += 1
        elif did_reactivate and not r.correct:
            by_topic[topic_key]['fp'] += 1
        elif is_reactivate_ground_truth and not did_reactivate:
            by_topic[topic_key]['fn'] += 1

    # Compute precision/recall/F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute thrash rate (rapid switches)
    thrash_count = _count_thrash_events(results, window=3)
    thrash_rate = thrash_count / max(total, 1)

    return ReplayMetrics(
        total=total,
        correct=correct,
        accuracy=correct / max(total, 1),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        thrash_rate=thrash_rate,
        missed_resumes=missed_resumes,
        by_topic=by_topic,
    )


def _count_thrash_events(results: List[ReplayResult], window: int = 3) -> int:
    """Count rapid topic switches within a window."""
    thrash_count = 0
    recent_topics: List[Optional[str]] = []

    for r in results:
        if r.probe_decision == 'REACTIVATE':
            recent_topics.append(r.probe_topic)
        else:
            recent_topics.append(None)

        # Keep only last `window` items
        if len(recent_topics) > window:
            recent_topics.pop(0)

        # Check for thrash: multiple different reactivations in window
        reactivated_topics = [t for t in recent_topics if t is not None]
        unique_topics = set(reactivated_topics)
        if len(unique_topics) >= 2 and len(reactivated_topics) >= 2:
            thrash_count += 1

    return thrash_count


def export_features(
    results: List[ReplayResult],
    output_path: str
) -> None:
    """
    Export feature data for external analysis.

    Args:
        results: List of ReplayResult objects
        output_path: Path to write JSONL file
    """
    with open(output_path, 'w') as f:
        for r in results:
            record = {
                'turn_id': r.turn_id,
                'ground_truth': r.ground_truth,
                'probe_decision': r.probe_decision,
                'probe_topic': r.probe_topic,
                'correct': r.correct,
                **r.features
            }
            f.write(json.dumps(record) + '\n')

    logger.info(f"Exported {len(results)} feature records to {output_path}")


def get_replay_summary(metrics: ReplayMetrics) -> str:
    """Generate human-readable summary of replay metrics."""
    lines = [
        "Reactivation Replay Summary",
        "=" * 40,
        f"Total turns: {metrics.total}",
        f"Accuracy: {metrics.accuracy:.1%} ({metrics.correct}/{metrics.total})",
        "",
        "Classification:",
        f"  True Positives:  {metrics.true_positives}",
        f"  False Positives: {metrics.false_positives}",
        f"  True Negatives:  {metrics.true_negatives}",
        f"  False Negatives: {metrics.false_negatives}",
        "",
        f"Precision: {metrics.precision:.1%}",
        f"Recall:    {metrics.recall:.1%}",
        f"F1 Score:  {metrics.f1:.1%}",
        "",
        f"Thrash Rate:    {metrics.thrash_rate:.1%}",
        f"Missed Resumes: {metrics.missed_resumes}",
    ]

    if metrics.by_topic:
        lines.append("")
        lines.append("By Topic:")
        for topic, counts in sorted(metrics.by_topic.items()):
            lines.append(f"  {topic}: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}")

    return "\n".join(lines)
