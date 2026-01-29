"""
Metrics computation for LLM verifier experiment.

Computes:
- Precision@N and Recall@N for N=5,10
- Hard-negative false accept rate
- Format failure rate (quote/offset validation failures)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from verifier import VerifierStats, Relation

EXPERIMENT_DIR = Path(__file__).parent
QUERY_CASES_PATH = EXPERIMENT_DIR / "query_cases.json"


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    description: str

    # Precision/Recall
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float

    # Coverage
    has_relevant_in_top_5: bool
    has_relevant_in_top_10: bool

    # Hard negatives
    hard_negative_count: int
    hard_negative_accepted: int
    hard_negative_false_accept_rate: float

    # Format failures
    format_failure_count: int
    format_failure_rate: float

    # Ambiguity tracking
    ambiguous_quote_count: int = 0

    # Details
    accepted_ids: list[int] = field(default_factory=list)
    relevant_ids: list[int] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries."""
    num_queries: int

    # Precision/Recall averages
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_recall_at_5: float
    mean_recall_at_10: float

    # Coverage
    coverage_at_5: float  # fraction of queries with at least 1 relevant in top-5
    coverage_at_10: float

    # Hard negatives
    total_hard_negatives: int
    total_hard_negative_accepts: int
    hard_negative_false_accept_rate: float

    # Format failures
    total_format_failures: int
    total_judgments: int
    format_failure_rate: float

    # Call stats
    total_llm_calls: int
    total_cache_hits: int
    mean_calls_per_query: float

    # Ambiguity tracking
    total_ambiguous_quotes: int = 0
    candidates_with_ambiguous_quotes: int = 0


def compute_query_metrics(stats: VerifierStats, case: dict) -> QueryMetrics:
    """Compute metrics for a single query given verification stats and gold labels."""
    gold_relevant = case.get("gold_relevant", {})
    hard_negatives = set(case.get("hard_negatives", []))

    # Get accepted statement IDs (those that passed verification)
    accepted_ids = [r.statement_id for r in stats.results
                    if r.relation != Relation.UNRELATED and r.quote_check_passed]

    # Get all relevant statement IDs from gold
    relevant_ids = [int(sid) for sid, label in gold_relevant.items() if label == 1]

    # Top-N results (accepted, sorted by original candidate order which was by sim)
    candidate_order = {cid: i for i, cid in enumerate(case["candidates"])}
    accepted_sorted = sorted(accepted_ids, key=lambda x: candidate_order.get(x, 999))

    top_5 = accepted_sorted[:5]
    top_10 = accepted_sorted[:10]

    # Precision@N = (relevant in top-N) / N
    relevant_in_top_5 = len([x for x in top_5 if str(x) in gold_relevant and gold_relevant[str(x)] == 1])
    relevant_in_top_10 = len([x for x in top_10 if str(x) in gold_relevant and gold_relevant[str(x)] == 1])

    precision_at_5 = relevant_in_top_5 / 5 if len(top_5) > 0 else 0.0
    precision_at_10 = relevant_in_top_10 / 10 if len(top_10) > 0 else 0.0

    # Actually, precision should be based on how many we returned
    # If we returned fewer than N, precision = relevant / returned
    precision_at_5 = relevant_in_top_5 / len(top_5) if len(top_5) > 0 else 0.0
    precision_at_10 = relevant_in_top_10 / len(top_10) if len(top_10) > 0 else 0.0

    # Recall@N = (relevant in top-N) / total_relevant
    total_relevant = len(relevant_ids)
    recall_at_5 = relevant_in_top_5 / total_relevant if total_relevant > 0 else 0.0
    recall_at_10 = relevant_in_top_10 / total_relevant if total_relevant > 0 else 0.0

    # Coverage
    has_relevant_in_top_5 = relevant_in_top_5 > 0
    has_relevant_in_top_10 = relevant_in_top_10 > 0

    # Hard negatives
    hard_negative_count = len(hard_negatives)
    hard_negative_accepted = len([x for x in accepted_ids if x in hard_negatives])
    hard_negative_fpr = hard_negative_accepted / hard_negative_count if hard_negative_count > 0 else 0.0

    # Format failures (quote validation failures)
    format_failures = len([r for r in stats.results
                           if r.relation != Relation.UNRELATED and not r.quote_check_passed])
    total_non_unrelated = len([r for r in stats.results if r.relation != Relation.UNRELATED])
    format_failure_rate = format_failures / total_non_unrelated if total_non_unrelated > 0 else 0.0

    # Ambiguous quotes (for monitoring)
    ambiguous_count = sum(len(getattr(r, 'ambiguous_quotes', [])) for r in stats.results)

    return QueryMetrics(
        query=stats.query,
        description=case.get("description", ""),
        precision_at_5=precision_at_5,
        precision_at_10=precision_at_10,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        has_relevant_in_top_5=has_relevant_in_top_5,
        has_relevant_in_top_10=has_relevant_in_top_10,
        hard_negative_count=hard_negative_count,
        hard_negative_accepted=hard_negative_accepted,
        hard_negative_false_accept_rate=hard_negative_fpr,
        format_failure_count=format_failures,
        format_failure_rate=format_failure_rate,
        ambiguous_quote_count=ambiguous_count,
        accepted_ids=accepted_ids,
        relevant_ids=relevant_ids,
    )


def compute_aggregate_metrics(
    all_stats: list[VerifierStats],
    cases: list[dict]
) -> tuple[list[QueryMetrics], AggregateMetrics]:
    """Compute per-query and aggregate metrics."""
    query_to_case = {c["query"]: c for c in cases}

    query_metrics = []
    for stats in all_stats:
        case = query_to_case.get(stats.query, {})
        qm = compute_query_metrics(stats, case)
        query_metrics.append(qm)

    # Aggregate
    n = len(query_metrics)
    if n == 0:
        return [], AggregateMetrics(
            num_queries=0,
            mean_precision_at_5=0, mean_precision_at_10=0,
            mean_recall_at_5=0, mean_recall_at_10=0,
            coverage_at_5=0, coverage_at_10=0,
            total_hard_negatives=0, total_hard_negative_accepts=0,
            hard_negative_false_accept_rate=0,
            total_format_failures=0, total_judgments=0, format_failure_rate=0,
            total_llm_calls=0, total_cache_hits=0, mean_calls_per_query=0,
        )

    agg = AggregateMetrics(
        num_queries=n,
        mean_precision_at_5=sum(qm.precision_at_5 for qm in query_metrics) / n,
        mean_precision_at_10=sum(qm.precision_at_10 for qm in query_metrics) / n,
        mean_recall_at_5=sum(qm.recall_at_5 for qm in query_metrics) / n,
        mean_recall_at_10=sum(qm.recall_at_10 for qm in query_metrics) / n,
        coverage_at_5=sum(1 for qm in query_metrics if qm.has_relevant_in_top_5) / n,
        coverage_at_10=sum(1 for qm in query_metrics if qm.has_relevant_in_top_10) / n,
        total_hard_negatives=sum(qm.hard_negative_count for qm in query_metrics),
        total_hard_negative_accepts=sum(qm.hard_negative_accepted for qm in query_metrics),
        hard_negative_false_accept_rate=(
            sum(qm.hard_negative_accepted for qm in query_metrics) /
            sum(qm.hard_negative_count for qm in query_metrics)
            if sum(qm.hard_negative_count for qm in query_metrics) > 0 else 0.0
        ),
        total_format_failures=sum(qm.format_failure_count for qm in query_metrics),
        total_judgments=sum(len(s.results) for s in all_stats),
        format_failure_rate=(
            sum(qm.format_failure_count for qm in query_metrics) /
            sum(len(s.results) for s in all_stats)
            if sum(len(s.results) for s in all_stats) > 0 else 0.0
        ),
        total_llm_calls=sum(s.llm_calls for s in all_stats),
        total_cache_hits=sum(s.cache_hits for s in all_stats),
        mean_calls_per_query=sum(s.llm_calls for s in all_stats) / n,
        total_ambiguous_quotes=sum(qm.ambiguous_quote_count for qm in query_metrics),
        candidates_with_ambiguous_quotes=sum(
            1 for s in all_stats for r in s.results
            if len(getattr(r, 'ambiguous_quotes', [])) > 0
        ),
    )

    return query_metrics, agg


def format_metrics_report(
    query_metrics: list[QueryMetrics],
    agg: AggregateMetrics
) -> str:
    """Format a human-readable metrics report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM VERIFIER EXPERIMENT RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Aggregate metrics
    lines.append("AGGREGATE METRICS")
    lines.append("-" * 40)
    lines.append(f"Queries evaluated: {agg.num_queries}")
    lines.append("")
    lines.append(f"Precision@5:  {agg.mean_precision_at_5:.1%}")
    lines.append(f"Precision@10: {agg.mean_precision_at_10:.1%}")
    lines.append(f"Recall@5:     {agg.mean_recall_at_5:.1%}")
    lines.append(f"Recall@10:    {agg.mean_recall_at_10:.1%}")
    lines.append("")
    lines.append(f"Coverage@5:   {agg.coverage_at_5:.1%} (queries with ≥1 relevant in top-5)")
    lines.append(f"Coverage@10:  {agg.coverage_at_10:.1%}")
    lines.append("")

    # Hard negatives
    lines.append("HARD NEGATIVE ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Total hard negatives:     {agg.total_hard_negatives}")
    lines.append(f"Incorrectly accepted:     {agg.total_hard_negative_accepts}")
    lines.append(f"False accept rate:        {agg.hard_negative_false_accept_rate:.1%}")
    lines.append("")

    # Format failures
    lines.append("FORMAT/VALIDATION FAILURES")
    lines.append("-" * 40)
    lines.append(f"Total judgments:          {agg.total_judgments}")
    lines.append(f"Quote validation failures:{agg.total_format_failures}")
    lines.append(f"Failure rate:             {agg.format_failure_rate:.1%}")
    lines.append("")

    # Call stats
    lines.append("LLM CALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total LLM calls:          {agg.total_llm_calls}")
    lines.append(f"Cache hits:               {agg.total_cache_hits}")
    lines.append(f"Mean calls per query:     {agg.mean_calls_per_query:.2f}")
    lines.append("")

    # Ambiguity monitoring
    if agg.total_ambiguous_quotes > 0:
        lines.append("QUOTE AMBIGUITY (monitoring)")
        lines.append("-" * 40)
        lines.append(f"Ambiguous quotes:         {agg.total_ambiguous_quotes}")
        lines.append(f"Candidates affected:      {agg.candidates_with_ambiguous_quotes}")
        lines.append("")

    # Per-query breakdown for hard negative cases
    hard_neg_queries = [qm for qm in query_metrics if qm.hard_negative_count > 0]
    if hard_neg_queries:
        lines.append("HARD NEGATIVE CASES (detail)")
        lines.append("-" * 40)
        for qm in hard_neg_queries:
            status = "✓ PASS" if qm.hard_negative_accepted == 0 else f"✗ FAIL ({qm.hard_negative_accepted}/{qm.hard_negative_count})"
            lines.append(f"  {qm.query:20s} {status}")
            lines.append(f"    {qm.description}")
        lines.append("")

    # Per-query breakdown for abstract/semantic cases
    abstract_queries = [qm for qm in query_metrics if qm.hard_negative_count == 0]
    if abstract_queries:
        lines.append("SEMANTIC/ABSTRACT CASES")
        lines.append("-" * 40)
        for qm in abstract_queries:
            lines.append(f"  {qm.query:20s} P@5={qm.precision_at_5:.0%} R@5={qm.recall_at_5:.0%}")
            lines.append(f"    accepted: {len(qm.accepted_ids)}, relevant: {len(qm.relevant_ids)}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with dummy data
    print("Metrics module loaded successfully")
