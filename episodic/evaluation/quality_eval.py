"""
Quality evaluation runner for comparing context recovery modes.

Runs each resume moment through multiple modes and collects outputs
for side-by-side comparison and human review.
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .resume_moments import ResumeMoment, load_resume_moments

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class ModeResult:
    """Result from running a moment in a specific mode."""

    mode: str
    prompt_fingerprint: str  # Hash of the assembled prompt
    included_node_ids: List[str]
    contamination_count: int  # Foreign nodes (not in expected topic)
    token_breakdown: Dict[str, int]
    total_tokens: int
    assembly_ms: float
    response: str  # LLM response text
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MomentEvalResult:
    """Result from evaluating a single moment across modes."""

    moment_id: str
    user_query: str
    expected_active_topic: str
    category: str
    mode_results: Dict[str, ModeResult]  # mode -> result
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class QualityEvalReport:
    """Full quality evaluation report."""

    timestamp: str
    config: Dict[str, Any]
    moments_evaluated: int
    modes: List[str]
    results: List[MomentEvalResult]
    summary: Dict[str, Any] = field(default_factory=dict)


def compute_prompt_fingerprint(messages: List[Dict[str, str]]) -> str:
    """Compute a hash of the prompt for reproducibility tracking."""
    content = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def count_contamination(
    included_node_ids: List[str],
    expected_topic_id: str,
    conn: sqlite3.Connection,
) -> int:
    """Count nodes that don't belong to the expected topic."""
    if not included_node_ids or expected_topic_id == "disambiguate":
        return 0

    contamination = 0
    cursor = conn.cursor()

    for node_id in included_node_ids:
        cursor.execute(
            "SELECT topic_start_node_id FROM topic_nodes WHERE node_id = ?",
            (node_id,),
        )
        row = cursor.fetchone()
        if row:
            # Node has a topic assignment
            if row[0] != expected_topic_id:
                contamination += 1
        # Nodes without topic assignment are not counted as contamination

    return contamination


def run_single_mode(
    moment: ResumeMoment,
    mode: str,
    conn: sqlite3.Connection,
    call_llm: bool = False,
    model: str = None,
) -> ModeResult:
    """
    Run context assembly for a moment in a specific mode.

    Args:
        moment: The resume moment to evaluate
        mode: "ancestry", "hybrid", or "topic_local"
        conn: Database connection
        call_llm: Whether to actually call the LLM (expensive)
        model: Model to use for LLM call

    Returns:
        ModeResult with assembled context and optional response
    """
    from episodic.config import config
    from episodic.context_recovery.strategy import ContextRecoveryMode, select_strategy

    # Map mode string to enum
    mode_map = {
        "ancestry": ContextRecoveryMode.ANCESTRY,
        "hybrid": ContextRecoveryMode.HYBRID,
        "topic_local": ContextRecoveryMode.TOPIC_LOCAL,
    }
    mode_enum = mode_map.get(mode, ContextRecoveryMode.HYBRID)

    # Select strategy
    strategy = select_strategy(mode_enum)

    # Run assembly
    start_time = time.perf_counter()
    try:
        result = strategy.assemble(
            user_turn_text=moment.user_query,
            user_node_id=None,
            active_topic_start_node_id=moment.expected_active_topic,
            user_embedding=None,  # Would need embedding for anchor retrieval
            token_budget=config.get("context_token_budget", 4000),
            conn=conn,
        )
    except Exception as e:
        logger.warning(f"Assembly failed for {moment.moment_id} in {mode}: {e}")
        return ModeResult(
            mode=mode,
            prompt_fingerprint="error",
            included_node_ids=[],
            contamination_count=0,
            token_breakdown={},
            total_tokens=0,
            assembly_ms=0,
            response=f"Assembly error: {e}",
            debug={"error": str(e)},
        )

    assembly_ms = (time.perf_counter() - start_time) * 1000

    # Extract debug info
    debug = result.debug
    included_node_ids = debug.get("included_node_ids", [])
    token_breakdown = debug.get("token_breakdown", {})
    total_tokens = token_breakdown.get("total_tokens", 0)

    # Count contamination
    contamination = count_contamination(
        included_node_ids, moment.expected_active_topic, conn
    )

    # Compute prompt fingerprint
    prompt_fingerprint = compute_prompt_fingerprint(result.messages)

    # Optionally call LLM
    response = ""
    if call_llm and result.messages:
        try:
            from episodic.llm import llm_query

            # Add user query to messages
            messages = result.messages + [{"role": "user", "content": moment.user_query}]
            response = llm_query(
                messages=messages,
                model=model or config.get("default_model"),
                temperature=0,  # Deterministic for comparison
            )
        except Exception as e:
            logger.warning(f"LLM call failed for {moment.moment_id} in {mode}: {e}")
            response = f"LLM error: {e}"

    return ModeResult(
        mode=mode,
        prompt_fingerprint=prompt_fingerprint,
        included_node_ids=included_node_ids,
        contamination_count=contamination,
        token_breakdown=token_breakdown,
        total_tokens=total_tokens,
        assembly_ms=assembly_ms,
        response=response,
        debug={
            "working_set_used": debug.get("working_set_used"),
            "summary_included": debug.get("summary_included"),
            "anchors": debug.get("anchors", {}).get("included_count", 0),
            "fallback_reason": debug.get("fallback_reason"),
        },
    )


def run_quality_eval(
    moments: Optional[List[ResumeMoment]] = None,
    modes: List[str] = None,
    call_llm: bool = False,
    model: str = None,
    category_filter: Optional[str] = None,
) -> QualityEvalReport:
    """
    Run quality evaluation on resume moments across modes.

    Args:
        moments: List of moments to evaluate. Loads from fixtures if None.
        modes: Modes to test. Defaults to ["ancestry", "hybrid", "topic_local"]
        call_llm: Whether to call LLM for responses (expensive)
        model: Model to use for LLM calls
        category_filter: Only evaluate moments in this category

    Returns:
        QualityEvalReport with all results
    """
    from episodic.db_connection import get_connection

    if moments is None:
        moments = load_resume_moments(category=category_filter)
    elif category_filter:
        moments = [m for m in moments if m.category == category_filter]

    if modes is None:
        modes = ["ancestry", "hybrid", "topic_local"]

    config_snapshot = {
        "modes": modes,
        "call_llm": call_llm,
        "model": model,
        "category_filter": category_filter,
        "moment_count": len(moments),
    }

    results: List[MomentEvalResult] = []

    with get_connection() as conn:
        for i, moment in enumerate(moments):
            logger.info(f"Evaluating moment {i + 1}/{len(moments)}: {moment.moment_id}")

            mode_results: Dict[str, ModeResult] = {}
            for mode in modes:
                mode_results[mode] = run_single_mode(
                    moment, mode, conn, call_llm=call_llm, model=model
                )

            results.append(
                MomentEvalResult(
                    moment_id=moment.moment_id,
                    user_query=moment.user_query,
                    expected_active_topic=moment.expected_active_topic,
                    category=moment.category,
                    mode_results=mode_results,
                )
            )

    # Compute summary statistics
    summary = compute_eval_summary(results, modes)

    return QualityEvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config_snapshot,
        moments_evaluated=len(results),
        modes=modes,
        results=results,
        summary=summary,
    )


def compute_eval_summary(
    results: List[MomentEvalResult], modes: List[str]
) -> Dict[str, Any]:
    """Compute summary statistics from evaluation results."""
    summary: Dict[str, Any] = {
        "by_mode": {},
        "by_category": {},
    }

    for mode in modes:
        mode_stats = {
            "total_contamination": 0,
            "moments_with_contamination": 0,
            "total_tokens": 0,
            "total_assembly_ms": 0,
        }

        for result in results:
            mr = result.mode_results.get(mode)
            if mr:
                mode_stats["total_contamination"] += mr.contamination_count
                if mr.contamination_count > 0:
                    mode_stats["moments_with_contamination"] += 1
                mode_stats["total_tokens"] += mr.total_tokens
                mode_stats["total_assembly_ms"] += mr.assembly_ms

        n = len(results)
        summary["by_mode"][mode] = {
            "contamination_rate": mode_stats["moments_with_contamination"] / n if n else 0,
            "avg_tokens": mode_stats["total_tokens"] / n if n else 0,
            "avg_assembly_ms": mode_stats["total_assembly_ms"] / n if n else 0,
        }

    # By category
    categories: Dict[str, List[MomentEvalResult]] = {}
    for result in results:
        cat = result.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    for cat, cat_results in categories.items():
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "modes": {},
        }
        for mode in modes:
            contamination_sum = sum(
                r.mode_results.get(mode, ModeResult("", "", [], 0, {}, 0, 0, "")).contamination_count
                for r in cat_results
            )
            summary["by_category"][cat]["modes"][mode] = {
                "total_contamination": contamination_sum,
            }

    return summary


def export_for_human_review(
    report: QualityEvalReport,
    output_path: Optional[Path] = None,
    format: str = "markdown",
) -> Path:
    """
    Export evaluation results for human review.

    Args:
        report: The quality evaluation report
        output_path: Where to save. Defaults to reports/quality_eval_review.md
        format: "markdown" or "csv"

    Returns:
        Path to the exported file
    """
    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ext = "md" if format == "markdown" else "csv"
        output_path = REPORTS_DIR / f"quality_eval_review.{ext}"

    if format == "markdown":
        content = _format_markdown_review(report)
    else:
        content = _format_csv_review(report)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def _format_markdown_review(report: QualityEvalReport) -> str:
    """Format report as markdown for human review."""
    lines = [
        "# Quality Evaluation Review",
        "",
        f"Generated: {report.timestamp}",
        f"Moments evaluated: {report.moments_evaluated}",
        f"Modes: {', '.join(report.modes)}",
        "",
        "---",
        "",
    ]

    for result in report.results:
        lines.append(f"## Moment: {result.moment_id}")
        lines.append(f"**Query:** {result.user_query}")
        lines.append(f"**Expected topic:** {result.expected_active_topic}")
        lines.append(f"**Category:** {result.category}")
        lines.append("")

        for mode, mr in result.mode_results.items():
            lines.append(f"### {mode}")
            if mr.response:
                # Truncate long responses
                response_preview = mr.response[:500] + "..." if len(mr.response) > 500 else mr.response
                lines.append(f"```")
                lines.append(response_preview)
                lines.append(f"```")
            lines.append(f"- Contamination: {mr.contamination_count} foreign nodes")
            lines.append(f"- Tokens: {mr.total_tokens}")
            lines.append(f"- Assembly: {mr.assembly_ms:.2f}ms")
            lines.append(f"- Nodes included: {len(mr.included_node_ids)}")
            if mr.debug.get("fallback_reason"):
                lines.append(f"- Fallback: {mr.debug['fallback_reason']}")
            lines.append("")

        lines.append("### Scoring (fill in):")
        lines.append(f"- [ ] Stays on correct topic ({result.expected_active_topic})")
        lines.append("- [ ] Uses correct prior information")
        lines.append("- [ ] Does not reference unrelated topics")
        lines.append("- [ ] Response quality is good")
        lines.append("")
        lines.append("**Notes:**")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary section
    lines.append("# Summary Statistics")
    lines.append("")
    lines.append("## By Mode")
    lines.append("")
    lines.append("| Mode | Contamination Rate | Avg Tokens | Avg Assembly (ms) |")
    lines.append("|------|-------------------|------------|-------------------|")
    for mode, stats in report.summary.get("by_mode", {}).items():
        lines.append(
            f"| {mode} | {stats['contamination_rate']:.1%} | "
            f"{stats['avg_tokens']:.0f} | {stats['avg_assembly_ms']:.2f} |"
        )
    lines.append("")

    lines.append("## By Category")
    lines.append("")
    for cat, cat_stats in report.summary.get("by_category", {}).items():
        lines.append(f"### {cat} ({cat_stats['count']} moments)")
        lines.append("")
        lines.append("| Mode | Total Contamination |")
        lines.append("|------|---------------------|")
        for mode, mode_stats in cat_stats.get("modes", {}).items():
            lines.append(f"| {mode} | {mode_stats['total_contamination']} |")
        lines.append("")

    return "\n".join(lines)


def _format_csv_review(report: QualityEvalReport) -> str:
    """Format report as CSV for spreadsheet review."""
    lines = [
        "moment_id,category,expected_topic,mode,contamination,tokens,assembly_ms,response_preview"
    ]

    for result in report.results:
        for mode, mr in result.mode_results.items():
            response_preview = mr.response[:100].replace(",", ";").replace("\n", " ") if mr.response else ""
            lines.append(
                f"{result.moment_id},{result.category},{result.expected_active_topic},"
                f"{mode},{mr.contamination_count},{mr.total_tokens},{mr.assembly_ms:.2f},"
                f"\"{response_preview}\""
            )

    return "\n".join(lines)


def save_report(report: QualityEvalReport, path: Optional[Path] = None) -> Path:
    """Save the full report as JSON."""
    if path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "quality_eval_report.json"

    # Convert to serializable format
    def serialize_result(r: MomentEvalResult) -> Dict:
        return {
            "moment_id": r.moment_id,
            "user_query": r.user_query,
            "expected_active_topic": r.expected_active_topic,
            "category": r.category,
            "timestamp": r.timestamp,
            "mode_results": {
                mode: asdict(mr) for mode, mr in r.mode_results.items()
            },
        }

    with open(path, "w") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "config": report.config,
                "moments_evaluated": report.moments_evaluated,
                "modes": report.modes,
                "summary": report.summary,
                "results": [serialize_result(r) for r in report.results],
            },
            f,
            indent=2,
        )

    return path
