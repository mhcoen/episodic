"""KG ablation evaluation harness.

Measures whether KG context injection improves LLM answer quality.
Three conditions:
  A (baseline): no KG context — LLM sees bare prompt only
  B (KG only):  KG system message + bare prompt, kg_max_derived=0
  C (KG+closure): KG system message + bare prompt, kg_max_derived=3

Architecture:
  Preload phase: create eval DB, insert setup_context as nodes, run KG
  extraction (one-time LLM cost). Eval phase: messages contain ONLY the
  bare prompt. setup_context is NOT in messages for ANY condition.
"""

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from episodic.config import config
from episodic.kg.context_source import get_kg_context
from episodic.kg.schema import ensure_kg_schema
from episodic.llm import _execute_llm_query


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    prompt_id: str
    condition: str  # "A", "B", "C"
    category: str
    # Correctness
    required_facts_found: int = 0
    required_facts_total: int = 0
    expected_contains_found: int = 0
    expected_contains_total: int = 0
    factual_score: float = 0.0
    # Cost
    kg_block_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # Latency
    context_build_ms: float = 0.0
    llm_response_ms: float = 0.0
    # Raw
    llm_response: str = ""
    kg_context_text: str = ""


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)
    by_category: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)
    preload_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Condition configs
# ---------------------------------------------------------------------------

CONDITIONS = {
    "A": {"kg_context": False, "kg_max_derived": 0},
    "B": {"kg_context": True, "kg_max_derived": 0},
    "C": {"kg_context": True, "kg_max_derived": 3},
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response(response_text: str, answer_key: dict) -> dict:
    """Score LLM response against answer key. Deterministic substring match."""
    response_lower = response_text.lower()

    required = answer_key.get("required_facts", [])
    expected = answer_key.get("expected_answer_contains", [])

    facts_found = sum(1 for f in required if f.lower() in response_lower)
    contains_found = sum(1 for e in expected if e.lower() in response_lower)

    total = len(required) + len(expected)
    score = (facts_found + contains_found) / total if total > 0 else 0.0

    return {
        "required_facts_found": facts_found,
        "required_facts_total": len(required),
        "expected_contains_found": contains_found,
        "expected_contains_total": len(expected),
        "factual_score": score,
    }


# ---------------------------------------------------------------------------
# Message assembly (no setup_context — bare prompt only)
# ---------------------------------------------------------------------------

def build_messages(
    prompt: str,
    kg_context_text: str | None = None,
) -> list[dict]:
    """Build the messages list for the LLM call.

    Layout:
      [system: KG context (if any)]
      [user: prompt]
    No setup_context — the LLM sees ONLY the prompt + optional KG block.
    """
    messages = []
    if kg_context_text:
        messages.append({"role": "system", "content": kg_context_text})
    messages.append({"role": "user", "content": prompt})
    return messages


# ---------------------------------------------------------------------------
# Eval DB preload
# ---------------------------------------------------------------------------

def _create_eval_db(db_path: str) -> sqlite3.Connection:
    """Create a fresh eval DB with nodes table + KG schema."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            short_id TEXT UNIQUE,
            role TEXT,
            provider TEXT,
            model TEXT,
            is_meta_query INTEGER DEFAULT 0,
            FOREIGN KEY(parent_id) REFERENCES nodes(id)
        )
    """)
    ensure_kg_schema(conn)
    return conn


def preload_kg_from_dataset(
    dataset: list[dict],
    conn: sqlite3.Connection,
    progress_callback=None,
    db_path: str | None = None,
) -> dict:
    """Extract KG triples from all setup_context strings. One-time cost.

    Inserts user messages from setup_context as nodes, then runs the full
    extraction pipeline (extract → validate → apply).

    Returns: dict with preload stats.
    """
    from episodic.kg.batch import run_batch

    # Insert setup_context user messages as nodes
    node_count = 0
    for item in dataset:
        for i, ctx in enumerate(item.get("setup_context", [])):
            if i % 2 != 0:
                continue  # Only extract from user messages (even indices)
            node_count += 1
            node_id = f"eval-{item['id']}-{i}"
            conn.execute(
                "INSERT INTO nodes (id, content, role, is_meta_query) "
                "VALUES (?, ?, 'user', 0)",
                (node_id, ctx),
            )
    conn.commit()

    if node_count == 0:
        return {"nodes_inserted": 0, "extraction": {}}

    # Run batch extraction on all inserted nodes
    # db_path enables extraction threads to open connections to the eval DB
    result = run_batch(
        lookback=0, conn=conn, progress_callback=progress_callback,
        db_path=db_path,
    )

    return {
        "nodes_inserted": node_count,
        "extraction": {
            "nodes_processed": result.get("nodes_processed", 0),
            "patches_applied": result.get("patches_applied", 0),
            "patches_rejected": result.get("patches_rejected", 0),
            "nodes_qa_filtered": result.get("nodes_qa_filtered", 0),
        },
    }


# ---------------------------------------------------------------------------
# Single prompt evaluation
# ---------------------------------------------------------------------------

def evaluate_prompt(
    item: dict,
    condition: str,
    conn: sqlite3.Connection,
    model: str,
    dry_run: bool = False,
) -> EvalResult:
    """Evaluate a single prompt under a single condition."""
    prompt_id = item["id"]
    prompt_text = item["prompt"]
    answer_key = item["answer_key"]
    category = answer_key["category"]

    cond_cfg = CONDITIONS[condition]
    kg_text = ""
    kg_tokens = 0

    # Build KG context (conditions B and C only)
    ctx_start = time.monotonic()
    if cond_cfg["kg_context"]:
        old_derived = config.get("kg_max_derived", 3)
        config.set("kg_max_derived", cond_cfg["kg_max_derived"])
        try:
            result = get_kg_context(prompt_text, conn)
            if result:
                kg_text = result.text
                kg_tokens = result.budget_used
        finally:
            config.set("kg_max_derived", old_derived)
    ctx_ms = (time.monotonic() - ctx_start) * 1000

    # Build messages — NO setup_context, bare prompt only
    messages = build_messages(prompt_text, kg_text or None)

    if dry_run:
        return EvalResult(
            prompt_id=prompt_id,
            condition=condition,
            category=category,
            required_facts_total=len(answer_key.get("required_facts", [])),
            expected_contains_total=len(answer_key.get("expected_answer_contains", [])),
            kg_block_tokens=kg_tokens,
            context_build_ms=ctx_ms,
            kg_context_text=kg_text,
        )

    # Call LLM
    llm_start = time.monotonic()
    response_text, cost_info = _execute_llm_query(
        messages, model, temperature=0.3, max_tokens=512
    )
    llm_ms = (time.monotonic() - llm_start) * 1000

    # Score
    scores = score_response(response_text, answer_key)

    return EvalResult(
        prompt_id=prompt_id,
        condition=condition,
        category=category,
        **scores,
        kg_block_tokens=kg_tokens,
        total_prompt_tokens=cost_info.get("input_tokens", 0),
        total_completion_tokens=cost_info.get("output_tokens", 0),
        context_build_ms=ctx_ms,
        llm_response_ms=llm_ms,
        llm_response=response_text,
        kg_context_text=kg_text,
    )


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------

def run_ablation(
    dataset_path: str | None = None,
    db_path: str | None = None,
    model: str | None = None,
    conditions: list[str] | None = None,
    dry_run: bool = False,
    skip_preload: bool = False,
) -> EvalSummary:
    """Run the ablation evaluation.

    Args:
        dataset_path: Path to eval_dataset.json. Defaults to bundled dataset.
        db_path: Path to eval DB. Defaults to ~/.episodic/eval_kg.db.
        model: Model for eval LLM calls. Defaults to config model.
        conditions: Which conditions to run. Defaults to ["A", "B", "C"].
        dry_run: If True, skip eval LLM calls. Show KG context per condition.
        skip_preload: If True, skip preload (DB already populated).

    Returns:
        EvalSummary with all results.
    """
    if dataset_path is None:
        dataset_path = str(Path(__file__).parent / "eval_dataset.json")
    if db_path is None:
        db_path = os.path.expanduser("~/.episodic/eval_kg.db")
    if model is None:
        model = config.get("model", "gpt-4o-mini")
    if conditions is None:
        conditions = ["A", "B", "C"]

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    summary = EvalSummary()

    # Phase 1: Preload KG from setup_context
    if skip_preload and os.path.exists(db_path):
        print(f"  Reusing existing eval DB: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=ON")
    else:
        print("  Creating eval DB and running KG extraction...")
        conn = _create_eval_db(db_path)

        def progress(nid, idx, tot):
            print(f"    Extracting node {nid} ({idx}/{tot})...", flush=True)

        preload_stats = preload_kg_from_dataset(
            dataset, conn, progress, db_path=db_path,
        )
        summary.preload_stats = preload_stats
        ps = preload_stats.get("extraction", {})
        print(f"  Preload complete: {preload_stats['nodes_inserted']} nodes, "
              f"{ps.get('patches_applied', 0)} applied, "
              f"{ps.get('patches_rejected', 0)} rejected, "
              f"{ps.get('nodes_qa_filtered', 0)} QA-filtered")

    # Phase 2: Evaluate
    total = len(dataset) * len(conditions)
    done = 0

    try:
        for item in dataset:
            for cond in conditions:
                done += 1
                pid = item["id"]
                if not dry_run:
                    print(f"  [{done}/{total}] {pid} cond={cond} ...",
                          end="", flush=True)

                result = evaluate_prompt(item, cond, conn, model, dry_run)
                summary.results.append(result)

                if not dry_run:
                    print(f" score={result.factual_score:.2f}"
                          f" kg={result.kg_block_tokens}tok"
                          f" llm={result.llm_response_ms:.0f}ms")
                elif dry_run:
                    print(f"  {pid} cond={cond}: "
                          f"kg_tokens={result.kg_block_tokens}")
                    if result.kg_context_text:
                        for line in result.kg_context_text.split("\n")[:5]:
                            print(f"    {line}")
    finally:
        conn.close()

    _compute_summary(summary, conditions)
    return summary


def _compute_summary(summary: EvalSummary, conditions: list[str]):
    """Compute per-category and overall averages."""
    from collections import defaultdict

    cat_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cat_kg_tokens: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cat_latency: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in summary.results:
        cat_scores[r.category][r.condition].append(r.factual_score)
        cat_kg_tokens[r.category][r.condition].append(r.kg_block_tokens)
        cat_latency[r.category][r.condition].append(r.llm_response_ms)

    by_cat = {}
    for cat in sorted(cat_scores.keys()):
        by_cat[cat] = {}
        for cond in conditions:
            scores = cat_scores[cat].get(cond, [])
            tokens = cat_kg_tokens[cat].get(cond, [])
            lats = cat_latency[cat].get(cond, [])
            by_cat[cat][cond] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "avg_kg_tokens": sum(tokens) / len(tokens) if tokens else 0,
                "avg_latency_ms": sum(lats) / len(lats) if lats else 0,
                "count": len(scores),
            }
    summary.by_category = by_cat

    overall = {}
    for cond in conditions:
        cond_results = [r for r in summary.results if r.condition == cond]
        scores = [r.factual_score for r in cond_results]
        tokens = [r.kg_block_tokens for r in cond_results]
        lats = [r.llm_response_ms for r in cond_results]
        overall[cond] = {
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_kg_tokens": sum(tokens) / len(tokens) if tokens else 0,
            "avg_latency_ms": sum(lats) / len(lats) if lats else 0,
            "count": len(scores),
        }
    summary.overall = overall


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

COND_LABELS = {"A": "A (base)", "B": "B (KG)", "C": "C (KG+cl)"}


def format_summary_table(summary: EvalSummary, conditions: list[str]) -> str:
    """Format the summary as a text table."""
    lines = []
    lines.append("KG Ablation Results")
    lines.append("=" * 60)

    hdr = f"{'Category':<22}"
    for c in conditions:
        hdr += f"| {COND_LABELS.get(c, c):>10} "
    lines.append(hdr)
    lines.append("-" * 22 + ("+" + "-" * 12) * len(conditions))

    for cat in sorted(summary.by_category.keys()):
        row = f"{cat:<22}"
        for c in conditions:
            stats = summary.by_category[cat].get(c, {})
            score = stats.get("avg_score", 0)
            row += f"|  {score:>8.2f}  "
        lines.append(row)

    lines.append("-" * 22 + ("+" + "-" * 12) * len(conditions))

    row = f"{'OVERALL':<22}"
    for c in conditions:
        stats = summary.overall.get(c, {})
        score = stats.get("avg_score", 0)
        row += f"|  {score:>8.2f}  "
    lines.append(row)

    lines.append("-" * 22 + ("+" + "-" * 12) * len(conditions))

    row = f"{'Avg KG tokens':<22}"
    for c in conditions:
        stats = summary.overall.get(c, {})
        tok = stats.get("avg_kg_tokens", 0)
        row += f"|  {tok:>8.0f}  "
    lines.append(row)

    row = f"{'Avg latency (ms)':<22}"
    for c in conditions:
        stats = summary.overall.get(c, {})
        lat = stats.get("avg_latency_ms", 0)
        row += f"|  {lat:>8.0f}  "
    lines.append(row)

    lines.append("=" * 60)
    return "\n".join(lines)


def save_results(summary: EvalSummary, output_path: str | None = None):
    """Save full results to JSON."""
    if output_path is None:
        output_path = str(Path(__file__).parent / "eval_results.json")

    data = {
        "results": [asdict(r) for r in summary.results],
        "by_category": summary.by_category,
        "overall": summary.overall,
        "preload_stats": summary.preload_stats,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path
