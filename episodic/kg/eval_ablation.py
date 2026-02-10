"""KG ablation evaluation harness.

Three conditions: A (no KG), B (KG, max_derived=0), C (KG+closure, max_derived=3).
Preload: insert setup_context as nodes + extract. Eval: bare prompt only.
"""

import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from episodic.config import config
from episodic.kg.context_source import get_kg_context
from episodic.kg.schema import ensure_kg_schema
from episodic.llm import _execute_llm_query


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
    # Closure tracking
    derived_edges_count: int = 0
    closure_expected: bool = False
    closure_rule: str = ""
    # Derived relevance (condition C only, 0 for A/B)
    derived_max_relevance: float = 0.0
    derived_mean_relevance: float = 0.0
    derived_top_fact: str = ""
    derived_top_relevance: float = 0.0
    # Oracle (closure_expected items only)
    oracle_hit: bool = False
    oracle_fact: str = ""
    # Raw
    llm_response: str = ""
    kg_context_text: str = ""


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)
    by_category: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)
    preload_stats: dict = field(default_factory=dict)
    closure_checks: dict = field(default_factory=dict)


CONDITIONS = {
    "A": {"kg_context": False, "kg_max_derived": 0},
    "B": {"kg_context": True, "kg_max_derived": 0},
    "C": {"kg_context": True, "kg_max_derived": 3},
}


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


_STOPWORDS = frozenset(
    "the a an is are my i do does have has what where how much "
    "can it on at in of to for with".split()
)

_WORD_RE = re.compile(r'[a-z0-9]+')


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punct, remove stopwords."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compute_derived_relevance(
    prompt: str, derived_facts: list,
) -> dict:
    """Compute Jaccard relevance of derived facts against the prompt."""
    prompt_tokens = _tokenize(prompt)
    if not derived_facts or not prompt_tokens:
        return {"max_rel": 0.0, "mean_rel": 0.0, "top_fact": "", "top_rel": 0.0}

    scores = []
    for d in derived_facts:
        fact_text = f"{d.subj_name} {d.predicate} {d.obj_name}"
        fact_tokens = _tokenize(fact_text)
        score = _jaccard(prompt_tokens, fact_tokens)
        scores.append((score, fact_text))

    scores.sort(key=lambda x: x[0], reverse=True)
    all_scores = [s for s, _ in scores]
    return {
        "max_rel": max(all_scores),
        "mean_rel": sum(all_scores) / len(all_scores),
        "top_fact": scores[0][1],
        "top_rel": scores[0][0],
    }


def compute_oracle_hit(
    derived_facts: list, closure_rule: str, required_facts: list[str],
) -> dict:
    """Check if any derived fact matches the closure_rule AND mentions required_facts."""
    for d in derived_facts:
        if d.rule != closure_rule:
            continue
        fact_text = f"{d.subj_name} {d.predicate} {d.obj_name}"
        fact_lower = fact_text.lower()
        if any(rf.lower() in fact_lower for rf in required_facts):
            return {"oracle_hit": True, "oracle_fact": fact_text}
    return {"oracle_hit": False, "oracle_fact": ""}


def build_messages(
    prompt: str, kg_context_text: str | None = None,
) -> list[dict]:
    """Build messages: optional KG system msg + user prompt. No setup_context."""
    messages = []
    if kg_context_text:
        messages.append({"role": "system", "content": kg_context_text})
    messages.append({"role": "user", "content": prompt})
    return messages


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
    dataset: list[dict], conn: sqlite3.Connection,
    progress_callback=None, db_path: str | None = None,
) -> dict:
    """Insert setup_context as nodes, run extraction pipeline. Returns stats."""
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
    closure_expected = item.get("closure_expected", False)
    closure_rule = item.get("closure_rule", "")
    required_facts = answer_key.get("required_facts", [])
    kg_text = ""
    kg_tokens = 0
    derived_count = 0
    derived_list = []

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
                derived_count = result.derived_count
                derived_list = result.derived
        finally:
            config.set("kg_max_derived", old_derived)
    ctx_ms = (time.monotonic() - ctx_start) * 1000

    # Derived-fact relevance + oracle (C condition only)
    rel = compute_derived_relevance(prompt_text, derived_list)
    oracle = compute_oracle_hit(
        derived_list, closure_rule, required_facts,
    ) if closure_expected else {"oracle_hit": False, "oracle_fact": ""}

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
            derived_edges_count=derived_count,
            closure_expected=closure_expected,
            closure_rule=closure_rule,
            derived_max_relevance=rel["max_rel"],
            derived_mean_relevance=rel["mean_rel"],
            derived_top_fact=rel["top_fact"],
            derived_top_relevance=rel["top_rel"],
            oracle_hit=oracle["oracle_hit"],
            oracle_fact=oracle["oracle_fact"],
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
        derived_edges_count=derived_count,
        closure_expected=closure_expected,
        closure_rule=closure_rule,
        derived_max_relevance=rel["max_rel"],
        derived_mean_relevance=rel["mean_rel"],
        derived_top_fact=rel["top_fact"],
        derived_top_relevance=rel["top_rel"],
        oracle_hit=oracle["oracle_hit"],
        oracle_fact=oracle["oracle_fact"],
        llm_response=response_text,
        kg_context_text=kg_text,
    )


def run_ablation(
    dataset_path: str | None = None, db_path: str | None = None,
    model: str | None = None, conditions: list[str] | None = None,
    dry_run: bool = False, skip_preload: bool = False,
    filter_closure: bool = False,
) -> EvalSummary:
    """Run ablation evaluation across conditions A/B/C."""
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

    if filter_closure:
        dataset = [item for item in dataset if item.get("closure_expected", False)]
        print(f"  Filtered to {len(dataset)} closure_expected items")

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
                    extra = ""
                    if result.closure_expected:
                        extra += f" derived={result.derived_edges_count}"
                        if result.derived_max_relevance > 0:
                            extra += f" rel={result.derived_max_relevance:.2f}"
                        if result.oracle_hit:
                            extra += " ORACLE_HIT"
                    print(f"  {pid} cond={cond}: "
                          f"kg_tokens={result.kg_block_tokens}"
                          f"{extra}")
                    if result.kg_context_text:
                        for line in result.kg_context_text.split("\n")[:5]:
                            print(f"    {line}")
    finally:
        conn.close()

    _compute_summary(summary, conditions)
    return summary


def _avg(vals):
    return sum(vals) / len(vals) if vals else 0


def _compute_summary(summary: EvalSummary, conditions: list[str]):
    """Compute per-category and overall averages."""
    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(lambda: {"s": [], "t": [], "l": []}))
    for r in summary.results:
        b = buckets[r.category][r.condition]
        b["s"].append(r.factual_score)
        b["t"].append(r.kg_block_tokens)
        b["l"].append(r.llm_response_ms)

    by_cat = {}
    for cat in sorted(buckets.keys()):
        by_cat[cat] = {}
        for cond in conditions:
            b = buckets[cat].get(cond, {"s": [], "t": [], "l": []})
            by_cat[cat][cond] = {
                "avg_score": _avg(b["s"]), "avg_kg_tokens": _avg(b["t"]),
                "avg_latency_ms": _avg(b["l"]), "count": len(b["s"]),
            }
    summary.by_category = by_cat

    overall = {}
    for cond in conditions:
        cr = [r for r in summary.results if r.condition == cond]
        overall[cond] = {
            "avg_score": _avg([r.factual_score for r in cr]),
            "avg_kg_tokens": _avg([r.kg_block_tokens for r in cr]),
            "avg_latency_ms": _avg([r.llm_response_ms for r in cr]),
            "count": len(cr),
        }
    summary.overall = overall
    summary.closure_checks = _check_closure_differentiation(summary, conditions)


def _group_by_prompt(results) -> dict[str, dict[str, 'EvalResult']]:
    by_prompt: dict[str, dict[str, EvalResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_id, {})[r.condition] = r
    return by_prompt


def _check_closure_differentiation(summary: EvalSummary, conditions: list[str]) -> dict:
    """Check closure_expected items: B must have 0 derived, C must have >=1."""
    checks: dict = {"items": [], "pass_count": 0, "fail_count": 0}
    if "B" not in conditions or "C" not in conditions:
        return checks
    for pid, conds in sorted(_group_by_prompt(summary.results).items()):
        b, c = conds.get("B"), conds.get("C")
        if not b or not c or not c.closure_expected:
            continue
        b_ok, c_ok = b.derived_edges_count == 0, c.derived_edges_count >= 1
        passed = b_ok and c_ok
        checks["items"].append({
            "prompt_id": pid, "closure_rule": c.closure_rule,
            "b_derived": b.derived_edges_count, "c_derived": c.derived_edges_count,
            "b_ok": b_ok, "c_ok": c_ok, "passed": passed,
        })
        checks["pass_count" if passed else "fail_count"] += 1
    return checks


COND_LABELS = {"A": "A (base)", "B": "B (KG)", "C": "C (KG+cl)"}


def _build_closure_analysis(summary: EvalSummary, conditions: list[str]) -> list[str]:
    """Build closure analysis table with B/C scores, oracle hits, avg relevance."""
    if "B" not in conditions or "C" not in conditions:
        return []
    by_rule: dict[str, dict] = {}
    for _pid, conds in _group_by_prompt(summary.results).items():
        c, b = conds.get("C"), conds.get("B")
        if not c or not c.closure_expected:
            continue
        rule = c.closure_rule or "UNKNOWN"
        e = by_rule.setdefault(rule, {
            "bs": [], "cs": [], "oh": 0, "ot": 0, "rels": [],
        })
        if b:
            e["bs"].append(b.factual_score)
        e["cs"].append(c.factual_score)
        e["ot"] += 1
        if c.oracle_hit:
            e["oh"] += 1
        if c.derived_max_relevance > 0:
            e["rels"].append(c.derived_max_relevance)
    if not by_rule:
        return []

    hsep = "-" * 20 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 12
    lines = [
        "Closure Analysis (closure_expected items only)", "=" * 68,
        f"{'':20s}|{'B score':>10s}|{'C score':>10s}|{'oracle_hit':>12s}|{'avg_max_rel':>12s}",
        hsep,
    ]
    ab, ac, ah, at_, ar = [], [], 0, 0, []
    for rule in sorted(by_rule.keys()):
        e = by_rule[rule]
        lines.append(
            f"{rule:20s}|{_avg(e['bs']):10.2f}|{_avg(e['cs']):10.2f}"
            f"|{e['oh']:>5d}/{e['ot']:<5d}|{_avg(e['rels']):12.2f}"
        )
        ab.extend(e["bs"]); ac.extend(e["cs"])
        ah += e["oh"]; at_ += e["ot"]; ar.extend(e["rels"])
    lines.append(hsep)
    lines.append(
        f"{'ALL closure':20s}|{_avg(ab):10.2f}|{_avg(ac):10.2f}"
        f"|{ah:>5d}/{at_:<5d}|{_avg(ar):12.2f}"
    )
    return lines


def format_summary_table(summary: EvalSummary, conditions: list[str]) -> str:
    """Format the summary as a text table."""
    sep = "-" * 22 + ("+" + "-" * 12) * len(conditions)

    def _row(label, vals):
        r = f"{label:<22}"
        for v in vals:
            r += f"|  {v:>8}  "
        return r

    lines = ["KG Ablation Results", "=" * 60]
    lines.append(_row("Category", [COND_LABELS.get(c, c) for c in conditions]))
    lines.append(sep)
    for cat in sorted(summary.by_category.keys()):
        lines.append(_row(cat, [f"{summary.by_category[cat].get(c, {}).get('avg_score', 0):.2f}" for c in conditions]))
    lines.append(sep)
    lines.append(_row("OVERALL", [f"{summary.overall.get(c, {}).get('avg_score', 0):.2f}" for c in conditions]))
    lines.append(sep)
    lines.append(_row("Avg KG tokens", [f"{summary.overall.get(c, {}).get('avg_kg_tokens', 0):.0f}" for c in conditions]))
    lines.append(_row("Avg latency (ms)", [f"{summary.overall.get(c, {}).get('avg_latency_ms', 0):.0f}" for c in conditions]))
    lines.append("=" * 60)

    # Closure differentiation section
    checks = summary.closure_checks
    if checks and checks.get("items"):
        lines.append("")
        lines.append("Closure Differentiation (B vs C)")
        lines.append("-" * 60)
        p = checks["pass_count"]
        f_ = checks["fail_count"]
        lines.append(f"  {p} passed, {f_} failed out of {p + f_} closure items")
        if f_ > 0:
            lines.append("  FAILURES:")
            for item in checks["items"]:
                if not item["passed"]:
                    lines.append(
                        f"    {item['prompt_id']} [{item['closure_rule']}]: "
                        f"B_derived={item['b_derived']} "
                        f"C_derived={item['c_derived']}"
                    )

    # Closure analysis table (relevance + oracle)
    closure_analysis = _build_closure_analysis(summary, conditions)
    if closure_analysis:
        lines.append("")
        lines.extend(closure_analysis)

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
        "closure_checks": summary.closure_checks,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path
