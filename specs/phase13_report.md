# Phase 1.3 Report: KG Ablation Eval + Operator UX

## Part 1: Ablation Evaluation Harness

### Files Created
- `episodic/kg/eval_dataset.json` — 51 prompts across 5 categories
  - relational_bridging: 11
  - alias: 11
  - temporal_negation: 7
  - multi_hop: 11
  - baseline_factual: 12

- `episodic/kg/eval_ablation.py` — Complete harness (~280 lines)
  - 3 conditions: A (no KG), B (KG, no closure), C (KG + closure)
  - Uses live DB directly, no re-extraction
  - Calls `get_kg_context()` per condition, assembles messages, calls LLM
  - Deterministic scoring: substring matching + required fact presence
  - Outputs summary table + full results JSON

- `tests/kg/test_kg_eval.py` — 10 tests (scoring, message building, dry-run)

### CLI Command
`/kg eval [dataset_path] [--model MODEL] [--conditions A,B,C] [--dry-run]`

### Dry-Run Results
Ran against live DB. Key prompts with KG matches:
- rel_bridge_05 (pizza oven): 18 tok
- rel_bridge_09 (standing desk): 62 tok, 4 edges
- alias_04 (MacBook/M3 Max): 43 tok, 3 edges
- multi_hop_03/07 (Emma/MIT): 60 tok, 4 edges each

Many dataset prompts use synthetic entities not in live DB — by design, to measure what KG adds when entities exist vs. not.

## Part 2: Operator UX

### Dropped Edges Tracking
- `KGContextResult` now has `dropped_edges` and `dropped_derived` fields
- `format_kg_context()` returns `(text, dropped_edges, dropped_derived)` tuple
- Budget-dropped facts are tracked with their rank scores

### _last_kg_result Persistence
- Module-level `_last_kg_result` in context_source.py
- `get_last_kg_result()` accessor function
- Populated on every successful `get_kg_context()` call

### `/kg explain` (renamed from `/kg explain last`)
Shows full injection report:
```
KG Context Injection Report
--------------------------------------------------
Matched entities:
  1. e808 "emma" (w=1.0)
  2. e809 "mit" (w=1.0)

Selected edges (6):
  + Emma --located_at--> MIT  rank=9.981  [node:714]
  + <user> --related_to--> Emma  rank=-0.049  [node:666]
  ...

Derived rules fired (0): (none)

Budget: 69/500 tokens  Cache: rebuilt

Dropped edges: (none — all fit within budget)
```

### `/kg blame <text>`
Shows full provenance chain:
```
Edge: Emma located_at MIT
--------------------------------------------------
Type: direct
Rank: 9.981
Source: node 714, assertion 208
Polarity: affirm  Certainty: explicit
Span [0:63]: "I work with Emma at MIT on natural language processing research"
Subject mention: "Emma" [12:16] conf=0.95
Object mention: "MIT" [20:23] conf=0.95
```

For derived edges, shows the rule name and source node content.

### Files Modified
- `episodic/kg/context_source.py` — dropped_edges/derived fields, _last_kg_result, format return type
- `episodic/commands/kg.py` — routing for eval/explain/blame (compacted to 599 lines)
- `tests/kg/test_kg_context.py` — updated for format_kg_context tuple return
- `tests/unit/test_kg_context.py` — same

### Files Created
- `episodic/commands/kg_explain.py` — explain and blame command logic (~175 lines)

## Test Results
- KG tests: 55 passed (37 original + 10 eval + 8 readside regression = 55)
- Full suite: 3167 passed (13 pre-existing smoke test failures)

## Verification
```
VERIFIED:
- .venv/bin/python -m pytest tests/kg/ tests/unit/test_kg_context.py -x → 55 passed
- .venv/bin/python -m episodic --script /tmp/claude/test_kg_explain.txt
  → /kg probe Emma studies at MIT: 6 edges, 69/500 tokens
  → /kg explain: full report with matched entities, edges, ranks, budget
  → /kg blame Emma located_at: assertion 208, node 714, span text, mention spans
- Dry-run eval: 51 prompts × condition B, KG tokens range 0-62
- Full suite: 3167 passed, 13 pre-existing failures
```

## Ready for Full Eval Run
To run the actual ablation (costs ~150 LLM calls):
```
/kg eval --model gpt-4o-mini
```
This will produce the summary table and save results to `episodic/kg/eval_results.json`.
