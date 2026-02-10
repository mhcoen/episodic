# KG Context Impact Evaluation + Operator UX

## Part 1: Ablation Harness

### Overview

Measure whether KG injection improves answer quality under the same token budget. Three conditions:

- **A** (baseline): ancestry + topic summaries. `kg_context=False`.
- **B** (KG, no closure): `kg_context=True`, `kg_max_derived=0`.
- **C** (KG + closure): `kg_context=True`, `kg_max_derived=3`.

### Architecture

New module: `episodic/kg/eval_ablation.py`

The harness:
1. Loads a dataset of (prompt, answer_key, category) tuples from a JSON file
2. For each prompt, builds context under all 3 conditions
3. Sends each context + prompt to the LLM
4. Scores the response against the answer key
5. Records token cost of injected KG block and wall-clock latency
6. Produces a summary table

### Dataset: `episodic/kg/eval_dataset.json`

JSON array of objects:

```json
[
  {
    "id": "rel_bridge_01",
    "prompt": "Can she run local models on it?",
    "setup_context": [
      "My daughter Emma has a MacBook Pro M3 Max with 64 gigs of RAM",
      "She's studying computer science at MIT"
    ],
    "answer_key": {
      "required_facts": ["Emma", "MacBook Pro M3 Max", "64 gigs of RAM"],
      "expected_answer_contains": ["yes", "64"],
      "category": "relational_bridging"
    }
  },
  ...
]
```

Fields:
- `id`: unique identifier
- `prompt`: the user query to evaluate
- `setup_context`: prior conversation turns that establish facts (loaded into ancestry before the prompt). These are inserted as alternating user/assistant messages into the conversation DAG before the evaluation prompt.
- `answer_key.required_facts`: entity names or facts that MUST appear in a correct response
- `answer_key.expected_answer_contains`: substring checks on the LLM response (case-insensitive)
- `answer_key.category`: one of `relational_bridging`, `alias`, `temporal_negation`, `multi_hop`, `baseline_factual`

### Dataset Contents (minimum 50 prompts)

Create the dataset with these categories. Draw from real transcript content where possible (entity names, predicates already in the KG). Supplement with synthetic cases.

**Relational bridging** (10–15 prompts): Queries requiring entity-bridging through KG edges.
- "Can she run local models on it?" (requires Emma → MacBook → RAM)
- "Does he have room for a garden there?" (requires person → location → property)
- "What does my neighbor use for pizza?" (requires Dave → Ooni Koda)

**Alias resolution** (10–15 prompts): Queries using non-canonical surface forms.
- "How much RAM does my laptop have?" (laptop → MacBook Pro M3 Max)
- "What switches are in my keyboard?" (keyboard → Keychron Q1)
- "Does vim support LSP natively?" (vim → Neovim)
- "my M3 Max" → MacBook Pro M3 Max

**Temporal/negation** (5–10 prompts): Queries where TIME_PAST or negation matters.
- "Where did I used to work?" (should surface TIME_PAST edges)
- "What editor did I switch away from?" (temporal cue)

**Multi-hop** (10–15 prompts): Queries requiring 2+ edge traversals or closure.
- "Where does my daughter study?" (user → related_to → Emma, Emma → located_at → MIT)
- "What specs does my main machine have?" (user → has → MacBook, MacBook → has → RAM)
- "What kind of dog does my wife's family have?" (requires multiple hops)

**Baseline factual** (10–15 prompts): Simple factual queries that KG should help with but don't require bridging.
- "What keyboard do I use?"
- "What programming languages do I prefer?"
- "Do I have any pets?"

### Scoring

For each (prompt, condition) pair, produce:

```python
@dataclass
class EvalResult:
    prompt_id: str
    condition: str  # "A", "B", "C"
    category: str
    # Correctness
    required_facts_found: int
    required_facts_total: int
    expected_contains_found: int
    expected_contains_total: int
    factual_score: float  # (facts_found + contains_found) / (facts_total + contains_total)
    # Cost
    kg_block_tokens: int  # 0 for condition A
    total_prompt_tokens: int  # from LLM response usage
    total_completion_tokens: int
    # Latency
    context_build_ms: float  # wall clock for context assembly
    llm_response_ms: float  # wall clock for LLM response
    # Raw
    llm_response: str  # full response text
    kg_context_text: str  # injected KG block (empty for A)
```

`factual_score` = (required_facts_found + expected_contains_found) / (required_facts_total + expected_contains_total)

This is deliberately simple — no LLM-as-judge. Substring matching + required fact presence. Auditable, deterministic, reproducible.

### Summary Output

After all prompts evaluated, produce:

```
KG Ablation Results
═══════════════════════════════════════════════════
Category             |  A (base) |  B (KG)   |  C (KG+cl)
─────────────────────┼───────────┼───────────┼───────────
relational_bridging  |  0.42     |  0.78     |  0.85
alias                |  0.55     |  0.82     |  0.82
temporal_negation    |  0.60     |  0.70     |  0.70
multi_hop            |  0.30     |  0.65     |  0.80
baseline_factual     |  0.85     |  0.90     |  0.90
─────────────────────┼───────────┼───────────┼───────────
OVERALL              |  0.54     |  0.77     |  0.81
─────────────────────┼───────────┼───────────┼───────────
Avg KG tokens        |  0       |  42       |  55
Avg latency (ms)     |  320     |  325      |  328
═══════════════════════════════════════════════════
```

Also save full results to `episodic/kg/eval_results.json` for inspection.

### Implementation Details

#### Context Assembly for Evaluation

The harness must simulate real context building. For each prompt:

1. Create a temporary SQLite DB with `ensure_kg_schema()`
2. Insert the `setup_context` messages as nodes in the conversation DAG (alternating user/assistant roles)
3. Run KG extraction on the user nodes (using the real extraction pipeline, not mocks)
4. Set the config flags for the current condition (A/B/C)
5. Build context via `ContextBuilder._add_kg_context()` (or the equivalent path)
6. Send to LLM, collect response
7. Score

**Important**: steps 1–3 happen ONCE per prompt (shared across conditions A/B/C). Only steps 4–7 vary per condition. This ensures the KG state is identical across conditions.

However, this makes the harness expensive (each prompt requires LLM extraction calls for setup + 3 LLM chat calls for evaluation). For cost control:
- Use the live production DB for the eval dataset (entities/edges already extracted). The `setup_context` messages reference entities already in the KG.
- Build context from the live DB, varying only kg_context and kg_max_derived configs.
- This avoids re-extraction entirely. The harness only makes 3 LLM chat calls per prompt (one per condition).

#### Using the Live DB

```python
def run_ablation(dataset_path: str, db_path: str = None):
    """Run ablation against existing DB with extracted KG."""
    # Open existing DB (production or test copy)
    # For each prompt:
    #   Insert prompt as a temporary node (or just call get_kg_context directly with prompt text)
    #   For each condition:
    #     Set config
    #     Build context (messages list)
    #     Call LLM
    #     Score
```

Actually, even simpler: don't insert nodes. Just call `get_kg_context(prompt_text, conn)` directly — it only needs the user text and the KG tables. Then prepend the setup_context as conversation history messages manually. The ContextBuilder is not needed; we assemble the messages list directly.

```python
messages = []
# Add setup context as conversation history
for i, ctx in enumerate(setup_context):
    role = "user" if i % 2 == 0 else "assistant"
    messages.append({"role": role, "content": ctx})

# Add KG context (condition B or C)
if condition != "A":
    kg_result = get_kg_context(prompt, conn)
    if kg_result:
        messages.insert(0, {"role": "system", "content": kg_result.text})

# Add the eval prompt
messages.append({"role": "user", "content": prompt})

# Call LLM
response = litellm.completion(model=model, messages=messages)
```

This is clean, auditable, and avoids any dependency on the conversation DAG.

### CLI Command

`/kg eval [dataset_path] [--model MODEL] [--conditions A,B,C] [--dry-run]`

- `--dry-run`: show the dataset, KG context per condition, but don't call the LLM. Useful for verifying the harness before spending tokens.
- Default model: current chat model from config.
- Default dataset: `episodic/kg/eval_dataset.json`.
- Output: summary table to terminal + full results to `episodic/kg/eval_results.json`.

---

## Part 2: Operator UX Commands

### `/kg explain last`

Show what happened on the most recent KG context injection.

Implementation: `ContextBuilder._add_kg_context()` already stores `self.kg_context` (a `KGContextResult`). The `/kg explain last` command reads from the most recent ContextBuilder instance.

But the ContextBuilder is transient — it's created per `handle_chat_message`. We need to persist the last KGContextResult. Two options:

**Option A** (simplest): Store on a module-level variable in context_source.py alongside `_mention_dict`. After each `get_kg_context()` call, save the result to `_last_kg_result`.

```python
# In context_source.py
_last_kg_result: Optional[KGContextResult] = None

def get_kg_context(...) -> Optional[KGContextResult]:
    global _last_kg_result
    ...
    _last_kg_result = result
    return result

def get_last_kg_result() -> Optional[KGContextResult]:
    return _last_kg_result
```

**Output format** (rich table or plain text):

```
/kg explain last
═══════════════════════════════════════════════════
KG Context Injection Report
───────────────────────────────────────────────────
Matched entities:
  1. emma (id=763, w=1.0, match="canonical")
  2. mit (id=765, w=1.0, match="canonical")

Selected edges (4):
  ✓ <user> related_to Emma         [node:662] rank=10.00066
  ✓ Emma located_at MIT             [node:714] rank=10.00071
  ✓ Emma studies computer science   [node:666] rank=0.00067
  ✓ Emma wants NLP research         [node:666] rank=0.00066

Derived rules fired (0):
  (none — closure suppressed: Emma→located_at→MIT already in edges)

Budget: 45/500 tokens
Cache: hit

Dropped edges (budget):
  (none — all edges fit within budget)
```

The key additions vs `/kg probe`:
- **rank scores** for each edge (so you can see why edges were ordered as they were)
- **dropped edges** with their scores (edges that were retrieved but cut for budget)
- **derived rules fired** with suppression reasons

To support "dropped edges", `get_kg_context()` needs to track pre-truncation state. Currently `format_kg_context()` silently drops edges over budget. Change:

Add to `KGContextResult`:
```python
dropped_edges: list[EdgeFact] = field(default_factory=list)
dropped_derived: list[DerivedFact] = field(default_factory=list)
```

In `format_kg_context()`, return a richer object (or modify `get_kg_context` to track what was dropped):

```python
def format_kg_context(facts, derived_facts, budget_tokens=500):
    # ... existing logic ...
    # Track what was dropped
    dropped = [(rank, line) for rank, line in lines if not included]
    return text, dropped_facts, dropped_derived
```

Or simpler: have `get_kg_context()` compare the full edge list against what made it into the formatted text.

### `/kg blame <line>`

Given a line from the injected KG block, show its full provenance chain.

Input: a line number (1-indexed) from the last injected block, or a substring match.

```
/kg blame "Emma located_at MIT"
═══════════════════════════════════════════════════
Edge: Emma located_at MIT
───────────────────────────────────────────────────
Type: direct (not derived)
Source: node 714, assertion 201
Assertion span: "I work with Emma at MIT on natural language processing research"
                                    ^^^^     ^^^
Subject mention: "Emma" [span 17:21, conf=0.95]
Object mention: "MIT" [span 25:28, conf=0.95]
Predicate: located_at
Edge confidence: 0.95
Tags: []
Extraction model: gpt-4o
───────────────────────────────────────────────────
```

For derived edges:
```
/kg blame "Emma located_at MIT" (if derived)
═══════════════════════════════════════════════════
Edge: Emma located_at MIT
───────────────────────────────────────────────────
Type: derived (rule: KINSHIP_LOCATION)
Source edges:
  1. <user> related_to Emma [node:662, assertion:150]
     Span: "My daughter Emma studies computer science"
  2. Emma located_at MIT [node:714, assertion:201]
     Span: "I work with Emma at MIT on natural language processing research"
───────────────────────────────────────────────────
```

Implementation: the `KGContextResult` already contains `edges` (list of EdgeFact with assertion_id) and `derived` (list of DerivedFact with source_node_ids). `/kg blame` looks up the edge by matching the formatted line text, then queries the DB for assertion details:

```sql
SELECT a.assertion_id, a.source_node_id, a.span_start, a.span_end, a.tags,
       n.content
FROM kg_assertions a
JOIN nodes n ON a.source_node_id = n.rowid
WHERE a.assertion_id = ?
```

Extract the span text: `content[span_start:span_end]`.

For mention details:
```sql
SELECT m.span_start, m.span_end, m.surface_text, m.confidence
FROM kg_mentions m
WHERE m.node_id = ? AND m.entity_id = ?
```

### Changes Summary for Part 2

1. `context_source.py`:
   - Add `_last_kg_result` module-level variable
   - Add `dropped_edges`, `dropped_derived` fields to `KGContextResult`
   - Track dropped edges in `format_kg_context()` or `get_kg_context()`
   - Add `get_last_kg_result()` accessor

2. `commands/kg.py`:
   - Add `/kg explain last` subcommand
   - Add `/kg blame <text>` subcommand
   - Both query `get_last_kg_result()` and the DB for provenance

---

## Implementation Order

1. Create `episodic/kg/eval_dataset.json` (50+ prompts, 5 categories)
2. Create `episodic/kg/eval_ablation.py` (harness)
3. Add `/kg eval` CLI command
4. Run `--dry-run` to verify dataset and KG context per condition
5. Run full eval with LLM calls, save results
6. Report summary table

7. Add dropped_edges tracking to context_source.py
8. Add `_last_kg_result` persistence
9. Implement `/kg explain last`
10. Implement `/kg blame`
11. Run pytest, report results

Report: eval summary table + test results.
