# Correctness and Replay Infrastructure

This document describes the correctness substrate implemented as prerequisites to Phase 2 (relevance-aware truncation). These components ensure that context assembly is bounded, auditable, reproducible, and testable for semantic safety.

## 1. Token Budget Enforcement

### Module: `episodic/token_guard.py`

Provides end-to-end token budget enforcement on fully assembled message lists, not just memory blocks.

### Core Functions

| Function | Purpose |
|----------|---------|
| `validate_assembly(messages, budget, counter)` | Check total tokens ≤ cap, return validation result |
| `guard_assembly(messages, budget)` | Convenience wrapper with automatic drop policy |
| `estimate_tokens_messages(messages, counter)` | Full assembly token count with breakdown |

### Fail-Closed Drop Policy

When assembled tokens exceed cap, drops occur in this order:
1. Truncate summary further (down to `token_summary_min`)
2. Drop recency exchanges
3. Drop anchor exchanges
4. Abort with safe fallback response

No path can exceed cap without being logged as a bug event.

### TokenCounter Protocol

```python
class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...
    def count_messages(self, messages: List[Dict]) -> int: ...
    def is_exact(self) -> bool: ...
    def backend_name(self) -> str: ...
```

### Safety Factor Semantics

- Config: `token_safety_factor_heuristic` (default 1.2)
- Applied **iff** `counter.is_exact() == False`
- Effective count = `raw_count × safety_factor` for heuristic counters
- Ensures unknown-tokenizer backends respect caps by construction

### Registry

Token counters are registered by `(provider_id, model_id)`. If no exact backend exists, `HeuristicTokenCounter` (chars/4) is returned.

### Configuration

```json
{
  "token_full_cap": 8000,
  "token_summary_min": 100,
  "token_overhead_reserve": 500,
  "token_safety_factor_heuristic": 1.2
}
```

---

## 2. Guard Event Log Integrity

### Module: `episodic/token_guard_events.py`

Provides schema-versioned, tamper-evident logging for every token guard decision.

### Schema Version

`SCHEMA_VERSION = "1.0"`

Forward-compatible: unknown fields are allowed; required fields are enforced.

### Event Types

```python
class EventType(Enum):
    TOKEN_OK = "token_ok"
    TOKEN_OVERFLOW_RECOVERED = "token_overflow_recovered"
    TOKEN_OVERFLOW_ABORT = "token_overflow_abort"
```

### TokenGuardEvent Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | str | "1.0" |
| `event_type` | EventType | Decision outcome |
| `run_id` | str | Unique per CLI session |
| `turn_id` | str | Unique per conversation turn |
| `assembly_id` | str | Unique per assembly call |
| `ts` | str | ISO 8601 timestamp |
| `event_seq` | int | Monotonic per run_id |
| `counter_backend` | str | TokenCounter backend name |
| `counter_exact` | bool | Whether counter is exact |
| `applied_safety_factor` | float/null | Factor applied (if heuristic) |
| `raw_tokens` | int | Pre-factor token count |
| `effective_tokens` | int | Post-factor token count |
| `cap` | int | Configured cap |
| `budget_breakdown` | dict | Per-component token counts |
| `prev_hash` | str/null | Previous event hash (null for first) |
| `hash` | str | SHA-256 of this event |

### TokenGuardEvent.extra (when truncation occurs)

| Field | Type | Description |
|-------|------|-------------|
| `truncation.tokens_before` | int | Token count before truncation |
| `truncation.tokens_after` | int | Token count after truncation |
| `truncation.tokens_freed` | int | Tokens removed |
| `truncation.decisions_count` | int | Total number of drop decisions (full count) |
| `truncation.decisions` | list | First 10 drop decisions (capped for log size) |

Note: `decisions` is capped at 10 entries; use `decisions_count` to detect if truncation details were themselves truncated.

### Exactly-Once Semantics

- Each call to `validate_assembly()` emits exactly one event
- Same `assembly_id` never produces multiple events
- `EventLogger` enforces this constraint

### Hash Chain

```
h_i = SHA256(h_{i-1} || canonical_json(event_i))
```

- `canonical_json()`: sorted keys, compact, stable UTF-8 encoding
- Enables tamper detection: mutating any event breaks the chain

### EventVerifier

```python
verifier = EventVerifier()
verifier.verify_required_fields(event)  # Check required fields present
verifier.verify_hash(event, prev_hash)  # Check single event hash
verifier.verify_stream(events)          # Full stream validation
```

### Output Format

JSONL (one JSON object per line), suitable for streaming and archival.

---

## 3. Snapshot Replay

### Module: `episodic/replay.py`

Provides deterministic replay of context assembly for regression testing and debugging.

### Schema Version

`SNAPSHOT_SCHEMA_VERSION = "1.0"`

### ReplaySnapshot Structure

```python
@dataclass
class ReplaySnapshot:
    # Metadata
    schema_version: str
    run_id: str
    turn_id: str
    provider_id: str
    model_id: str
    tokenizer_backend_name: str
    exact_flag: bool
    safety_factor_config: float
    created_at: str
    
    # Inputs
    context_inputs: ContextInputs  # user_turn, summary, anchors, recency, etc.
    
    # Retrieval state (frozen)
    retrieval_state: RetrievalState  # embedding model, query vector, results, topic mapping
    
    # Outputs
    assembled_messages: List[Dict]  # Exact messages as sent to model
    
    # Token guard state
    token_guard_config: TokenGuardConfig
    token_guard_events: List[Dict]  # JSONL lines
    final_event_hash: str
```

### What "Deterministic Replay" Means

Given a snapshot, `replay()` reproduces:
1. **Assembled messages** — byte-identical strings
2. **Token counts** — same raw and effective counts
3. **Guard decisions** — same event_type sequence
4. **Event hash chain** — verifiable integrity

Timestamps are sourced from snapshot or excluded from equality checks.

### Replay Runner

```python
result = replay(snapshot_path)
if result.success:
    print("Replay matched")
else:
    print(f"Divergence at: {result.diff.field_path}")
    print(f"Expected: {result.diff.expected_snippet}")
    print(f"Actual: {result.diff.actual_snippet}")
```

### ReplayDiff Reporting

On mismatch, returns:
- `field_path`: e.g., `"assembled_messages[2].content"`
- `expected_snippet`: truncated expected value
- `actual_snippet`: truncated actual value
- `message`: human-readable description

### Helper Functions

| Function | Purpose |
|----------|---------|
| `create_snapshot(...)` | Create snapshot during normal operation |
| `assemble_from_snapshot(snapshot)` | Re-assemble using frozen inputs |
| `replay(snapshot_path)` | Full replay with verification |

---

## 4. False Attribution Harness

### Module: `episodic/attribution.py`

Quantifies and regress-tests "false attribution" failures without requiring a live model for the detection logic.

### Definition

A response is a **false attribution** if it contains claims about prior statements, memory, tool results, or temporal facts without support in the supplied context.

### Claim Types

```python
class ClaimType(Enum):
    PRIOR_CONVO = "prior_convo"  # "you said earlier", "we agreed"
    MEMORY = "memory"            # "I remember", "I have in memory"
    TOOL = "tool"                # "I looked it up", "search results show"
    TEMPORAL = "temporal"        # "currently", "latest version"
```

### Attribution Detectors

```python
claims = detect_claims(response_text)
# Returns List[AttributionClaim] with type, span, text_snippet
```

Pattern-based (regex + shallow parsing), deterministic, no LLM in harness.

### Support Checking Rules

| Claim Type | Supported If |
|------------|--------------|
| `prior_convo` | Quoted/paraphrased referent exists in context, OR uncertainty marker present |
| `memory` | Quoted/paraphrased referent exists in context, OR uncertainty marker present |
| `tool` | Quoted/paraphrased referent exists in context, OR uncertainty marker present |
| `temporal` | Tool/web context was present in snapshot |

**Matching thresholds:**
- Key word overlap: 60%+ required
- Similarity threshold: 0.6 minimum for partial matches

**Uncertainty markers:** Claims with "I don't see that in context" are considered supported.

### False Attribution Rate (FAR)

```
FAR = (# unsupported claims) / (# total claims)
```

- FAR = 0 for positive controls (claims are present in context)
- FAR ≤ threshold for mixed sets

### Analysis Functions

```python
# Analyze a single response
report = analyze_response(response_text, assembled_messages)
print(f"FAR: {report.far}")
for claim in report.claims:
    print(f"  {claim.type}: {claim.supported} - {claim.evidence_block or 'none'}")

# Analyze with ReplaySnapshot
report = analyze_snapshot_response(snapshot, response_text)
```

### Mitigation Prompt Knob (test-only)

```python
from episodic.attribution import get_mitigation_prompt, apply_mitigation_to_messages

# Get the mitigation instruction text
prompt = get_mitigation_prompt()

# Apply to message list for testing
mitigated_messages = apply_mitigation_to_messages(messages)
```

The mitigation forces the model to:
- Avoid "you said earlier" unless it can cite a quoted span
- Otherwise say "I don't see that in the provided context"

**Note:** This is for harness testing only, not shipped to production.

---

## 5. How to Run

### Run All Correctness Tests

```bash
# All token guard and replay tests
pytest tests/unit/test_token_guard.py tests/unit/test_token_guard_events.py tests/unit/test_replay.py tests/unit/test_attribution.py -v

# Quick check (all 153 tests)
pytest tests/unit/test_token_guard*.py tests/unit/test_replay.py tests/unit/test_attribution.py
```

### Run Specific Test Categories

```bash
# Token budget enforcement
pytest tests/unit/test_token_guard.py -v

# Log integrity
pytest tests/unit/test_token_guard_events.py -v

# Snapshot replay
pytest tests/unit/test_replay.py -v

# False attribution harness
pytest tests/unit/test_attribution.py -v
```

### Generate and Replay a Snapshot

```python
from episodic.replay import create_snapshot, replay

# During normal operation, create a snapshot
snapshot = create_snapshot(
    run_id="session-123",
    turn_id="turn-456",
    provider_id="openai",
    model_id="gpt-4",
    context_inputs=inputs,
    retrieval_state=retrieval,
    assembled_messages=messages,
    token_guard_config=config,
    token_guard_events=events
)
snapshot.save("/path/to/snapshot.json")

# Later, replay and verify
result = replay("/path/to/snapshot.json")
if not result.success:
    print(f"Divergence: {result.diff}")
```

### Run Attribution Suite and Interpret FAR

```python
from episodic.attribution import analyze_response, analyze_snapshot_response

# Analyze a response
report = analyze_response(
    response_text="You said earlier you wanted to learn Rust...",
    assembled_messages=messages
)

print(f"Total claims: {report.total_claims}")
print(f"Unsupported claims: {report.unsupported_claims}")
print(f"FAR: {report.far:.2%}")

# Per-claim details
for claim in report.claims:
    status = "✓" if claim.supported else "✗"
    print(f"  {status} [{claim.type}] {claim.text_snippet[:50]}...")
    if claim.evidence_block:
        print(f"      Evidence: {claim.evidence_block}")
```

---

## Test Coverage Summary

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_token_guard.py` | 58 | Budget enforcement, drop policy, safety factor, truncation integration, verification audit |
| `test_token_guard_events.py` | 30 | Schema, exactly-once, hash chain, tamper detection |
| `test_replay.py` | 37 | Snapshot serialization, replay, mutation detection |
| `test_attribution.py` | 45 | Claim detection, support checking, FAR, adversarial suite |
| `test_truncation.py` | 35 | Score computation, reference detection, drop order, determinism |
| **Total** | **205** | |

### Verification Audit Tests (`TestVerificationAudit`)

The verification audit tests validate Phase 1+2 completion invariants:

| Test | Purpose |
|------|---------|
| `test_fail_fast_invariant_enforced` | ValueError raised if `enable_relevance_truncation=True` without `anchor_indices` |
| `test_no_fail_fast_when_truncation_disabled` | Fail-fast does NOT fire when truncation disabled |
| `test_anchor_preservation_property` | Within relevance truncation, anchors never dropped before non-anchors exhausted |
| `test_adversarial_determinism_100_runs` | 100 runs produce byte-identical results |
| `test_token_counter_identity_in_truncation` | Same counter used for enforcement and truncation measurement |
| `test_fallback_correctness_legacy_policy_works` | Legacy drop policy functions when truncation disabled |
| `test_call_site_inventory_production_safe` | Production call sites safe by default (truncation disabled) |
| `test_replay_snapshot_golden_determinism` | SHA-256 golden hash matches across 10 replays |

---

## 6. Phase 2: Relevance-Aware Truncation

### Module: `episodic/truncation.py`

When truncation is required (assembled context exceeds token budget), Phase 2 drops by **importance score** rather than age.

### Enabling

```json
{
  "enable_relevance_truncation": true
}
```

**Default:** `false` (no behavior change unless explicitly enabled)

### Importance Score Formula

For each exchange `e`:
```
score(e) = w_anchor × I_anchor(e)
         + w_early × I_early(e)
         + w_lex × lex_sim(e, user_turn)
         + w_ref × I_referenced(e)
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `I_anchor` | 100 | 1 if exchange is a selected anchor, else 0 |
| `I_early` | 3 | 1 if among first `m` exchanges of topic, else 0 |
| `lex_sim` | 2 | Token Jaccard similarity to current query [0,1] |
| `I_referenced` | 5 | 1 if later exchange references this one, else 0 |

**Reference detection (deterministic):**
- Quote overlap ≥ 40 characters (longest common substring), OR
- Explicit markers ("as above", "you said", "we discussed") + ≥ 6 shared key words

### Drop Policy

1. **Drop recency-only exchanges first** (ascending score)
2. **Then drop anchor exchanges** (ascending score, only if still over budget)
3. **Ties broken by older-first** (lower index) for determinism

**Never dropped:**
- System prompt
- Topic header and summary (subject to `s_min`)
- Current user message

### Integration Point

Truncation runs in `validate_assembly()` at the exact point where:
- Final message list is assembled
- Concrete token budget is known
- AFTER Phase 1 anchor/recency assembly
- BEFORE legacy drop policy (summary → recency → anchors)

**Double-drop prevention:** When relevance truncation successfully brings tokens under budget, the legacy drop policy (summary truncation → recency drop → anchor drop) does NOT fire. Only one truncation path executes per assembly. The legacy policy only fires as fallback if relevance truncation alone cannot meet the budget.

### Token Counter Consistency

**Critical invariant:** The `TokenCounter` instance used for `tokens_before`/`tokens_after` in `TruncationResult` is the **same counter** used by `validate_assembly()` for enforcement.

This eliminates audit ambiguity: truncation measurement = enforcement measurement.

When calling `validate_assembly()` with explicit `counter` parameter, that same counter is passed to `truncate_by_relevance()`. When counter is inferred from registry, the same inferred counter is used throughout.

### Invariants

1. **Precondition**: `enable_relevance_truncation=True` requires `anchor_indices` to be non-empty. Raises `ValueError` otherwise.

2. **Determinism**: Truncation decisions are deterministic given: message list, budget, current_query, anchor_indices, and token counter.

3. **Token counting**: Truncation uses the same token counter instance as enforcement—no measurement ambiguity.

4. **Anchor preservation scope**: Anchors are preserved over non-anchors only within relevance truncation. If legacy fallback fires under extreme budget pressure, it is anchor-unaware and may drop anchors.

### Data Structures

```python
@dataclass
class MessageScore:
    index: int
    anchor_contrib: float
    early_contrib: float
    lex_contrib: float
    ref_contrib: float
    total: float

@dataclass
class TruncationDecision:
    index: int
    message_class: str  # "recency" or "anchor"
    score: float
    reason: str

@dataclass
class TruncationResult:
    messages: List[Dict]
    tokens_before: int
    tokens_after: int
    decisions: List[TruncationDecision]
    scores: List[MessageScore]
```

### Logging

Truncation data is included in `TokenGuardEvent.extra` (respects exactly-once constraint):

```python
"truncation": {
    "tokens_before": 5000,
    "tokens_after": 3800,
    "tokens_freed": 1200,
    "decisions_count": 4,
    "decisions": [
        {"index": 5, "class": "recency", "score": 2.3, "reason": "budget"},
        ...
    ]
}
```

### API

```python
from episodic.truncation import truncate_by_relevance, score_messages

# Score messages without truncating
scores = score_messages(messages, current_query, anchor_indices)

# Truncate to fit budget
result = truncate_by_relevance(
    messages=messages,
    token_budget=4000,
    counter=token_counter,
    current_query="What did we decide about the API?",
    anchor_indices={2, 5, 8}
)

print(f"Freed {result.tokens_before - result.tokens_after} tokens")
for d in result.decisions:
    print(f"  Dropped [{d.index}] ({d.message_class}): score={d.score:.2f}")
```

### Integration with validate_assembly()

```python
from episodic.token_guard import validate_assembly

result = validate_assembly(
    messages=messages,
    budget=budget,
    counter=counter,
    enable_relevance_truncation=True,  # Override config
    current_query="What about the deadline?",
    anchor_indices={1, 3, 7}
)
```

---

## Contract for Future Phases

Any new truncation or drop behavior must:
1. **Not break replay determinism** — same snapshot → same outputs
2. **Extend observability** — new decisions must be observable via existing event fields or new schema version
3. **Respect exactly-once logging** — no duplicate event records per assembly
