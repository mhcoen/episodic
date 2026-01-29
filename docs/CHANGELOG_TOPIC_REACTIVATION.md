# Topic Reactivation System: Technical Changelog

This document describes what changed in the Topic Reactivation engineering roadmap (Phases 1-4 + Tasks 3-5). It serves as a technical reference for systems and ML engineers.

---

## Table of Contents

1. [System Behavior Changes](#1-system-behavior-changes)
2. [UI Changes](#2-ui-changes)
3. [New Commands & Flags](#3-new-commands--flags)
4. [Configuration Keys](#4-configuration-keys)
5. [Invariants & Guarantees](#5-invariants--guarantees)
6. [Performance & Latency](#6-performance--latency)
7. [Database Schema](#7-database-schema)
8. [Module Reference](#8-module-reference)

---

## 1. System Behavior Changes

### 1.1 Reactivation Probe

The reactivation probe (`episodic/recall/reactivation.py`) determines whether a user message relates to a previously discussed dormant topic.

**Decision Types:**
- `CONTINUE` - Stay in current topic (default)
- `REACTIVATE` - Switch context to a dormant topic
- `DISAMBIGUATE` - Multiple topics match, present user with choices

**Gate Sequence (evaluated in order):**
1. **Cooldown Gate** - Returns `CONTINUE` if `cooldown_turns > 0`
2. **Input Length Gate** - Requires >= 4 words in user input
3. **Topic Existence Gate** - Requires topics with centroids in database
4. **Dormancy Filter** - Only considers topics inactive >= 4 turns (`DORMANCY_MIN`)
5. **Similarity Threshold** - Best topic must have similarity >= 0.3
6. **Support Gate** - Best topic must have `support_count >= S_SUPPORT` (default: 2)
7. **Ambiguity Check** - Detects when multiple topics have similar scores with support

**Key Constants:**
```python
K_TOPICS = 7           # Topics to consider in ANN search
M_EXCHANGES = 12       # Exchanges to check for support
S_SUPPORT = 2          # Minimum support count
DELTA_BAND = 0.15      # Similarity band for support matching
COOLDOWN_TURNS = 3     # Turns to wait after reactivation
DORMANCY_MIN = 4       # Minimum turns before reactivation eligible
```

**Confidence Scoring:**
Computed from similarity, support_count, uniqueness, and dormancy. Higher confidence = more reliable reactivation.

### 1.2 Context Recovery Modes

Three modes are available (`context_recovery_mode` config key):

| Mode | Behavior |
|------|----------|
| `ancestry` | Traditional DAG traversal. Includes recent messages regardless of topic boundaries. |
| `topic_local` | Topic-isolated context. Only includes messages from the active topic. |
| `hybrid` | Dynamic selection: uses `topic_local` when reactivation fires, otherwise `ancestry`. **This is the default.** |

**Hybrid Mode Logic:**
```python
if decision.action == "REACTIVATE":
    return TopicLocalStrategy()
else:
    return AncestryStrategy()
```

### 1.3 Topic-Local Assembly

When `topic_local` mode is active, context is assembled as:

1. System prompt
2. Global scratchpad (if any)
3. Topic context block:
   - Topic name
   - Structured summary (from `topic_working_set.summary_md`)
   - Semantic anchors (via Chroma, filtered by topic)
   - Last N exchanges (from `topic_nodes`, ordered by `turn_idx`)
4. Cross-topic imports (if detected)
5. Current user message

**Key Property: "B Disappears"**
When active topic is A, **no nodes from topic B appear in assembled messages**. This is the core contamination guarantee.

### 1.4 topic_nodes Membership

Every exchange is registered to its topic via the `topic_nodes` table:

```sql
CREATE TABLE topic_nodes (
    topic_start_node_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    PRIMARY KEY(topic_start_node_id, node_id)
);
```

**Population Hooks:**
- `add_nodes_to_current_topic()` called after each exchange
- Backfill migration for existing conversations: `m015_topic_local_tables`

### 1.5 Structured Summaries

Topic working sets store structured summaries with explicit schema:

```python
# topic_working_set fields
summary_md: str              # Markdown summary
decisions_json: str          # JSON array of key decisions
open_loops_json: str         # JSON array of unresolved questions
entities_json: str           # JSON array of mentioned entities
summary_version: int         # Schema version (currently 1)
```

**Provenance:**
- `last_summarized_turn_idx` - Which turn was last summarized
- `last_updated_at` - Timestamp of last update

### 1.6 Anchor Retrieval Rules

Semantic anchors are retrieved from Chroma with these rules (in order):

1. **Topic Filter** - Only anchors with matching `topic_start_node_id` in metadata
2. **Similarity Threshold** - Minimum `anchor_similarity_threshold` (default: 0.5)
3. **Recency Dedup** - Skip if already in last N exchanges
4. **Near-Duplicate Dedup** - Skip if cosine similarity > 0.95 with another anchor
5. **Novelty Check** - Reject if similarity > 0.97 with summary (redundant)
6. **Budget Enforcement** - Limit to `anchor_count` (default: 3)

**Anchor Retrieval Config:**
```python
anchor_count = 3                    # Max anchors to include
anchor_similarity_threshold = 0.5   # Min similarity
anchor_retrieval_count = 10         # Candidates to fetch from Chroma
```

### 1.7 Chroma Backfill & Reconciliation

Existing conversations can be backfilled via migration:

```bash
python -m episodic.migrations.m015_topic_local_tables backfill
```

**Reconciliation Report:**
- Topics processed
- Nodes added to topic_nodes
- Chroma entries updated with topic metadata
- Errors encountered

### 1.8 Thin Topic Fallback

When `topic_local` context is insufficient, it falls back to `ancestry`:

**Fallback Triggers (all must be true):**
- No summary text
- < `min_anchors_for_topic_local` anchors (default: 2)
- < `min_tokens_for_topic_local` total tokens (default: 500)

**Debug Output:**
```python
debug["fallback_reason"] = "thin_topic_local"
debug["thin_fallback_details"] = {
    "has_summary": False,
    "anchor_count": 1,
    "token_count": 342
}
```

---

## 2. UI Changes

### 2.1 Disambiguation Flow

When `DISAMBIGUATE` is returned, users see:

```
I found multiple topics that might match:

[1] python-debugging (12 turns ago)
    - "How do I fix IndexError?"
    - "What about try-except?"
    3 matching exchanges

[2] coffee-brewing (45 turns ago)
    - "Best pour-over ratio?"
    2 matching exchanges

[0] Neither / Continue current topic

Which topic?
```

**Display Elements:**
- Up to 3 options shown
- Each shows: topic name, turns ago, snippets (max 2), support count
- Option `[0]` always available to continue current topic

**Input Handling:**
- `0` - Continue current topic
- `1`-`N` - Select that option
- Invalid - Reprompt once, then auto-continue

### 2.2 Debug Output Enhancements

**Timing Spans (in debug):**
```python
debug["timing"] = {
    "sqlite_ops_ms": float,
    "chroma_query_ms": float,
    "context_assembly_ms": float  # Total excluding embedding
}
```

**Token Breakdown (in debug):**
```python
debug["token_breakdown"] = {
    "summary_tokens": int,
    "recency_tokens": int,
    "anchor_tokens": int,
    "scratchpad_tokens": int,
    "import_tokens": int,
    "total_tokens": int
}
```

**Reactivation Decision Debug:**
```python
debug = {
    "cooldown_turns": int,
    "active_topic": str,
    "candidates": [{"topic": str, "sim": float, "rank": int, "dormancy": int}, ...],
    "best_similarity": float,
    "best_topic": str,
    "support_counts": {"topic": count, ...},
    "gates_passed": [str, ...],
    "gates_failed": [str, ...],
    "confidence": float,
    "top_k_similarities": [(name, sim), ...],
    "ambiguity_detected": bool,
    "dormancy_turns": int
}
```

---

## 3. New Commands & Flags

### 3.1 Evaluation Commands

```bash
# Reactivation replay
/evaluate reactivation              # Replay recent 100 turns
/evaluate reactivation --all        # Full history
/evaluate reactivation --limit 50   # Specific limit
/evaluate reactivation --export features.jsonl  # Export for analysis
/evaluate reactivation --labeled    # Only ground truth labeled

# Resume benchmark
/evaluate benchmark                 # Compare modes on resume scenarios
/evaluate benchmark --min-gap 10    # Filter by gap size
/evaluate benchmark --export bench.json

# Quality evaluation
/evaluate quality                   # Run on all 50 labeled moments
/evaluate quality --category short_gap  # Filter by category
/evaluate quality --export          # Export markdown for human review
/evaluate quality --llm             # Actually call LLM (expensive)

# Calibration
/evaluate calibrate                 # Run with seed 42
/evaluate calibrate --seed 123      # Custom seed
/evaluate calibrate --no-cv         # Skip cross-validation
```

### 3.2 Config Commands

```bash
/set context_recovery_mode hybrid
/set enable_topic_reactivation true
/set context_token_budget 4000
/set min_anchors_for_topic_local 2
/set min_tokens_for_topic_local 500
/set debug true  # Enable debug output
```

---

## 4. Configuration Keys

### 4.1 Core Reactivation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_topic_reactivation` | bool | `false` | Enable implicit topic reactivation |
| `context_recovery_mode` | str | `"hybrid"` | `ancestry`, `topic_local`, or `hybrid` |
| `context_token_budget` | int | `4000` | Max tokens for context assembly |
| `reactivation_log_features` | bool | `true` | Log probe features for every decision |

### 4.2 Anchor & Retrieval

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `anchor_count` | int | `3` | Semantic anchors to retrieve |
| `anchor_similarity_threshold` | float | `0.5` | Min similarity for anchors |
| `anchor_retrieval_count` | int | `10` | Candidates to fetch from Chroma |

### 4.3 Topic-Local Fallback

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_anchors_for_topic_local` | int | `2` | Min anchors before topic_local |
| `min_tokens_for_topic_local` | int | `500` | Min tokens before topic_local |

### 4.4 Cross-Topic Imports

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `import_detection_enabled` | bool | `true` | Enable cross-topic import detection |
| `import_token_budget` | int | `100` | Max tokens for imported context |

---

## 5. Invariants & Guarantees

### 5.1 Contamination Guarantee

**Invariant:** When in topic A, assembled context contains **zero** nodes from topic B.

- Verified by unit tests (`test_b_disappears`)
- Enforced at assembly time via `topic_nodes` membership
- Measured by `contamination_rate` metric (must be 0%)

### 5.2 Cooldown Invariant

**Invariant:** After a reactivation, no further reactivation occurs for `COOLDOWN_TURNS` (default: 3).

- Prevents rapid topic switching
- Cooldown decremented each turn
- Can be configured via `reactivation_cooldown`

### 5.3 Dormancy Invariant

**Invariant:** Only topics inactive for >= `DORMANCY_MIN` turns (default: 4) are considered for reactivation.

- Prevents switching to recently active topics
- Active topic always excluded from candidates

### 5.4 Support Invariant

**Invariant:** Reactivation requires >= `S_SUPPORT` (default: 2) matching exchanges in recent history.

- Prevents false positives from single keywords
- Support computed within `DELTA_BAND` similarity of best match

### 5.5 Logged/Validated Data

**Per-turn logging (when `reactivation_log_features=true`):**
- `reactivation_decisions` table: decision, reason, confidence, candidates, gates
- `reactivation_labels` table: ground truth for evaluation

**Validation at runtime:**
- `topic_nodes` membership checked on assembly
- Anchor dedup verified before inclusion
- Token budget enforced with truncation info

---

## 6. Performance & Latency

### 6.1 Latency Expectations

| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Context assembly (total) | < 20ms | < 50ms | < 100ms |
| Chroma query | < 10ms | < 30ms | < 50ms |
| SQLite ops | < 5ms | < 15ms | < 30ms |

**Measured via:** `scripts/latency_benchmark.py`

### 6.2 Token Budget Adherence

- Total tokens <= `context_token_budget` (default: 4000)
- Enforced via truncation with priority: summary > anchors > recency
- p99 should be <= budget (verified by benchmark)

### 6.3 Benchmark Artifacts

**Output:** `episodic/evaluation/reports/latency_benchmark.json`
```json
{
  "p50_assembly_ms": 18.2,
  "p95_assembly_ms": 42.1,
  "p99_assembly_ms": 67.4,
  "token_budget_violations": 0,
  "iterations": 60,
  "timestamp": "..."
}
```

---

## 7. Database Schema

### 7.1 topic_nodes

```sql
CREATE TABLE topic_nodes (
    topic_start_node_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(topic_start_node_id, node_id)
);
CREATE INDEX idx_topic_nodes_turn ON topic_nodes(topic_start_node_id, turn_idx);
CREATE INDEX idx_topic_nodes_node ON topic_nodes(node_id);
```

### 7.2 topic_working_set

```sql
CREATE TABLE topic_working_set (
    topic_start_node_id TEXT PRIMARY KEY,
    topic_name TEXT,
    summary_md TEXT NOT NULL DEFAULT '',
    decisions_json TEXT NOT NULL DEFAULT '[]',
    open_loops_json TEXT NOT NULL DEFAULT '[]',
    entities_json TEXT NOT NULL DEFAULT '[]',
    last_summarized_turn_idx INTEGER,
    last_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    summary_version INTEGER NOT NULL DEFAULT 1
);
```

### 7.3 topic_centroids

```sql
CREATE TABLE topic_centroids (
    start_node_id TEXT PRIMARY KEY,
    centroid_medoid_exchange_id TEXT,
    exchange_count INTEGER DEFAULT 0,
    last_active_turn_idx INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_topic_centroids_turn ON topic_centroids(last_active_turn_idx);
```

### 7.4 reactivation_decisions

```sql
CREATE TABLE reactivation_decisions (
    user_node_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    reason TEXT,
    confidence REAL,
    topic_name TEXT,
    topic_start_node_id TEXT,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    support_counts_json TEXT NOT NULL DEFAULT '{}',
    gates_json TEXT NOT NULL DEFAULT '{"passed": [], "failed": []}',
    best_similarity REAL,
    best_support_count INTEGER,
    dormancy_turns INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

---

## 8. Module Reference

### 8.1 Evaluation Modules

| Module | Purpose |
|--------|---------|
| `episodic/evaluation/resume_moments.py` | ResumeMoment dataclass, fixtures loading |
| `episodic/evaluation/quality_eval.py` | Quality evaluation runner, export functions |
| `episodic/evaluation/calibration.py` | Calibration sweep, LOBO-CV, parameter selection |
| `episodic/evaluation/reactivation_replay.py` | Replay historical decisions |
| `episodic/evaluation/resume_benchmark.py` | Mode comparison benchmark |

### 8.2 Calibration System

**Metrics:**
```python
@dataclass
class CalibrationMetrics:
    reactivation_precision: float  # Correct reactivations / total reactivations
    reactivation_recall: float     # Detected resumes / actual resumes
    thrash_rate: float             # Rapid topic switches
    disambiguation_burden: float   # False disambiguation rate
    thin_fallback_rate: float      # Thin topic fallback frequency
    contamination_rate: float      # Must be 0% (hard constraint)
```

**LOBO-CV Protocol:**
- Leave-One-Bucket-Out cross-validation
- 5 buckets: short_gap, medium_gap, long_gap, ambiguous, thin_topic
- Train on 4, evaluate on held-out
- Rotate through all 5

**Lexicographic Objective:**
```python
OBJECTIVE_WEIGHTS = {
    "reactivation_precision": 5,   # Highest priority
    "thrash_rate": -3,             # Minimize
    "disambiguation_burden": -2,   # Minimize
    "reactivation_recall": 1,      # Lowest priority
}
```

**Output Artifacts:**
- `episodic/evaluation/reports/calibrated_params.json` - Best config
- `episodic/evaluation/reports/calibration_report.csv` - All results

### 8.3 CI Hardening

**Chroma Test Isolation:**
- Hermetic fixtures in `tests/chroma_isolation.py`
- Unique `tmp_path` per test
- Unique collection names (uuid4 hex)
- Explicit client cleanup on teardown
- `@pytest.mark.serial` for sequential execution

**Anti-Flake Verification:**
- 10x loop test before CI enabling
- All memory integration tests pass consistently

---

## Appendix: Resume Moments Categories

| Category | gap_turns | Description |
|----------|-----------|-------------|
| `short_gap` | 5-15 | Quick topic switches, usually successful |
| `medium_gap` | 20-50 | Moderate gap, may need summary |
| `long_gap` | 100+ | Long dormancy, requires summary |
| `ambiguous` | varies | Multiple topics could match (java programming vs java coffee) |
| `thin_topic` | varies | Topic lacks sufficient history |

**Fixture:** `episodic/evaluation/fixtures/resume_moments.json` (50 labeled moments)
