# Topic Reactivation in Episodic

> **Note:** This document is the original architecture reference. For the latest information, see:
> - [CHANGELOG_TOPIC_REACTIVATION.md](./CHANGELOG_TOPIC_REACTIVATION.md) - Technical changelog for systems/ML engineers
> - [user_guide_topic_reactivation.md](./user_guide_topic_reactivation.md) - User guide for CLI users

## Overview

Topic reactivation automatically detects when a user's message relates to a previously discussed topic and switches context to that topic. The key behavioral property: **resuming topic A completely excludes topic B from the prompt**.

This document reflects the original architecture design.

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Recovery System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Reactivation     │    │ Context Recovery │                   │
│  │ Probe            │───▶│ Strategy Router  │                   │
│  │ (probe_reactivation)  │ (select_strategy)│                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
│                    ┌──────────────┼──────────────┐              │
│                    ▼              ▼              ▼              │
│           ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│           │ Ancestry   │  │ Topic-Local│  │ Hybrid     │       │
│           │ Strategy   │  │ Strategy   │  │ (dynamic)  │       │
│           └────────────┘  └────────────┘  └────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### File Layout

```
episodic/
├── context_recovery/
│   ├── __init__.py
│   ├── strategy.py          # Interface, router, enums
│   ├── ancestry.py          # Traditional DAG traversal
│   └── topic_local.py       # Topic-isolated assembly
├── recall/
│   ├── reactivation.py      # Probe logic
│   └── centroid.py          # Medoid maintenance
├── db_topic_nodes.py        # Topic membership operations
└── conversation.py          # Integration point
```

---

## Database Schema

### topic_nodes
Fast topic membership mapping for "last N exchanges in topic X" queries.

```sql
CREATE TABLE topic_nodes (
    topic_start_node_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,      -- SQLite rowid from nodes table
    role TEXT NOT NULL,             -- 'user' or 'assistant'
    PRIMARY KEY(topic_start_node_id, node_id)
);

CREATE INDEX idx_topic_nodes_turn ON topic_nodes(topic_start_node_id, turn_idx);
CREATE INDEX idx_topic_nodes_node ON topic_nodes(node_id);
```

### topic_working_set
Persistent topic state for resumption without full transcript.

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

### topic_centroids
Medoid tracking for ANN-based topic retrieval.

```sql
CREATE TABLE topic_centroids (
    start_node_id TEXT PRIMARY KEY,
    centroid_medoid_exchange_id TEXT,
    exchange_count INTEGER DEFAULT 0,
    last_active_turn_idx INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Context Recovery Modes

### Mode: `ancestry` (default)
Traditional behavior. Traverses DAG from current head, includes recent messages regardless of topic boundaries.

### Mode: `topic_local`
Topic-isolated context. Only includes messages from the active topic. Other topics are completely excluded.

### Mode: `hybrid`
Dynamic selection:
- If reactivation fires → use `topic_local`
- Otherwise → use `ancestry`

### Configuration
```bash
/set context_recovery_mode hybrid
/set context_token_budget 4000
```

---

## Context Recovery Strategy Interface

### ContextRecoveryMode (enum)
```python
class ContextRecoveryMode(Enum):
    ANCESTRY = "ancestry"
    TOPIC_LOCAL = "topic_local"
    HYBRID = "hybrid"
```

### ContextAssemblyResult (dataclass)
```python
@dataclass
class ContextAssemblyResult:
    messages: List[Dict[str, str]]  # role/content dicts
    debug: Dict[str, Any]           # Instrumentation data
```

Debug fields:
- `mode`: Which strategy was used
- `active_topic_start_node_id`: Topic context was built from
- `included_node_ids`: List of node IDs in the assembled context
- `token_counts`: Token estimates by section
- `truncation_info`: What was dropped, if anything
- `reactivation_decision`: CONTINUE/REACTIVATE/DISAMBIGUATE
- `reactivation_reason`: Why that decision was made

### Strategy Protocol
```python
class ContextRecoveryStrategy(Protocol):
    def assemble(
        self,
        user_turn_text: str,
        user_node_id: Optional[str],
        active_topic_start_node_id: Optional[str],
        user_embedding: Optional[np.ndarray],
        token_budget: int,
        conn: sqlite3.Connection,
        chroma_collection: Optional[Collection],
    ) -> ContextAssemblyResult:
        ...
```

### Router
```python
def select_strategy(
    mode: ContextRecoveryMode,
    reactivation_decision: Optional[str] = None
) -> ContextRecoveryStrategy:
    if mode == ContextRecoveryMode.ANCESTRY:
        return AncestryStrategy()
    elif mode == ContextRecoveryMode.TOPIC_LOCAL:
        return TopicLocalStrategy()
    elif mode == ContextRecoveryMode.HYBRID:
        if reactivation_decision == "REACTIVATE":
            return TopicLocalStrategy()
        return AncestryStrategy()
```

---

## Topic-Local Assembly

When `topic_local` strategy is selected, context is built as:

```
1. System prompt
2. Global scratchpad (if any)
3. Topic context block:
   - Topic name
   - Summary (from topic_working_set.summary_md, if non-empty)
   - Last N exchanges (from topic_nodes, ordered by turn_idx)
4. Current user message
```

### Key Property: "B Disappears"
If active topic is A, **no nodes from topic B appear in the assembled messages**. This is verified by unit tests.

### Exchange Retrieval
```python
def get_last_n_exchanges_in_topic(
    topic_start_node_id: str,
    n: int = 4,  # 4 exchanges = 8 messages
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """Returns user/assistant pairs ordered oldest to newest."""
```

---

## Reactivation Probe

### Probe Decision Types
```python
@dataclass
class ReactivationDecision:
    action: str                     # "CONTINUE", "REACTIVATE", "DISAMBIGUATE"
    topic_name: Optional[str]
    topic_start_node_id: Optional[str]
    reason: str
    confidence: float
    debug: Dict[str, Any]
    options: Optional[List[...]]    # For DISAMBIGUATE
```

### Probe Flow
```python
def probe_reactivation(
    user_input: str,
    user_embedding: np.ndarray,
    active_topic_start_node_id: Optional[str],
    cooldown_turns: int,
    now: datetime,
    recent_nodes: List[Dict]
) -> ReactivationDecision:
```

Gates (in order):
1. **Cooldown**: If `cooldown_turns > 0`, return CONTINUE
2. **No eligible topics**: If all topics too recent, return CONTINUE
3. **Reactivate to self**: If best topic = active topic, return CONTINUE
4. **Support threshold**: If insufficient matching exchanges, return CONTINUE
5. **Rank gap**: If active topic too close in ranking, return CONTINUE
6. **Ambiguity**: If multiple topics competitive, return DISAMBIGUATE

---

## Integration in conversation.py

### Call Order (Critical)
```python
# 1. Compute user_embedding (already done for drift)
user_embedding = ...

# 2. Run reactivation probe
decision = probe_reactivation(
    user_input=user_input,
    user_embedding=user_embedding,
    active_topic_start_node_id=self.current_topic[1] if self.current_topic else None,
    cooldown_turns=self.reactivation_cooldown_turns,
    now=datetime.now(),
    recent_nodes=recent_nodes
)

# 3. If REACTIVATE: set current_topic BEFORE building messages
if decision.action == "REACTIVATE":
    self.set_current_topic(decision.topic_name, decision.topic_start_node_id)
    self.reactivation_cooldown_turns = 3

# 4. Build messages via strategy (uses POST-reactivation current_topic)
result = self.context_builder.build_with_strategy(
    user_node_id=user_node_id,
    user_input=user_input,
    active_topic_start_node_id=self.current_topic[1] if self.current_topic else None,
    reactivation_decision=decision.action,
    user_embedding=user_embedding,
    token_budget=config.get("context_token_budget", 4000)
)
messages = result.messages

# 5. Call LLM with those messages
response = llm_query(messages, ...)

# 6. Topic boundary handling with override if reactivation fired
if decision.action == "REACTIVATE":
    self.topic_handler.detect_and_handle_topic_change(
        ..., decision_override="FORCE_CONTINUE"
    )
else:
    self.topic_handler.detect_and_handle_topic_change(...)
```

### Both Code Paths
The strategy must be used in:
- Normal path (full LLM response)
- `skip_llm_response` path (testing mode)

---

## Population Hooks

### Per-Turn Hook
After each exchange, nodes are added to `topic_nodes`:

```python
# In ConversationManager.handle_chat_message(), after topic boundaries handled:
self.add_nodes_to_current_topic(user_node_id, assistant_node_id)
```

### On Topic Creation
When a new topic is created, its initial nodes are added:

```python
# In TopicHandler.check_and_create_first_topic():
add_nodes_to_topic_range(topic_start_node_id, first_node_id, last_node_id)
ensure_topic_working_set(topic_start_node_id, topic_name)
```

### Backfill for Existing Data
```bash
python -m episodic.migrations.m015_topic_local_tables backfill
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_recovery_mode` | `"ancestry"` | `ancestry`, `topic_local`, or `hybrid` |
| `context_token_budget` | `4000` | Max tokens for assembled context |
| `enable_topic_reactivation` | `false` | Enable reactivation probe |
| `reactivation_cooldown` | `3` | Turns to wait after reactivation |
| `reactivation_dormancy_min` | `4` | Min turns before topic eligible |
| `reactivation_support_threshold` | `2` | Min matching exchanges required |

---

## Debugging

### Enable Debug Output
```bash
/set debug true
```

### Debug Persistence
Assembly debug info is logged per turn with:
- Mode used
- Active topic ID
- Included node IDs
- Token estimates
- Reactivation decision and reason
- Truncation info

### Inspect Topic Membership
```python
from episodic.db_topic_nodes import get_topic_nodes, count_topic_nodes

# See all nodes in a topic
nodes = get_topic_nodes("node_abc123", limit=50)

# Count by role
user_count = count_topic_nodes("node_abc123", role="user")
```

---

## Tests

### Unit Tests (tests/test_context_recovery.py)

**"B Disappears" Test**:
- Create conversation: Topic A (3 exchanges) → Topic B (2 exchanges) → reactivate A
- Assert: Assembled messages contain only A's node IDs
- Assert: No B content appears in messages

**"Year-Later" Test**:
- Topic with summary but empty recent exchanges
- Assert: Summary is included in context
- Assert: Assembler functions without recent context

**"No Summary Yet" Test**:
- Topic with exchanges but no summary
- Assert: Last N exchanges are included
- Assert: No error from missing summary

### Test Results
- Context Recovery: 13/13 ✅
- Topic Reactivation: 19/19 ✅

---

## Implementation Status

### Completed ✅
- [x] `topic_nodes` schema and migration
- [x] `topic_working_set` schema and migration
- [x] `topic_centroids` schema and migration
- [x] Population hooks (per-turn, topic creation)
- [x] Backfill function for existing topics
- [x] `ContextRecoveryMode` enum
- [x] `ContextAssemblyResult` dataclass
- [x] `ContextRecoveryStrategy` protocol
- [x] `AncestryStrategy` (wrapped existing behavior)
- [x] `TopicLocalStrategy` (topic-isolated assembly)
- [x] `select_strategy()` router
- [x] `build_with_strategy()` in ContextBuilder
- [x] Unit tests for all modes
- [x] "B disappears" verification
- [x] Integration in `conversation.py` main loop
- [x] Debug persistence to sqlite
- [x] Config sourcing for token budget
- [x] Anchor retrieval within topic (Chroma filter by `topic_start_node_id`)
- [x] Summary generation (populate `summary_md` via LLM)
- [x] Disambiguation UI (present options when DISAMBIGUATE returned)
- [x] Structured working set fields
- [x] Cross-topic imports (explicit "bring B context into A")
- [x] Thin topic fallback to ancestry
- [x] Timing spans and token breakdown in debug
- [x] Evaluation harness (50 labeled resume moments)
- [x] Calibration system with LOBO-CV
- [x] Hermetic Chroma test isolation

### All Phases Complete
See [CHANGELOG_TOPIC_REACTIVATION.md](./CHANGELOG_TOPIC_REACTIVATION.md) for full details.

---

## API Reference

### db_topic_nodes.py
```python
add_node_to_topic(topic_start_node_id, node_id, role, conn=None) -> bool
add_nodes_to_topic_range(topic_start_node_id, from_node_id, to_node_id=None, conn=None) -> int
get_topic_nodes(topic_start_node_id, limit=None, role=None, order="DESC", conn=None) -> List[Dict]
get_last_n_exchanges_in_topic(topic_start_node_id, n=4, conn=None) -> List[Dict]
get_node_topic(node_id, conn=None) -> Optional[str]
count_topic_nodes(topic_start_node_id, role=None, conn=None) -> int
ensure_topic_working_set(topic_start_node_id, topic_name, conn=None) -> bool
get_topic_working_set(topic_start_node_id, conn=None) -> Optional[Dict]
update_topic_summary(topic_start_node_id, summary_md, last_summarized_turn_idx, conn=None) -> bool
```

### context_recovery/strategy.py
```python
class ContextRecoveryMode(Enum): ANCESTRY, TOPIC_LOCAL, HYBRID
class ContextAssemblyResult: messages, debug
class ContextRecoveryStrategy(Protocol): assemble(...)
select_strategy(mode, reactivation_decision=None) -> ContextRecoveryStrategy
get_mode_from_config() -> ContextRecoveryMode
```

### context_recovery/topic_local.py
```python
class TopicLocalStrategy:
    def assemble(...) -> ContextAssemblyResult
```

### context_recovery/ancestry.py
```python
class AncestryStrategy:
    def assemble(...) -> ContextAssemblyResult
```
