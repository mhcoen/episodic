# Episodic Knowledge Graph Integration: Design Specification

## Overview

This document specifies the integration of a knowledge graph (KG) into Episodic, a conversational memory system built on an append-only DAG of conversation nodes, automatic topic segmentation, and context compression. The KG is a structured projection over the existing conversation substrate — not a new subsystem. It extracts typed entities and relations from conversation turns, stores them with provenance pointers back to the source utterances, and makes them available as an additional source during context assembly.

## Architectural Position

### What the KG is

A batch-computed, provenance-linked fact store derived from the conversation DAG. Each conversation turn may produce a small graph patch containing entity and relation extractions. The KG is the fold over all patches applied in turn order.

### What the KG is not

- Not a real-time extraction system. Extraction is batch-only and never runs inline with the conversation loop. Retrieval from the KG *is* inline — context assembly reads from the KG during the live conversation, but the KG itself is updated only by the batch processor.
- Not a profile store. It is not MemGPT-style CRUD over user attributes. It is a typed relational graph with provenance at edge granularity.
- Not a world-knowledge ontology. It extracts only what is explicitly asserted in conversation, not inferred or implied.

### Relationship to existing Episodic components

| Component | Relationship to KG |
|---|---|
| Conversation DAG | Source of truth. KG is derived from DAG nodes. |
| Topic segmentation | Provides locality for entity resolution. Entities are scoped to topics for resolution in Phase 0. |
| Compression | KG preserves discrete facts that compression summaries lose. KG is built from raw turns, never from compressed summaries. |
| Context assembly | KG is a new context source, sitting between ancestry and topic summaries in priority. |
| RAG | Independent system. KG indexes conversation content; RAG indexes external documents. |

## Batch Processing Architecture

### Scheduling

The KG builder runs as a periodic background task on a configurable timer, or on explicit user command. It never blocks the conversation loop.

- **Timer-based:** Configurable interval via `/set kg-interval <seconds>`. Less frequent than existing timed processes (e.g., news updates). Default TBD based on typical conversation velocity.
- **Manual trigger:** `/kg update` runs the batch processor immediately.
- **Full rebuild:** `/kg rebuild` reprocesses the entire conversation history from the first node.

### High-water mark

The batch processor maintains a persistent high-water mark: the `node_id` of the last successfully processed turn. Each run:

1. Reads the current high-water mark from the database.
2. Queries the DAG for all nodes with `node_id > high_water_mark`, ordered by sequence. Skips any node_ids in the skip list.
3. Processes each turn through the extraction pipeline.
4. For each successful patch: writes the patch, applies it, advances the high-water mark.
5. On failure: stops. The high-water mark remains at the last successful patch. Next run resumes from there.

No partial graph application — patches are either fully applied or not applied. No reprocessing unless explicitly requested via `/kg rebuild`.

### Skip list

A single pathological node can block all future KG progress. To handle this without requiring a full rebuild:

- `/kg skip <node_id>` records a skip list entry, marks that node's patch (if any) as `skipped_by_user`, and advances the high-water mark past it.
- Skip list entries are stored in `kg_state` and respected by all future batch runs.
- `/kg rebuild` ignores the skip list and reprocesses everything.

### Invariant: extraction from raw turns only

Compression does not delete DAG nodes — it adds summary nodes. The KG batch processor always reads the original turn nodes, never compressed summaries. This is a hard invariant. Compressed summaries are lossy; the KG's purpose is to preserve the discrete facts that summaries elide.

## Ontology (Phase 0)

Optimized for extraction precision, not expressive coverage. The guiding principle: every entity type and relation must be extractable with high precision from conversational text. Anything ambiguous becomes a Mention or a tagged Assertion.

### Bookkeeping layer

These are not domain objects. They are audit infrastructure.

**Assertion**
- `assertion_id`
- `source_node_id` (foreign key to conversation DAG)
- `span_start`, `span_end` (character offsets into the node's `content` field)
- `asserted_by`: `user` only in Phase 0. `assistant` and `tool` deferred to Phase 1.
- `polarity`: `affirm` or `negate`
- `certainty`: `explicit` or `hedged`
- `status`: `active` only in Phase 0. `deprecated` added in Phase 1.
- `tags`: list of labels (see Tags section below)

All entities and relations are anchored to Assertions. Every edge in the graph traces back to an exact span in a specific conversation turn.

### Entity types (4)

1. **PERSON** — the user, and any third parties mentioned by name.
2. **ARTIFACT** — concrete things: tools, apps, repos, files, devices, models, documents, products.
3. **TOPIC** — abstract areas: "privacy," "vector DBs," "prompt injection," "machine learning."
4. **ORG** — organizations, companies, groups. Optional but usually extractable when mentioned.

Not included in Phase 0: PROJECT, TASK, EVENT, PLACE. These require inference too often and collapse extraction precision.

**Reserved entity: the user.** A PERSON entity with `canonical_key = 'user:self'` is created at KG initialization and persists across all topics. All first-person assertions ("I use," "I prefer," "I'm a") attach edges to this entity. This prevents creation of multiple PERSON nodes for "I/me/my" references and provides a stable anchor for the user's identity across the conversation.

### Relations (4)

All relations must be supported by an explicit assertion in the source span. No inferred relations in Phase 0.

1. **USES(Person, Artifact)** — the person uses or has used the artifact. Must be supported by explicit language.
2. **WANTS(Person, X)** — the person wants, needs, is looking for, or is hoping for X. X can be ARTIFACT or TOPIC.
3. **PREFERS(Person, X)** — the person prefers X, optionally over something else. X can be ARTIFACT or TOPIC.
4. **ROLE(Person, Topic)** — the person's role, profession, or identity. Roles are modeled as TOPIC nodes to avoid a separate Role entity type.

**Mentions are not edges.** Entity mentions are tracked exclusively in the `kg_mentions` table, not as edge predicates. This is a clean separation: `kg_edges` contains only semantically committed relations; `kg_mentions` captures all entity references at the utterance level regardless of semantic commitment. When the extractor cannot confidently assign a typed relation, it records a mention only — no edge is created.

Everything that does not fit these relations becomes either:
- A mention record in `kg_mentions`
- A tagged Assertion without a typed edge

### Assertion tags

Lightweight labels attached to Assertions. These replace structured objects (Opinion, Proficiency, etc.) with simple categorical markers. A tag is valid only when the source span contains explicit lexical evidence.

**Sentiment tags** (only when explicit words appear):
- `SENTIMENT_POS` — "love," "great," "excellent," "helpful"
- `SENTIMENT_NEG` — "hate," "terrible," "frustrating," "annoying"
- `SENTIMENT_MIXED` — explicit mixed signals in the same span

**Proficiency tags** (only when explicitly stated):
- `PROFICIENCY_BEGINNER`
- `PROFICIENCY_INTERMEDIATE`
- `PROFICIENCY_EXPERT`

**Constraint tags** (only when explicitly stated):
- `CONSTRAINT_PRIVACY`
- `CONSTRAINT_COST`
- `CONSTRAINT_TIME`

Tags compose with entities: an assertion tagged `SENTIMENT_NEG` whose span mentions an ARTIFACT entity means "negative sentiment about that artifact." No separate sentiment model required.

**Tag-entity binding rule:** when a tagged assertion's span contains exactly one entity mention, the tag binds to that entity. When the span contains multiple entity mentions, the tag is unbound — it attaches to the assertion only, not to any specific entity. Disambiguation of multi-entity spans is deferred to Phase 1.

**Temporal tags:**
- `TIME_PAST` — explicit past-tense markers in the span ("used to," "previously," "back when")
- `TIME_PLANNED` — explicit future/intent markers ("going to," "plan to," "will")

These attach to assertions. They do not change the edge predicate. A USES edge with a `TIME_PAST` tag means "usage relationship, explicitly marked as past." Temporal qualifiers that modify the edge type itself are Phase 1.

## Extraction Pipeline

Three-stage pipeline. The LLM is involved only in stage 1. Stages 2 and 3 are deterministic.

### Stage 1: Candidate extraction (LLM-assisted)

Input:
- The new turn's text
- A bounded context window: preceding turns only (configurable, default 3), the current entity dictionary, and the KG neighborhood of any entities mentioned in recent turns.
- No lookahead. The extractor does not see turns after the current one. This simplifies replay semantics and avoids coupling patch content to future context that could theoretically be edited. Lookahead may be added in Phase 1 if extraction quality requires it.

Output:
- A proposed patch in structured JSON: entity upserts (create or add alias), relation additions, mention records, assertion records with tags.

The extraction prompt requests structured output (JSON mode or equivalent). The LLM proposes; it does not decide.

### Stage 2: Deterministic validation

Checks applied to every proposed patch:

- **Schema compliance:** entity types are in the allowed set, relation predicates are in the allowed set, asserted_by is `user` only (Phase 0).
- **Span validity:** `span_start` and `span_end` are within bounds of the source node's content. The substring at those offsets must exist in the turn text.
- **Provenance required:** every entity and relation must reference an assertion with valid span offsets. No provenance, no entry.
- **Entailment check (trigger tokens):** for each proposed edge, the assertion span must contain at least one trigger token for the proposed predicate. Trigger token lists per predicate:
  - `uses`: "use", "using", "used", "run", "running", "rely on", "my setup", "work with", "working with", "daily", "regularly"
  - `wants`: "want", "need", "looking for", "hoping", "wish", "interested in", "would like", "plan to"
  - `prefers`: "prefer", "rather", "instead of", "better than", "favorite", "go-to"
  - `role`: "I'm a", "I am a", "my role", "I work as", "my job", "my position", "by profession", "my background"
  - Matching is case-insensitive. This check is deterministic and catches the most common class of extraction error: the LLM proposing a typed relation when the span only supports a mention.
- **Canonical key uniqueness:** if a new entity proposes a canonical_key that already exists, it must resolve to the existing entity or be rejected.
- **No cross-source contamination:** edges cannot be created from assistant turns in Phase 0 (asserted_by='user' only).

Patches that fail validation are stored with `applied=0` and a rejection reason. They are never applied to the graph.

### Stage 3: Application

Valid patches are applied in a single SQLite transaction:
- Entity upserts (create new or add alias to existing)
- Assertion inserts
- Relation inserts
- Mention inserts
- High-water mark update

On transaction failure, nothing is written. The high-water mark does not advance.

### Patch record

Every patch is stored regardless of whether it was applied:

- `patch_id`
- `node_id` (the conversation turn that produced this patch)
- `patch_json` (canonical JSON serialization, stable key order)
- `patch_hash` (SHA-256 over canonical JSON)
- `validator_version` (version string of the validation code)
- `applied` (0 or 1)
- `rejection_reason` (null if applied, text if rejected)

Replay invariant: starting from an empty database and applying all patches where `applied=1` in `node_id` order must reproduce the current graph state exactly.

## Storage Schema (SQLite)

All tables in the existing Episodic SQLite database. No new database files by default. The KG storage path is configurable via `/set kg-db-path <path>`, defaulting to the main Episodic database. This allows future use cases (e.g., corpus analysis) to use a separate database file if needed.

**Prerequisite:** Episodic must enable `PRAGMA foreign_keys=ON` for all connections that interact with KG tables.

### `kg_entities`

```sql
CREATE TABLE kg_entities (
    entity_id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('person', 'artifact', 'topic', 'org')),
    canonical_key TEXT,
    canonical_name TEXT NOT NULL,
    created_node_id INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX uq_kg_entities_canonical_key ON kg_entities(canonical_key) WHERE canonical_key IS NOT NULL;
CREATE INDEX idx_kg_entities_type_name ON kg_entities(entity_type, canonical_name);
```

On KG initialization, a reserved user entity is inserted:
```sql
INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at)
VALUES ('person', 'user:self', '<user>', 0, <init_timestamp>);
```

### `kg_entity_aliases`

```sql
CREATE TABLE kg_entity_aliases (
    alias_id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    alias TEXT NOT NULL,
    source_node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    UNIQUE(entity_id, alias)
);
```

### `kg_assertions`

```sql
CREATE TABLE kg_assertions (
    assertion_id INTEGER PRIMARY KEY,
    source_node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    asserted_by TEXT NOT NULL CHECK(asserted_by IN ('user')),  -- Phase 0: user only
    polarity TEXT NOT NULL CHECK(polarity IN ('affirm', 'negate')),
    certainty TEXT NOT NULL CHECK(certainty IN ('explicit', 'hedged')),
    status TEXT NOT NULL CHECK(status IN ('active')),  -- Phase 1 adds 'deprecated'
    tags TEXT,  -- JSON array of tag strings
    UNIQUE(source_node_id, span_start, span_end)
);
CREATE INDEX idx_kg_assertions_node ON kg_assertions(source_node_id);
```

### `kg_edges`

```sql
CREATE TABLE kg_edges (
    edge_id INTEGER PRIMARY KEY,
    subj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    predicate TEXT NOT NULL CHECK(predicate IN ('uses', 'wants', 'prefers', 'role')),
    obj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    assertion_id INTEGER NOT NULL REFERENCES kg_assertions(assertion_id),
    UNIQUE(subj_entity_id, predicate, obj_entity_id, assertion_id)
);
CREATE INDEX idx_kg_edges_subj ON kg_edges(subj_entity_id);
CREATE INDEX idx_kg_edges_obj ON kg_edges(obj_entity_id);
CREATE INDEX idx_kg_edges_pred ON kg_edges(predicate);
```

### `kg_mentions`

```sql
CREATE TABLE kg_mentions (
    mention_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    surface_text TEXT NOT NULL,
    entity_id INTEGER REFERENCES kg_entities(entity_id),  -- NULL if unresolved
    confidence REAL NOT NULL,
    UNIQUE(node_id, span_start, span_end)
);
```

### `kg_patches`

```sql
CREATE TABLE kg_patches (
    patch_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL UNIQUE,  -- one patch per node; re-extraction requires /kg rebuild
    patch_json TEXT NOT NULL,
    patch_hash TEXT NOT NULL,
    validator_version TEXT NOT NULL,
    applied INTEGER NOT NULL CHECK(applied IN (0, 1)),
    rejection_reason TEXT
);
CREATE INDEX idx_kg_patches_node ON kg_patches(node_id);
```

**Design constraint:** the UNIQUE on node_id enforces exactly one patch per conversation turn. Re-extraction of a single node without a full rebuild is not supported in Phase 0. This simplifies replay determinism.

### `kg_state`

```sql
CREATE TABLE kg_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Stores: high_water_mark, last_run_timestamp, schema_version, skip_list (JSON array of node_ids)
```

### Purge invariant

When a conversation node is deleted or redacted from the DAG, all KG rows referencing that node_id must be deleted or tombstoned:
- `kg_assertions` where `source_node_id` matches
- `kg_edges` whose `assertion_id` references an affected assertion
- `kg_mentions` where `node_id` matches
- `kg_entity_aliases` where `source_node_id` matches
- `kg_patches` where `node_id` matches

Entities are not deleted even if all their edges are purged — they may be referenced by edges from other nodes. Orphan entity cleanup is a separate maintenance operation (`/kg prune`).

## Context Assembly Integration

### When KG facts enter context

During context assembly, after ancestry is resolved and before topic summaries are appended:

1. Extract entity mentions from the current user turn (lightweight NER or exact string match against the entity dictionary — no LLM call). Matching applies case-folding, Unicode normalization (NFC), and punctuation trimming to both the user turn and the entity dictionary entries. This is deterministic.
2. For each matched entity, retrieve its KG neighborhood: all edges where the entity is subject or object, limited to `status='active'` assertions.
3. Format as a labeled context block with provenance annotations.
4. Insert into the context window subject to the KG budget.

### Budget and priority

KG context has a configurable token budget: `/set kg-budget <tokens>`. Default TBD.

Priority within the context window (highest first):
1. System prompt and safety invariants
2. Active thread ancestry (existing behavior)
3. **KG facts matching entities in the current turn**
4. Topic summary blocks
5. Everything else

Within the KG budget, facts are ranked by:
1. Exact entity match to surface forms in the current turn
2. Recency (by `source_node_id` — Episodic's DAG guarantees monotonic node_id assignment)
3. `asserted_by='user'` over other sources (Phase 1)
4. `status='active'` over `status='deprecated'` (Phase 1)

### Staleness

If the high-water mark lags the current conversation by more than N turns (configurable), context assembly annotates the KG context block with a staleness indicator. Retrieval is not blocked, but the model is informed that KG facts may be incomplete for recent turns — entities mentioned after the high-water mark will not appear in KG context, and the model should not treat absence from the KG as evidence that something was not discussed.

## Entity Resolution (Phase 0)

Deliberately conservative. The resolution search space is precisely defined:

**Search space for resolving a mention in node N within topic T:**
1. The reserved `user:self` entity (always in scope)
2. All entities whose `canonical_key` matches exactly (global scope — canonical keys are unambiguous by definition)
3. All entities whose `canonical_name` or any alias matches the mention surface text exactly (case-folded, NFC-normalized) AND whose `created_node_id` falls within the same topic T
4. LLM-proposed resolutions to entities within topic T, subject to validator approval

**Rules:**
- **Exact canonical key match:** if a mention resolves to an existing canonical_key (URL, file path, email), it resolves to that entity regardless of topic scope. Automatic.
- **Exact string match within topic:** if a mention's surface text matches an existing entity's canonical_name or any alias within the same topic, it resolves to that entity. Automatic.
- **LLM-proposed resolution:** the extraction prompt may propose that a new mention maps to an existing entity within the same topic. The validator checks that the proposed entity exists, is within topic scope, and that the surface form is a plausible alias. If accepted, a new alias record is created.
- **No cross-topic resolution beyond exact canonical keys in Phase 0.** This bounds the blast radius of resolution errors to a single topic.
- **No SAME_AS edges in Phase 0.** Deferred to Phase 1 with user confirmation.

## CLI Surface

```bash
# Core commands
/kg                  # Show KG status: entity count, edge count, high-water mark, staleness
/kg update           # Run batch extraction now (from high-water mark)
/kg rebuild          # Reprocess entire conversation history
/kg skip <node_id>   # Skip a problematic node and advance high-water mark
/kg search <query>   # Search entities by name/alias

# Inspection
/kg entities         # List all entities
/kg entity <id>      # Show entity detail: aliases, edges, source assertions
/kg edges <entity>   # Show all edges for an entity
/kg patch <node_id>  # Show the patch generated for a specific turn

# Maintenance
/kg prune            # Remove orphan entities with no remaining edges

# Configuration
/set kg-auto true    # Enable timer-based extraction
/set kg-interval 3600  # Extraction interval in seconds
/set kg-budget 500   # Token budget for KG context in assembly
/set kg-context true # Enable KG facts in context assembly

# Visualization
/kg visualize        # Open interactive graph visualization
/kg visualize --save <file>  # Export to standalone HTML file
```

## Dynamic Visualization

Visualization is a core feature, not an add-on. The KG must be interactively explorable in a browser.

### Implementation

**Stack:** NetworkX for graph computation, Plotly for interactive rendering. Both are existing Episodic dependencies. Output is a standalone HTML file opened via PyWebView (also an existing dependency) or the system browser.

**Alternative:** If richer interactivity is needed (drag, expand/collapse neighborhoods, live filtering), use Pyvis (vis.js wrapper) instead of Plotly. Pyvis produces standalone HTML with physics-based force layout, hover tooltips, click-to-select, and zoom/pan out of the box. Single additional dependency.

### Visual encoding

**Node types distinguished by shape and color:**
- PERSON: circle, distinct color
- ARTIFACT: square/diamond
- TOPIC: triangle
- ORG: hexagon

**Node size:** proportional to degree (number of edges). High-degree entities are visually prominent.

**Edge types distinguished by color or dash pattern:**
- `uses`: solid
- `wants`: distinct color
- `prefers`: distinct color
- `role`: distinct color

**Hover text on nodes:** canonical name, entity type, alias list, creation turn.

**Hover text on edges:** predicate, source assertion text (the exact quoted span), source node ID, tags.

**Click-to-expand:** clicking a node highlights its neighborhood and dims everything else. Clicking again resets. This is native in Pyvis; requires custom callbacks in Plotly.

### Layout

Force-directed layout by default. High-degree entity nodes (tools, topics mentioned by many people) pull toward the center. Peripheral nodes (entities mentioned once) drift to the edges. The bipartite structure (person nodes connected to entity nodes) emerges naturally from force-directed placement without requiring explicit bipartite layout.

### Filtering

The visualization must support filtering by:
- Entity type (show only ARTIFACTs, hide TOPICs)
- Relation type (show only `uses` edges)
- Time range (show only entities/edges from turns within a node_id range)
- Tag (show only assertions tagged `SENTIMENT_NEG`)

Filtering can be implemented as:
- Pre-generation parameters in the `/kg visualize` command (`/kg visualize --type artifact --relation uses`)
- Interactive dropdowns/checkboxes in the HTML output (requires JavaScript; Pyvis supports custom HTML panels)

### Provenance drill-down

Clicking an edge or node should display the supporting assertion text — the exact span from the conversation that justifies this graph element. This is the audit trail made visible. Implementation: Pyvis tooltip or a side panel in the HTML that updates on selection.

### Multi-session / corpus mode

When Episodic is used to analyze external conversation corpora (survey responses, interview transcripts), the visualization should support:
- Color-coding nodes by session/respondent
- Filtering by session
- Identifying bridge entities (entities that appear across multiple sessions)

This is the same visualization engine with a different data scope. No separate implementation required.

### Update behavior

`/kg visualize` always renders the current state of the KG tables. If the KG has been updated since the last visualization, the new visualization reflects the updates. There is no persistent visualization state — it is generated fresh each time from the SQLite tables.

## Phase 1 Additions (Not In Scope for Phase 0)

- `asserted_by` expands to include `assistant` and `tool`
- `status` expands to include `deprecated`
- Deprecation edges with reason and provenance
- Cross-topic entity resolution via alias expansion and lightweight string similarity
- User-confirmed `SAME_AS` edges
- Temporal qualifiers on a subset of predicates (`uses`, `role`, `located_in`)
- Opinion-as-object with structured polarity/intensity
- Assertion promotion: Mention → typed edge via corroborating context

## Open Questions

1. **Extraction prompt specification.** What structured output format? What context window size for batch extraction? What retry/fallback policy for malformed LLM output?
2. **Entity dictionary maintenance.** How is the entity dictionary (used for context assembly NER) kept in sync with the KG tables? Materialized view? In-memory cache refreshed on batch completion?
3. **Interaction with Muse mode.** Should web search results contribute to the KG? If so, under what trust policy? (Likely Phase 1+.)
4. **Migration.** How does the KG schema get added to existing Episodic databases? Episodic already has `/migrate` — this should be a new migration step.
5. **Which LLM context runs extraction?** Should this use the chat model, detection model, or a dedicated `kg` model context? A dedicated context allows independent model selection and parameter tuning.
6. **Visualization library choice.** Plotly (already a dependency, good for static interactive plots, weaker on graph-specific interactions) vs. Pyvis (one new dependency, purpose-built for interactive network graphs, native force layout and click interactions). Pyvis is the stronger candidate but adds a dependency.
