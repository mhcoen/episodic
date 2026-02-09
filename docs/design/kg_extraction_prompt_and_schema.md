# KG Extraction: Prompt Template + JSON Schema (Phase 0)

This document defines:
1) The JSON object the extractor must emit (the “patch”).
2) The system prompt to drive extraction.

It is designed to work with the Phase 0 spec:
- Entities: person, artifact, topic, org
- Predicates: uses, wants, prefers, role
- Mentions are recorded in kg_mentions only (not an edge predicate)
- Every entity, edge, and mention must be backed by an assertion span (span_start/span_end into the source turn text)
- asserted_by is user only (Phase 0)

## Definitions

- Source turn text is the exact `content` string stored in the conversation DAG node.
- Offsets are 0-based character indices into that exact string.
- span_end is exclusive (Python slicing convention): `text[span_start:span_end]`.
- The extractor must never normalize or alter the source text when computing offsets.

## JSON Patch Schema (Extractor Output)

Top-level object keys are fixed and must be emitted in this order:
- schema_version
- node_id
- assertions
- entities
- aliases
- mentions
- edges
- notes

### 1) schema_version

String. Must be exactly:
- "kg_patch_v1"

### 2) node_id

Integer. The conversation DAG node id being processed.

### 3) assertions

Array of assertion objects. Each assertion object has:

- assertion_key: string
  - A local identifier unique within this patch, used for references by other objects.
  - Format: "a1", "a2", ...

- span_start: integer (0-based, inclusive)
- span_end: integer (0-based, exclusive)

- asserted_by: string
  - Must be exactly "user" in Phase 0.

- polarity: string enum
  - "affirm" or "negate"

- certainty: string enum
  - "explicit" or "hedged"

- status: string enum
  - Must be exactly "active" in Phase 0.

- tags: array of strings (may be empty)
  - Allowed tags:
    - SENTIMENT_POS, SENTIMENT_NEG, SENTIMENT_MIXED
    - PROFICIENCY_BEGINNER, PROFICIENCY_INTERMEDIATE, PROFICIENCY_EXPERT
    - CONSTRAINT_PRIVACY, CONSTRAINT_COST, CONSTRAINT_TIME
    - TIME_PAST, TIME_PLANNED

The extractor should create the minimum number of assertions necessary. If two edges come from distinct spans, use two assertions.

### 4) entities

Array of entity objects. Each entity object has:

- entity_key: string
  - Local identifier unique within this patch, used for references.
  - Format: "e1", "e2", ...

- entity_type: string enum
  - "person" | "artifact" | "topic" | "org"

- canonical_name: string
  - The normalized display name (not necessarily equal to the surface form).

- canonical_key: string or null
  - Use only when the text contains an explicit unambiguous key:
    - URLs, file paths, emails, repo slugs, model ids, etc.
  - Otherwise null.

- created_by_assertion: string
  - assertion_key that supports the existence of this entity mention in the source span.
  - Required even for canonical keys.

- resolution_hint: object
  - Optional, may be null.
  - If present:
    - kind: "map_to_existing" | "new_entity"
    - candidate_entity_id: integer (only if map_to_existing; otherwise null)
    - confidence: number in [0,1]
  - In Phase 0, the extractor may propose mapping only within topic scope, except canonical_key matches.

Special rule:
- The reserved user entity (canonical_key "user:self") is not created here. It must already exist in the DB.
- When the span is first-person ("I", "me", "my"), refer to the reserved entity by using the special reference "user:self" in edges/mentions (see references section below), not by emitting a new entity.

### 5) aliases

Array of alias objects (may be empty). Each alias object:

- entity_ref: entity reference (see references section)
- alias_text: string
- source_assertion: assertion_key
- span_start: integer
- span_end: integer

Only emit aliases when the span explicitly equates surface form to an entity name or when the surface form is a clear alternate name used by the speaker in the same span. If unsure, do not create aliases.

### 6) mentions

Array of mention objects (may be empty). Each mention object:

- mention_key: string
  - Local unique id: "m1", "m2", ...

- span_start: integer
- span_end: integer
- surface_text: string
  - Must equal `source_text[span_start:span_end]` exactly.

- entity_ref: entity reference or null
  - If resolved confidently, provide entity_ref.
  - If unresolved, null.

- confidence: number in [0,1]
  - Reflects resolution confidence, not “did the mention exist.”

- source_assertion: assertion_key
  - The assertion whose span contains this mention.

Every resolved entity that participates in an edge must also appear in mentions.

### 7) edges

Array of edge objects (may be empty). Each edge object:

- subj_ref: entity reference
- predicate: string enum
  - "uses" | "wants" | "prefers" | "role"

- obj_ref: entity reference
- source_assertion: assertion_key
- confidence: number in [0,1]

Edge creation rules:
- Edges are only for semantically committed relations.
- If relation is uncertain, record mentions only (no edge).
- All edges must be supported by explicit trigger tokens in the assertion span (validator checks this).

### 8) notes

Optional string. The extractor may include short notes for debugging, but must not include additional facts not represented in structured fields.

## Entity Reference Format

Any field named *_ref or entity_ref must be one of:

A) Reserved user:
- "user:self"

B) Local entity key defined in this patch:
- "e1", "e2", ...

C) Existing DB entity id (only if mapping is proposed):
- "db:<integer>"
Example: "db:42"

The validator may reject "db:<id>" mappings that are out of topic scope unless the entity has an exact canonical_key match.

## System Prompt Template (Extractor)

The following is the system prompt text for the extraction model. It must be used verbatim except for bracketed inserts.

SYSTEM PROMPT:

You are a precise information extractor for a provenance-linked knowledge graph.
Your job is to propose a JSON patch for a single conversation turn.

Hard requirements:
- Output MUST be valid JSON and MUST conform exactly to the schema described below.
- Do NOT output any text outside the JSON.
- All offsets are 0-based character indices into the provided source text string.
- span_end is exclusive. The substring text[span_start:span_end] must match surface_text exactly.
- Never invent facts. Only extract what is explicitly asserted in the source span.
- If you are unsure whether a relation holds, record only mentions (no edge).
- In Phase 0, asserted_by MUST be "user" and you MUST ignore assistant/tool turns entirely.
- Do not infer cross-topic identity merges. Only propose mapping to an existing entity if:
  1) canonical_key matches exactly, or
  2) the user uses the exact same name within the same topic scope (when topic scope info is provided).

Trigger discipline:
- Only create an edge if the assertion span contains explicit trigger language for the predicate.
- If trigger is absent, do not create the edge.

Tag discipline:
- Emit SENTIMENT_* only if explicit sentiment words appear.
- Emit PROFICIENCY_* only if explicitly stated.
- Emit CONSTRAINT_* only if explicitly stated.
- Emit TIME_* only if explicit markers appear.

Reserved user identity:
- For first-person references ("I", "me", "my"), use subj_ref = "user:self".
- Do not create a new PERSON entity for the speaker.

Inputs you will receive:
- node_id: integer
- source_text: string (the exact stored turn text for this node)
- recent_context: array of strings (preceding turns only; may be empty)
- entity_dictionary: array of existing entities in-scope, each with:
  - entity_id, entity_type, canonical_name, canonical_key, aliases
- kg_neighborhood: optional summary of edges for recently mentioned entities

Your output must be a single JSON object with keys in this exact order:
schema_version, node_id, assertions, entities, aliases, mentions, edges, notes

JSON schema reminder:
[Insert the concise schema summary from this doc, or include a JSON Schema document out-of-band. Do not include it in the model output.]

## Model Input Wrapper (Recommended)

At call time, provide a single JSON input message (as the user content to the extractor), e.g.:

- node_id: ...
- source_text: ...
- recent_context: [...]
- entity_dictionary: [...]
- kg_neighborhood: [...]

This keeps the extractor deterministic and makes logging easy.

## Retry / Fallback Policy (Deterministic)

Because the system is replay/audit oriented, retries must be deterministic:

- Attempt 1: run extractor with this prompt and inputs.
- If output is not parseable JSON or violates the patch schema (structural):
  - Attempt 2: rerun once with identical inputs and an added instruction:
    - "Your previous output was invalid JSON or schema-invalid. Output ONLY valid JSON matching the schema."
- If Attempt 2 still fails:
  - Mark patch rejected with reason "extractor_output_invalid_json_or_schema".
  - Do not apply partial results.
  - Batch processor stops unless operator uses /kg skip.

Optional escalation (if enabled):
- If both attempts fail and kg_escalate_model is configured:
  - Rerun Attempt 2 using the stronger model, recording model_id in patch metadata.
  - This must be a fixed deterministic rule to preserve replay.

## Minimal Worked Example (Illustrative)

Given source_text:
"I use ChromaDB and I prefer SQLite over Postgres."

Expected extractor behavior:
- assertions: two assertions or one assertion spanning both clauses (either is acceptable if spans are correct)
- mentions: ChromaDB, SQLite, Postgres
- edges:
  - user:self uses ChromaDB (if trigger "use" is in span)
  - user:self prefers SQLite (predicate prefers; object SQLite)
- No edge for Postgres unless explicitly stated as preferred or used; it may appear only as a mention (or as part of the prefers assertion span if you decide to model comparative preference in Phase 1)
