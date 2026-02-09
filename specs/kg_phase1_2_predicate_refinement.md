# Phase 1.2: Predicate Refinement + Span Discipline

## Scope

Three new predicates, prompt projection rules, validator updates, read-side priority updates, and is_a span discipline tightening. No schema changes. No new tables. Edge dedup already shipped (Phase 1.1).

## Non-goals
- Do NOT change the KG schema (tables, columns, indices).
- Do NOT add new closure rules (read-side).
- Do NOT change the real-time extraction pipeline (`realtime.py`).
- Do NOT change the applicator (`applicator.py`) — it is predicate-agnostic.

---

## 1. New Predicates

### 1a. `studies(Person, Topic)`

**Semantics**: A person is studying, majoring in, or enrolled in a field/subject.

**Trigger phrases**: "studying X", "studies X", "majoring in X", "enrolled in X program", "taking X classes", "X major", "X student".

**Projection rule (CRITICAL)**: "studying X" and "studies X" MUST emit `studies`, NEVER `uses`. The phrase "studies at MIT" remains `located_at` (the object is an org, not a topic).

**Domain/Range**: subject must be `person`. Object must be `topic`.

### 1b. `affiliated_with(Person|Org, Org)`

**Semantics**: A person or org is associated with, from, or connected to an organization. Covers cases like "from Uplift", "associated with Google", "co-founded at Stanford" where the relationship is institutional affiliation rather than physical location or component membership.

**Trigger phrases**: "from X" (when X is an org), "affiliated with X", "associated with X", "connected to X", "represents X", "co-founder of X".

**Why not `part_of`**: `part_of` domain is `{artifact, org}` — no person subjects. `affiliated_with` fills the person→org affiliation gap.

**Why not `located_at`**: `located_at` implies current physical or institutional presence. `affiliated_with` is weaker — "from Uplift" means association, not necessarily current location.

**Domain/Range**: subject must be `person` or `org`. Object must be `org`.

### 1c. `works_on(Person, Artifact|Topic)`

**Semantics**: A person is actively building, developing, working on, or contributing to a project, system, or artifact.

**Trigger phrases**: "building X", "working on X", "developing X", "creating X", "contributing to X", "maintaining X", "hacking on X".

**Projection rule**: "building a marketplace" → `works_on`, not `uses` or `has`. The subject is the person doing the building. The object is the artifact/project being built.

**Domain/Range**: subject must be `person`. Object must be `artifact` or `topic`.

---

## 2. File Changes

### 2a. `prompt_template.py` — EXTRACTION_SYSTEM_PROMPT

Add to the "Predicate set and triggers" section, maintaining alphabetical insertion:

```
- AFFILIATED_WITH(Person|Org, Org) — from, affiliated with, associated with, connected to, represents, co-founder of
  Use when a person or org has an institutional association with an org. Do NOT use located_at for "from X" when it means affiliation rather than physical presence.
- STUDIES(Person, Topic) — studying, studies, majoring in, enrolled in, taking classes in, X major, X student
  CRITICAL projection rule: "studying X" and "studies X" MUST emit studies, NEVER uses. "Studies at MIT" remains located_at (object is org, not topic).
- WORKS_ON(Person, Artifact|Topic) — building, working on, developing, creating, contributing to, maintaining, hacking on
  Use when a person is actively building or developing something. Do NOT use uses or has for active development relationships.
```

Add to the "Mandatory domain and range constraints" section:

```
- studies: subject must be a person. object must be topic.
- affiliated_with: subject must be person or org. object must be org.
- works_on: subject must be a person. object must be artifact or topic.
```

Add to the "Allowed subject and object types per predicate" list:

```
- studies: subject must be a person. object must be topic.
- affiliated_with: subject must be person or org. object must be org.
- works_on: subject must be a person. object must be artifact or topic.
```

Update the worked example for "My daughter Emma studies computer science at MIT":

```
- My daughter Emma studies computer science at MIT
  Emit: user:self related_to Emma
  Emit: Emma located_at MIT
  Emit: Emma studies computer science       ← CHANGED from "uses"
  Do not emit: user:self located_at MIT
```

Update Worked Example 2 output to use `studies` instead of `uses`:
- The edge `{"subj_ref": "e1", "predicate": "uses", "obj_ref": "e2", ...}` becomes `{"subj_ref": "e1", "predicate": "studies", "obj_ref": "e2", ...}`

Add a new projection rules section AFTER the predicate triggers, BEFORE domain/range:

```
Projection rules (MANDATORY — override general triggers)
These rules take precedence over general trigger matching. Apply them first.

1. "studying X" / "studies X" / "majoring in X" / "enrolled in X" where X is a topic
   → MUST emit studies(Person, X). NEVER emit uses(Person, X).
   Exception: "studies at ORG" → located_at(Person, ORG) because object is org.

2. "from ORG" / "affiliated with ORG" where context indicates institutional association
   → MUST emit affiliated_with(Person, ORG). NEVER emit part_of(Person, ORG).
   Note: "from Paris" (a place, not an org) → skip or located_at if org-like.

3. "building X" / "working on X" / "developing X" where X is an artifact or project
   → MUST emit works_on(Person, X). NEVER emit uses(Person, X) or has(Person, X).
```

### 2b. `validator.py`

Update `ALLOWED_PREDICATES`:
```python
ALLOWED_PREDICATES = {
    'uses', 'wants', 'prefers', 'role', 'has', 'located_at',
    'part_of', 'related_to', 'is_a', 'powered_by',
    'studies', 'affiliated_with', 'works_on',  # Phase 1.2
}
```

Update `DOMAIN_RANGE`:
```python
'studies':        ({'person'},                     {'topic'}),
'affiliated_with':({'person', 'org'},              {'org'}),
'works_on':       ({'person'},                     {'artifact', 'topic'}),
```

### 2c. `context_source.py` — PREDICATE_PRIORITY

Update `PREDICATE_PRIORITY` dict. Insert new predicates with appropriate priority:
```python
PREDICATE_PRIORITY = {
    'has': 0,
    'is_a': 1,
    'located_at': 2,
    'affiliated_with': 3,   # Phase 1.2 — close to located_at
    'part_of': 4,
    'related_to': 5,
    'powered_by': 6,
    'role': 7,
    'studies': 8,            # Phase 1.2 — academic relation
    'works_on': 9,           # Phase 1.2 — active development
    'uses': 10,
    'prefers': 11,
    'wants': 12,
}
```

Rationale: `affiliated_with` is institutional context (high value, near `located_at`). `studies` and `works_on` are behavioral relations, ranked above `uses` but below structural facts.

---

## 3. is_a Span Discipline (Prompt-Level)

Add to the "Span discipline" section in the prompt:

```
is_a span discipline (MANDATORY)
For is_a(X, Y) edges, the assertion span MUST contain surface mentions of BOTH X and Y.
If X is mentioned in one clause and Y in another clause of the same sentence, create two
separate assertions and only emit the is_a edge from the assertion that contains both.

Example:
  "Biscuit loves the park. She's a golden retriever."
  Assertion a1: "Biscuit loves the park." — mention Biscuit only
  Assertion a2: "She's a golden retriever." — mention Biscuit (via "She"), golden retriever
  Edge: is_a(Biscuit, golden retriever) with source_assertion = a2 (contains both endpoints)

Do NOT emit is_a from a1 which only contains "Biscuit".
```

This is prompt-level guidance only. The validator already has mention-existence checks (9j) that will catch violations. No validator change needed.

---

## 4. Tests

Add to `tests/kg/test_kg_context.py` or create `tests/kg/test_kg_phase1_2.py`:

### T1: studies predicate accepted by validator
Create a patch with `predicate: "studies"`, subj_type=person, obj_type=topic.
Assert: validator passes (not stripped).

### T2: affiliated_with predicate accepted by validator
Create a patch with `predicate: "affiliated_with"`, subj_type=person, obj_type=org.
Assert: validator passes.

### T3: works_on predicate accepted by validator
Create a patch with `predicate: "works_on"`, subj_type=person, obj_type=artifact.
Assert: validator passes.

### T4: studies domain/range enforcement
Create a patch with `predicate: "studies"`, subj_type=artifact (wrong).
Assert: validator strips with domain_range_violation.

### T5: affiliated_with domain/range enforcement
Create a patch with `predicate: "affiliated_with"`, subj_type=artifact (wrong).
Assert: validator strips.

### T6: works_on domain/range enforcement
Create a patch with `predicate: "works_on"`, subj_type=org (wrong).
Assert: validator strips.

### T7: read-side predicate priority
Seed KG with edges using studies, affiliated_with, works_on predicates.
Call get_kg_context(). Assert: edges are returned and ranked per PREDICATE_PRIORITY.

### T8: studies in PREDICATE_PRIORITY dict
Assert all 13 predicates have entries in PREDICATE_PRIORITY. No KeyError on lookup.

---

## 5. Done Criteria

After implementation:
1. All new tests pass.
2. Full test suite: 0 regressions.
3. Run `python -m episodic.kg.batch --rebuild` (full rebuild).
4. Run audit: unique inter-entity edges not_wrong ≥ 95%.
5. wants_rate ≤ 15%.
6. No regression in mention completeness or strip accounting.

Report: test results, rebuild stats, audit numbers.

---

## 6. Implementation Order

1. Update `validator.py` (ALLOWED_PREDICATES, DOMAIN_RANGE) — smallest, most testable change
2. Update `context_source.py` (PREDICATE_PRIORITY) — no behavior change until edges exist
3. Update `prompt_template.py` (system prompt) — largest change, affects extraction quality
4. Write tests, run pytest
5. Full rebuild + audit
