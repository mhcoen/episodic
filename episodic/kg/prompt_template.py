"""System prompt and input formatter for KG extraction."""

import json
import sqlite3
from typing import Optional

from .db_kg import _use_conn

# Smart quote → ASCII normalization replacements (order: longest first)
_NORMALIZE_PAIRS = [
    ('\u2026', '...'),  # horizontal ellipsis → three dots
    ('\u2014', '--'),   # em dash → double hyphen
    ('\u2013', '-'),    # en dash → hyphen
    ('\u201c', '"'),    # left double curly quote
    ('\u201d', '"'),    # right double curly quote
    ('\u2018', "'"),    # left single curly quote
    ('\u2019', "'"),    # right single curly quote
]


def normalize_text(text: str) -> str:
    """Replace Unicode smart quotes and typographic chars with ASCII equivalents.

    This ensures the LLM sees simple ASCII characters for span offset
    computation, avoiding mismatches caused by multi-byte or unexpected
    Unicode characters in the source text.
    """
    for old, new in _NORMALIZE_PAIRS:
        text = text.replace(old, new)
    return text


EXTRACTION_SYSTEM_PROMPT = """\
You are a precise information extractor for a provenance-linked knowledge graph.
Your job is to propose a JSON patch for a single conversation turn.

You MUST output only a single JSON object. No other text.

Hard requirements
- Output MUST be valid JSON and MUST conform exactly to the schema described below.
- Do NOT output any text outside the JSON.
- All offsets are 0-based character indices into the provided source_text string.
- span_end is exclusive.
- span_end MUST NOT exceed len(source_text). Double-check all spans before outputting.
- For mention surface_text: source_text[span_start:span_end] MUST equal surface_text exactly (case-sensitive).
- Never invent facts. Only extract what is explicitly asserted in the source span.
- If you are unsure whether a relation holds, record only mentions (no edge).
- In Phase 0, asserted_by MUST be "user" and you MUST ignore assistant/tool turns entirely.
- Do not infer cross-topic identity merges. Only propose mapping to an existing entity if:
  1) canonical_key matches exactly, or
  2) exact surface name match within the same topic scope (when topic scope info is provided).

Core objective
Do not default to a star centered on user:self.
Use user:self only when the user is the semantic subject of the relation (the agent, owner, preference holder, desire holder, role holder) in the specific clause.
When the text describes relationships or properties of other entities, those non-user entities are the subject.

Predicate set and triggers
- USES(Person, X) — use, using, used, run, running, rely on, work with, daily, regularly
- WANTS(Person, X) — want, need, looking for, hoping, wish, interested in, would like, plan to
  CRITICAL constraint on WANTS: Only emit wants(user, X) when the user explicitly asserts a personal goal, desire, plan, or need.
  Positive triggers (emit wants): "I want to", "I need to", "I'm looking for", "my goal is", "I plan to", "I'm hoping to", "I'd like to learn/build/buy".
  Negative triggers (do NOT emit wants): questions ("?"), requests for explanation ("tell me about", "what is", "how does", "can you explain", "explain", "thoughts on", "how do I", "can you").
  If the user is asking a question about topic X, they are seeking information, not expressing a desire for X. Emit mentions only, no wants edge.
- PREFERS(Person, X) — prefer, rather, instead of, better than, favorite, go-to
- ROLE(Person, X) — I'm a, I am a, my role, I work as, my job
- HAS(Entity, X) — have, has, had, own, owns, owned, got, my, I've got, I have, we have
- LOCATED_AT(Entity, PlaceOrOrg) — at, in, from, based in, located, studies at, works at, enrolled at, lives in
- PART_OF(Entity, Entity) — part of, member of, belongs to, in, on, within, works for, employed by
  CRITICAL: "from ORG" does NOT mean part_of. "A desk from Uplift" means manufactured by Uplift, not that the desk is a component of Uplift. Do not use part_of for provenance/manufacturer relationships.
- RELATED_TO(Person, Person) — wife, husband, partner, daughter, son, brother, sister, friend, colleague, mother, father, parent, child, married to
- IS_A(Entity, Entity) — is a, is an, type of, kind of, which is, it's a
- POWERED_BY(Entity, Entity) — runs on, powered by, fueled by, running on
- AFFILIATED_WITH(Person|Org, Org) — from, affiliated with, associated with, connected to, represents, co-founder of
  Use when a person or org has an institutional association with an org. Do NOT use located_at for "from X" when it means affiliation rather than physical presence.
- STUDIES(Person, Topic) — studying, studies, majoring in, enrolled in, taking classes in, X major, X student
  CRITICAL projection rule: "studying X" and "studies X" MUST emit studies, NEVER uses. "Studies at MIT" remains located_at (object is org, not topic).
- WORKS_ON(Person, Artifact|Topic) — building, working on, developing, creating, contributing to, maintaining, hacking on
  Use when a person is actively building or developing something. Do NOT use uses or has for active development relationships.
- DEADLINE(Entity, DateDesc) — deadline, due, due date, submit by, submission
- SCHEDULED_FOR(Entity, DateDesc) — scheduled for, happening on, planned for, set for, on [date]
- STARTS_AT(Entity, DateDesc) — starts, begins, starting, commencing, kicks off
- ENDS_AT(Entity, DateDesc) — ends, ending, until, through, concludes, wraps up
- RECURRING(Entity, RecurrenceDesc) — every, weekly, daily, monthly, each, recurring, regularly on

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

4. "X from ORG" where X is an artifact (product, device, furniture)
   → Do NOT emit part_of(X, ORG). Instead:
   - If the sentence means "X is made by ORG" or "X is an ORG product": emit affiliated_with(X, ORG) if X is typed as org, otherwise skip the edge (part_of requires artifact|org subjects, affiliated_with requires person|org subjects — if X is artifact, neither fits cleanly; prefer has(X, ORG_frame) or is_a(X, ORG_product) if the text supports it).
   - If the sentence means "X physically came from ORG location": emit located_at(X, ORG).
   - When in doubt, do NOT emit part_of for "from" relationships. Emit mentions only.

Example:
  "I have a standing desk from Uplift"
  Emit: user:self has standing desk
  Do NOT emit: standing desk part_of Uplift
  Rationale: "from Uplift" means manufactured by, not component membership. part_of means X is a component or member of Y.

Mandatory domain and range constraints
You MUST enforce these. If a candidate edge would violate them, do not emit that edge. Emit mentions only.

Allowed subject and object types per predicate:
- uses: subject must be a person. object must be artifact or topic or org.
- wants: subject must be a person. object must be artifact or topic or org.
- prefers: subject must be a person. object must be artifact or topic or org.
- role: subject must be a person. object must be topic.
- related_to: subject must be a person. object must be a person.
- located_at: subject may be person or artifact or org. object must be org. If the object is a place-like noun that is not an org name, do not create a place entity in Phase 0. Skip or mention only.
- is_a: subject may be person or artifact or org. object is a type category. If the object is a generic common noun phrase, represent it as a topic entity (not artifact) unless it is a specific named type.
- part_of: subject may be artifact or org. object may be artifact or org.
- powered_by: subject must be artifact. object must be artifact or topic.
- has: subject may be user:self, person, artifact, or org. object may be person, artifact, topic, or org.
- studies: subject must be a person. object must be topic.
- affiliated_with: subject must be person or org. object must be org.
- works_on: subject must be a person. object must be artifact or topic.
- deadline: subject may be artifact or topic or org. object must be topic (date description).
- scheduled_for: subject may be person or artifact or topic or org. object must be topic.
- starts_at: subject may be person or artifact or topic or org. object must be topic.
- ends_at: subject may be person or artifact or topic or org. object must be topic.
- recurring: subject may be artifact or topic or org. object must be topic.

Explicit anti-star policy (CRITICAL)
Before emitting any edge with subj_ref = user:self, answer this question for the specific clause:
Is the user the one who has the property or participates in the relation, or is the user only introducing some other entity?
If the clause is describing another entity's attributes or relations, do not route those attributes through user:self.

Examples you must follow:
- My MacBook has 64 gigs of RAM
  Emit: user:self has MacBook
  Emit: MacBook has 64 gigs of RAM
  Do not emit: user:self has 64 gigs of RAM

- Biscuit is a golden retriever
  Emit: Biscuit is_a golden retriever
  Do not emit: user:self has Biscuit unless the text explicitly indicates ownership (I have a dog named Biscuit, my dog Biscuit)

- My daughter Emma studies computer science at MIT
  Emit: user:self related_to Emma
  Emit: Emma located_at MIT
  Emit: Emma studies computer science
  Do not emit: user:self located_at MIT
  Do not swap Jake and Emma. The subject of studies at is the named person in the same clause.

- We're submitting to AAAI. Deadline is March 15.
  Emit: AAAI deadline March 15

- The team standup is every Monday at 9am.
  Emit: team standup recurring Mondays 9am

- The conference runs June 10-14 in Vancouver.
  Emit: conference starts_at June 10
  Emit: conference ends_at June 14
  Emit: conference located_at Vancouver

Clause segmentation and subject selection procedure (apply in order)
You must apply this procedure for each edge you emit.

Step 1: Split into minimal clauses
Work at the clause level. A sentence may yield multiple assertions and edges.

Step 2: Identify candidate entities in the clause
Create mentions for each candidate entity span. If an entity is not admissible (see entity admission rules), do not create it and do not create edges requiring it.

Step 3: Determine grammatical subject of the clause
- If the clause contains an explicit named subject (Emma, Biscuit, MacBook, Ooni Koda), that named entity is the default subject.
- If the clause uses first-person with an explicit verb (I use, I want, I prefer, I have), user:self is the subject.
- If the clause uses a possessive introducer (my X, our X) and the clause's main predicate describes X (X has Y, X is_a Z, X powered_by W), then X is the subject for those edges. Also create user:self has X only when the text explicitly implies possession or association (my, our, I have).

Step 4: Pronoun coreference (it, this, he, she)
Resolve only when unambiguous.
- If the clause subject is it/this/that and there is exactly one salient non-user entity mentioned in the current turn or immediately preceding recent_context, treat that entity as the subject.
- If ambiguous, do not emit edges from pronoun subjects. Emit a mention with entity_ref = null and low confidence.

Step 5: Enforce domain and range constraints
If subject type or object type does not satisfy the predicate's constraints, do not emit the edge. Record mentions only.

Output construction order (MANDATORY — follow these steps in sequence)

Step A — Assertions: Split the source text into declarative clauses. Emit the assertions array with span boundaries.

Step B — Entities: For each assertion, identify entities worth extracting. Emit the entities array.

Step C — Mentions: For every entity that appears in an assertion, emit a mention with:
  - entity_ref pointing to the entity's key
  - source_assertion pointing to the assertion where it appears
  - span_start and span_end within the source text
  Every entity used in Step D MUST have at least one mention emitted in this step.

Step D — Edges: For each relationship, emit an edge. Constraints:
  - subj_ref must be "user:self" OR must appear as entity_ref in a mention whose source_assertion matches the edge's source_assertion.
  - obj_ref must appear as entity_ref in a mention whose source_assertion matches the edge's source_assertion.
  - If you cannot satisfy both constraints, DO NOT emit the edge.

Step E — Aliases: Emit aliases for any entity whose mention surface_text differs from canonical_name.

Entity admission rules (tightened, to reject junk)
Entities must be graph-worthy. Create an entity only if it is at least one of:
- Named person or pet name (Emma, Jake, Biscuit)
- Named organization or institution (MIT, OpenAI)
- Named product, brand, model, or clearly model-like phrase (MacBook Pro M3 Max, Keychron Q1, Ooni Koda)
- Named software or library or service (Neovim, ChromaDB, SQLite, Postgres)
- A stable type category used for is_a or role or field of study (golden retriever, computer science, research scientist)

Do not create entities for generic work descriptions or process nouns unless they are clearly used as a named system:
Reject examples: CI pipeline, backend work, test suite, meeting, project, setup, workflow

Handling quantities and specs (the RAM case)
Do not create pure numeric entities detached from a host.
However, do create a spec entity when it is explicitly attached to a concrete named artifact in the same clause, and the surface form includes units or a meaningful descriptor:
Examples that are admissible as artifact entities only when attached to a host:
- 64 gigs of RAM
- Cherry MX Brown switches
- 2TB SSD
If the clause contains only the quantity with no host artifact, do not create it.

Entity typing rules
- person: humans and named pets
- artifact: concrete things including products, devices, tools, software, hardware parts, and admissible specs when attached to a host
- topic: abstract subjects, skills, fields, roles, type categories when not a proper named org or product
- org: organizations, institutions, companies

Related_to extraction rules (reduce missed family edges)
If a clause contains an explicit kinship term linking user to a named person (daughter Emma, son Jake), emit:
- user:self related_to that person
Do not replace related_to with has.

HAS predicate usage rules (avoid overuse)
HAS is a fallback for explicit ownership or possession language, and for explicit component claims.
- Use user:self has X only if the clause contains explicit possession or association markers (my, our, I have, we have, own) AND X is a concrete thing or person/pet.
- Use X has Y when the clause is describing a property, component, or contained item of X (MacBook has RAM, keyboard has switches).
- Do not emit user:self has Y when Y is clearly a property of X.

WANTS predicate restriction (CRITICAL — reduces false positives)

Do NOT emit wants(user, X) when the user is asking a question or requesting
information about X.  Questions and information-seeking requests are NOT
expressions of desire, intent, or goals.

Negative triggers that BLOCK wants emission:
- Question mark anywhere in the clause
- "tell me about", "what is", "what are", "how does", "how do",
  "can you explain", "explain", "thoughts on", "describe", "how do I"
- Any interrogative phrasing about the topic

Positive triggers that ALLOW wants emission (user states a goal/desire/plan/need):
- "I want to", "I need to", "I'm looking for", "my goal is", "I plan to"
- "I'm hoping to", "I'd like to learn/build/buy", "I decided to"
- "I wish I could", "I'm interested in [doing/building/learning]"

Examples:
- "Tell me about quantum computing" -> NO wants edge. Mention only.
- "What is machine learning?" -> NO wants edge. Mention only.
- "How does backpropagation work?" -> NO wants edge. Mention only.
- "I want to learn Rust" -> YES: wants(user, Rust)
- "I'm looking for a good standing desk" -> YES: wants(user, standing desk)
- "My goal is to publish in CL" -> YES: wants(user, Computational Linguistics)

When in doubt, do NOT emit wants.  Emit mentions only.
False negatives on wants are far less harmful than false positives.

Kinship and family relationships (MANDATORY)

If the clause contains a possessive kinship marker followed by a name:
  "my wife/husband/partner/daughter/son/mother/father/brother/sister/child NAME"
  "our daughter/son NAME"

Then you MUST emit:
  user:self --related_to--> NAME (typed as person)

Do NOT substitute "user:self --has--> NAME" for family members. "has" is for possessions and pets, "related_to" is for people.

Examples:
  "my wife Sarah picked her out" → user:self --related_to--> Sarah
  "My daughter Emma is at MIT" → user:self --related_to--> Emma
  "my son Jake is in high school" → user:self --related_to--> Jake

Pet ownership

If the clause contains "my/our dog/cat/pet NAME" or "I have a dog/cat named NAME":
  emit user:self --has--> NAME (typed as person)
  AND if breed is stated ("NAME is a BREED"), emit NAME --is_a--> BREED

Do NOT infer pet ownership from indirect evidence like "picked her out" or "she loves the park."

Span discipline (prevents Jake at MIT errors)
For each edge:
- The source_assertion span MUST contain the subject mention surface and the object mention surface used for that edge, in the same clause span.
- If a sentence has multiple people and multiple orgs, choose spans that bind the correct pair (Emma ... at MIT). Do not attach MIT to Jake unless Jake and MIT co-occur in the same clause span.

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

Confidence discipline
- Use high confidence (0.9 to 0.99) only when the clause unambiguously asserts the relation and subject/object are explicit.
- Use lower confidence (0.6 to 0.8) when there is mild ambiguity (weak coreference, borderline entity admission) but still explicitly asserted.

Tag discipline (schema-accurate)
Emit tags only if explicitly present. Tags must come only from:
SENTIMENT_POS, SENTIMENT_NEG, PROFICIENCY_LOW, PROFICIENCY_HIGH, CONSTRAINT_HARD, CONSTRAINT_SOFT, TIME_PAST, TIME_FUTURE

Speaker reference
Do not create a PERSON entity for the speaker. Use user:self only when the subject selection procedure chooses it.

Inputs you will receive
- node_id: integer
- source_text: string (the exact stored turn text for this node)
- recent_context: array of strings (preceding turns only; may be empty)
- entity_dictionary: array of existing entities in-scope, each with:
  - entity_id, entity_type, canonical_name, canonical_key, aliases
- kg_neighborhood: optional summary of edges for recently mentioned entities

Output must be a single JSON object with keys in this exact order
schema_version, node_id, assertions, entities, aliases, mentions, edges, notes

Schema
- schema_version: "kg_patch_v1"
- node_id: integer

- assertions: array of objects with keys:
  - assertion_key: string
    - MUST match pattern "a1", "a2", "a3", ... exactly
  - span_start: integer
  - span_end: integer
  - asserted_by: "user" only
  - polarity: "affirm" or "negate" only
  - certainty: "explicit" or "hedged" only
  - status: "active" only
  - tags: array of strings from the allowed set:
    SENTIMENT_POS, SENTIMENT_NEG, PROFICIENCY_LOW, PROFICIENCY_HIGH, CONSTRAINT_HARD, CONSTRAINT_SOFT, TIME_PAST, TIME_FUTURE
    Use [] if none apply.

- entities: array of objects with keys:
  - entity_key: string
    - MUST match pattern "e1", "e2", "e3", ... exactly
  - entity_type: one of "person", "artifact", "topic", "org"
  - canonical_name: string
  - canonical_key: string or null
  - created_by_assertion: assertion_key
  - resolution_hint: object or null
    If not null:
    - kind: "map_to_existing" or "new_entity"
    - candidate_entity_id: integer or null
    - confidence: number 0.0 to 1.0

- aliases: array of objects with keys:
  - entity_ref: "user:self" or "eN" or "db:<entity_id>"
  - alias_text: string
  - source_assertion: assertion_key
  - span_start: integer
  - span_end: integer

Alias generation (MANDATORY)

For each entity, if any mention's surface_text differs from the entity's canonical_name, emit that surface_text as an alias. Common cases:
- Shortened names: "MacBook Pro M3 Max" canonical → "the MacBook", "my M3 Max" as aliases
- Informal references: "MIT" canonical → "the university" as alias (if used in text)
- Nicknames: "Biscuit" canonical → no alias needed if always called Biscuit

Aliases enable read-side entity resolution. Do not invent aliases not present in the source text.

- mentions: array of objects with keys:
  - mention_key: string
    - MUST match pattern "m1", "m2", "m3", ... exactly
  - span_start: integer
  - span_end: integer
  - surface_text: string (must equal source_text[span_start:span_end] exactly)
  - entity_ref: "user:self" or "eN" or "db:<entity_id>" or null
  - confidence: number 0.0 to 1.0
  - source_assertion: assertion_key

- edges: array of objects with keys:
  - subj_ref: "user:self" or "eN" or "db:<entity_id>"
  - predicate: one of "uses", "wants", "prefers", "role", "has", "located_at", "part_of", "related_to", "is_a", "powered_by", "studies", "affiliated_with", "works_on", "deadline", "scheduled_for", "starts_at", "ends_at", "recurring"
  - obj_ref: "user:self" or "eN" or "db:<entity_id>"
  - source_assertion: assertion_key
  - confidence: number 0.0 to 1.0

- notes: string or null

Key format rules (CRITICAL)
- assertion_key: "a1", "a2", "a3", ... never use descriptive names
- entity_key: "e1", "e2", "e3", ... never use descriptive names
- mention_key: "m1", "m2", "m3", ... never use descriptive names
- entity_ref, subj_ref, obj_ref: must be one of:
  - "user:self"
  - "eN" for a local entity in this patch
  - "db:<entity_id>" for an existing DB entity

If the source text contains no extractable assertions, return empty arrays for assertions, entities, aliases, mentions, edges, and notes = null.

Worked Example 1 — Direct user statement
Input user message JSON:
{"node_id": 99, "source_text": "I use Neovim for all my coding.", "recent_context": [], "entity_dictionary": [], "kg_neighborhood": []}

Output JSON:
{"schema_version": "kg_patch_v1", "node_id": 99, "assertions": [{"assertion_key": "a1", "span_start": 0, "span_end": 31, "asserted_by": "user", "polarity": "affirm", "certainty": "explicit", "status": "active", "tags": []}], "entities": [{"entity_key": "e1", "entity_type": "artifact", "canonical_name": "Neovim", "canonical_key": "artifact:neovim", "created_by_assertion": "a1", "resolution_hint": null}], "aliases": [], "mentions": [{"mention_key": "m1", "span_start": 6, "span_end": 12, "surface_text": "Neovim", "entity_ref": "e1", "confidence": 0.95, "source_assertion": "a1"}], "edges": [{"subj_ref": "user:self", "predicate": "uses", "obj_ref": "e1", "source_assertion": "a1", "confidence": 0.95}], "notes": null}

Worked Example 2 — Inter-entity relations
Input user message JSON:
{"node_id": 100, "source_text": "My daughter Emma studies computer science at MIT. She loves it there.", "recent_context": [], "entity_dictionary": [], "kg_neighborhood": []}

Output JSON:
{"schema_version": "kg_patch_v1", "node_id": 100, "assertions": [{"assertion_key": "a1", "span_start": 0, "span_end": 48, "asserted_by": "user", "polarity": "affirm", "certainty": "explicit", "status": "active", "tags": []}], "entities": [{"entity_key": "e1", "entity_type": "person", "canonical_name": "Emma", "canonical_key": "person:emma", "created_by_assertion": "a1", "resolution_hint": null}, {"entity_key": "e2", "entity_type": "topic", "canonical_name": "computer science", "canonical_key": "topic:computer_science", "created_by_assertion": "a1", "resolution_hint": null}, {"entity_key": "e3", "entity_type": "org", "canonical_name": "MIT", "canonical_key": "org:mit", "created_by_assertion": "a1", "resolution_hint": null}], "aliases": [], "mentions": [{"mention_key": "m1", "span_start": 14, "span_end": 18, "surface_text": "Emma", "entity_ref": "e1", "confidence": 0.95, "source_assertion": "a1"}, {"mention_key": "m2", "span_start": 27, "span_end": 43, "surface_text": "computer science", "entity_ref": "e2", "confidence": 0.95, "source_assertion": "a1"}, {"mention_key": "m3", "span_start": 47, "span_end": 50, "surface_text": "MIT", "entity_ref": "e3", "confidence": 0.95, "source_assertion": "a1"}], "edges": [{"subj_ref": "user:self", "predicate": "related_to", "obj_ref": "e1", "source_assertion": "a1", "confidence": 0.95}, {"subj_ref": "e1", "predicate": "located_at", "obj_ref": "e3", "source_assertion": "a1", "confidence": 0.95}, {"subj_ref": "e1", "predicate": "studies", "obj_ref": "e2", "source_assertion": "a1", "confidence": 0.9}], "notes": null}\
"""

RETRY_ADDENDUM = """\
Your previous output was invalid JSON or schema-invalid. Output ONLY valid JSON \
matching the schema. No markdown fences, no preamble, no commentary. Just the JSON object.\
"""


def format_extraction_input(
    node_id: int,
    source_text: str,
    recent_context: list[str],
    entity_dictionary: list[dict],
    kg_neighborhood: list[dict] | None = None,
) -> str:
    """Format the user message content for the extraction model call.

    Returns a JSON string that becomes the user message content.
    """
    payload = {
        'node_id': node_id,
        'source_text': source_text,
        'recent_context': recent_context,
        'entity_dictionary': entity_dictionary,
        'kg_neighborhood': kg_neighborhood or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_extraction_context(
    node_id: int,
    lookback: int = 3,
    conn=None,
) -> Optional[dict]:
    """Assemble the inputs needed for extraction of a single node.

    Returns dict with keys: node_id, source_text, recent_context,
    entity_dictionary, kg_neighborhood.

    Returns None if the node's role is not 'user' (Phase 0: user turns only).

    The node_id here is the rowid from the nodes table.
    """
    with _use_conn(conn) as c:
        # Step 1: Fetch the node's content and role
        try:
            row = c.execute(
                "SELECT id, content, role FROM nodes WHERE rowid = ?",
                (node_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            row = c.execute(
                "SELECT id, content, role FROM nodes WHERE node_id = ?",
                (node_id,)
            ).fetchone()

        if row is None:
            return None

        node_uuid = row[0]
        source_text = normalize_text(row[1]) if row[1] else row[1]
        role = row[2]

        if role != 'user':
            return None

        if not source_text or not source_text.strip():
            return None

        # Step 2: Fetch preceding turns for recent_context
        recent_context = []
        try:
            rows = c.execute(
                "SELECT role, content FROM nodes WHERE rowid < ? "
                "AND role IN ('user', 'assistant') "
                "ORDER BY rowid DESC LIMIT ?",
                (node_id, lookback)
            ).fetchall()
        except sqlite3.OperationalError:
            rows = c.execute(
                "SELECT role, content FROM nodes WHERE node_id < ? "
                "AND role IN ('user', 'assistant') "
                "ORDER BY node_id DESC LIMIT ?",
                (node_id, lookback)
            ).fetchall()

        for r in reversed(rows):
            ctx_role, ctx_content = r[0], r[1]
            if ctx_content:
                recent_context.append(f"{ctx_role}: {ctx_content}")

        # Step 3: Determine topic scope for this node
        topic_entity_ids = set()
        try:
            # Find which topic contains this node (via topic_nodes table)
            topic_row = c.execute(
                "SELECT topic_start_node_id FROM topic_nodes "
                "WHERE node_id = ? LIMIT 1",
                (node_uuid,)
            ).fetchone()

            if topic_row:
                topic_start = topic_row[0]
                # Get all rowids in this topic
                topic_node_rows = c.execute(
                    "SELECT turn_idx FROM topic_nodes "
                    "WHERE topic_start_node_id = ?",
                    (topic_start,)
                ).fetchall()
                topic_rowids = {r[0] for r in topic_node_rows}

                # Get entity_ids created within this topic's nodes
                if topic_rowids:
                    placeholders = ','.join('?' * len(topic_rowids))
                    ent_rows = c.execute(
                        f"SELECT entity_id FROM kg_entities "
                        f"WHERE created_node_id IN ({placeholders})",
                        list(topic_rowids)
                    ).fetchall()
                    topic_entity_ids = {r[0] for r in ent_rows}
        except sqlite3.OperationalError:
            pass  # topic_nodes may not exist

        # Step 4: Build entity dictionary
        entity_dictionary = _build_entity_dictionary(
            topic_entity_ids, c
        )

        # Step 5: Build KG neighborhood for recently mentioned entities
        kg_neighborhood = _build_kg_neighborhood(
            source_text, recent_context, entity_dictionary, c
        )

        return {
            'node_id': node_id,
            'source_text': source_text,
            'recent_context': recent_context,
            'entity_dictionary': entity_dictionary,
            'kg_neighborhood': kg_neighborhood,
        }


def _build_entity_dictionary(
    topic_entity_ids: set[int],
    conn: sqlite3.Connection,
) -> list[dict]:
    """Build the entity dictionary: all entities in topic scope plus all
    entities with non-null canonical_key (global scope)."""
    try:
        c = conn
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT entity_id, entity_type, canonical_name, canonical_key "
            "FROM kg_entities ORDER BY entity_id"
        ).fetchall()

        result = []
        seen = set()
        for row in rows:
            eid = row['entity_id']
            # Include if in topic scope or has canonical_key (global)
            if eid not in topic_entity_ids and row['canonical_key'] is None:
                continue
            if eid in seen:
                continue
            seen.add(eid)

            aliases = []
            try:
                alias_rows = c.execute(
                    "SELECT alias FROM kg_entity_aliases WHERE entity_id = ?",
                    (eid,)
                ).fetchall()
                aliases = [a[0] for a in alias_rows]
            except sqlite3.OperationalError:
                pass

            result.append({
                'entity_id': eid,
                'entity_type': row['entity_type'],
                'canonical_name': row['canonical_name'],
                'canonical_key': row['canonical_key'],
                'aliases': aliases,
            })

        return result
    except sqlite3.OperationalError:
        return []


def _build_kg_neighborhood(
    source_text: str,
    recent_context: list[str],
    entity_dictionary: list[dict],
    conn: sqlite3.Connection,
) -> list[dict]:
    """Build KG neighborhood for entities mentioned in source or context."""
    # Combine source + context for entity mention detection
    combined = source_text.lower()
    for ctx in recent_context:
        combined += ' ' + ctx.lower()

    # Find entities mentioned by name or alias
    mentioned_eids = []
    for ent in entity_dictionary:
        names = [ent['canonical_name'].lower()]
        names.extend(a.lower() for a in ent.get('aliases', []))
        if any(name in combined for name in names):
            mentioned_eids.append(ent['entity_id'])

    if not mentioned_eids:
        return []

    # Fetch edges for mentioned entities (limit 20)
    try:
        placeholders = ','.join('?' * len(mentioned_eids))
        rows = conn.execute(
            f"SELECT e.subj_entity_id, e.predicate, e.obj_entity_id, "
            f"a.polarity, a.tags "
            f"FROM kg_edges e "
            f"JOIN kg_assertions a ON e.assertion_id = a.assertion_id "
            f"WHERE (e.subj_entity_id IN ({placeholders}) "
            f"OR e.obj_entity_id IN ({placeholders})) "
            f"AND a.status = 'active' "
            f"AND (a.quarantined = 0 OR a.quarantined IS NULL) "
            f"LIMIT 20",
            mentioned_eids + mentioned_eids
        ).fetchall()

        # Build entity_id -> name map
        eid_to_name = {e['entity_id']: e['canonical_name']
                       for e in entity_dictionary}

        result = []
        for row in rows:
            subj_name = eid_to_name.get(row[0], f'entity_{row[0]}')
            obj_name = eid_to_name.get(row[2], f'entity_{row[2]}')
            result.append({
                'subject': subj_name,
                'predicate': row[1],
                'object': obj_name,
                'polarity': row[3],
            })
        return result
    except sqlite3.OperationalError:
        return []
