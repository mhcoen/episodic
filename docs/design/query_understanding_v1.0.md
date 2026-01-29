# Query Understanding Specification
## Phase 4 Completion for Episodic Retrieval System v1.1

---

## 1. Overview

Query understanding transforms natural language recall queries into structured retrieval parameters:

**Input:** Natural language string  
**Output:** `QueryIntent` with target, scopes, mode, and metadata

```python
@dataclass
class QueryIntent:
    target: str                           # Search terms (may be empty for browse)
    segment_scope: Optional[List[str]]    # None=all, []=failed resolution, [ids]=filter
    temporal_scope: Optional[Tuple[datetime, datetime]]  # Half-open [start, end)
    speaker_scope: Optional[str]          # None, "user", or "assistant"
    mode: Literal["answer", "browse", "summarize"]
    confidence: float                     # 0.0-1.0, extraction confidence
    ambiguity: Optional[str]              # If ambiguous, describe the ambiguity
```

---

## 1b. Recommended Canonical Forms

**Explicit command forms collapse ambiguity and should be documented to users:**

| Form | Mode | Examples |
|------|------|----------|
| `Browse: ...` | browse | `Browse: last 10 exchanges` |
| `Answer: ...` | answer | `Answer: when we discussed coffee, what did we decide?` |
| `Summarize: ...` | summarize | `Summarize: our discussion about BM25 yesterday` |
| `Show ...` | browse | `Show only what I said about segment cache last month` |
| `in topic X ...` | (adds scope) | `Browse: in topic "FTS migration" last week` |
| `topic:X ...` | (adds scope) | `Answer: topic:coffee what did we decide?` |

**Terse command-style variants:**
```
recall coffee yesterday
browse segment:fts-migration
summarize speaker:me topic:segment-cache last-week
answer 'bm25 orientation' date:2026-01-24
```

These forms are predictable and avoid the regex extraction risks of fully natural language.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Query                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Pattern Extractor                          │
│  - Mode detection (answer/browse/summarize)                  │
│  - Temporal phrase extraction                                │
│  - Segment/topic phrase extraction                           │
│  - Speaker phrase extraction                                 │
│  - Target extraction (remainder after scope removal)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Temporal        │  │ Segment         │  │ Speaker         │
│ Resolver        │  │ Resolver        │  │ Resolver        │
│                 │  │                 │  │                 │
│ "yesterday" →   │  │ "coffee topic"→ │  │ "I said" →      │
│ [start, end)    │  │ [node_ids]      │  │ "user"          │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       QueryIntent                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Retrieval Pipeline                         │
│                   (Already Implemented)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Pattern Extractor

### 3.1 Mode Detection

**Priority order** (first match wins):

| Pattern | Mode | Examples |
|---------|------|----------|
| Explicit prefix | As stated | "Answer: ...", "Browse: ...", "Summarize: ..." |
| "summarize", "summary of" | summarize | "summarize our coffee discussion" |
| "show", "list", "browse", "pull up", "display" | browse | "show the last 10 messages" |
| "what were we", "what did we" (deictic) | browse | "what were we just talking about?" |
| Default | answer | "when we discussed coffee, what did we decide?" |

**Regex patterns:**
```python
MODE_PATTERNS = [
    (r'^(?:answer|ans):\s*', 'answer'),
    (r'^(?:browse|show|list):\s*', 'browse'),
    (r'^(?:summarize|summary):\s*', 'summarize'),
    (r'\b(?:summarize|summary of|summarise)\b', 'summarize'),
    (r'^(?:show|list|browse|pull up|display)\b', 'browse'),
    (r'^what (?:were|was|did) (?:we|I|you)\b', 'browse'),
]
```

### 3.2 Temporal Phrase Extraction

**Patterns to extract** (removed from target, passed to resolver):

| Pattern | Examples |
|---------|----------|
| Relative day | "yesterday", "today", "the day before yesterday" |
| Relative period | "last week", "this week", "last month", "past 3 days" |
| Named day | "on Monday", "last Tuesday", "this Friday" |
| Explicit date | "on January 24", "on 2026-01-24", "Jan 24" |
| Date range | "between Jan 13 and Jan 20", "from Monday to Wednesday" |
| Recency | "just now", "recently", "earlier today", "this morning" |
| In-phrase | "in the last exchange", "in the last 10 messages" |

**CRITICAL: Month name whitelist to avoid false matches**

The date regex MUST be constrained to actual month names to avoid matching phrases like "segment 1", "phase 4", "BM25 999":

```python
MONTH_NAMES = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
DAY_NAMES = r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
```

**Regex patterns:**
```python
TEMPORAL_PATTERNS = [
    # Explicit ranges (month names required)
    (r'between\s+(' + MONTH_NAMES + r'\s+\d{1,2})\s+and\s+(' + MONTH_NAMES + r'\s+\d{1,2})', 'range'),
    (r'from\s+(' + DAY_NAMES + r')\s+to\s+(' + DAY_NAMES + r')', 'day_range'),
    
    # Relative periods (unambiguous keywords)
    (r'\b(yesterday|today|the day before yesterday)\b', 'relative_day'),
    (r'\b(last|this|past)\s+(week|month|day|\d+\s+days?)\b', 'relative_period'),
    (r'\b(earlier\s+today|this\s+morning|this\s+afternoon|this\s+evening)\b', 'time_of_day'),
    
    # Named days (require prefix to avoid bare weekday in content)
    (r'\b(?:on|last|this)\s+(' + DAY_NAMES + r')\b', 'named_day'),
    
    # Explicit dates (MONTH NAME REQUIRED - not arbitrary word+number)
    (r'\b(?:on\s+)?(' + MONTH_NAMES + r'\s+\d{1,2}(?:,?\s+\d{4})?)\b', 'explicit_date'),
    (r'\b(\d{4}-\d{2}-\d{2})\b', 'iso_date'),
    
    # Deictic message counts (not temporal, sets deictic_limit)
    (r'\b(?:the\s+)?last\s+(\d+)\s+(messages?|exchanges?)\b', 'deictic_count'),
    (r'\b(?:the\s+)?last\s+exchange\b', 'deictic_one'),
    
    # Recency keywords
    (r'\b(just\s+now)\b', 'recency_immediate'),
    (r'\b(recently)\b', 'recency_recent'),
    (r'\b(earlier)\b', 'recency_earlier'),
]
```

**Deictic defaults:**

| Phrase | deictic_limit | temporal_scope |
|--------|---------------|----------------|
| "just now" / "last exchange" | 1 | None |
| "recently" | 10 | None |
| "earlier today" | None | today |
| "last N messages" | N | None |

### 3.3 Segment/Topic Phrase Extraction

**CRITICAL: Only explicit scope language triggers segment resolver**

Users saying "when we discussed coffee yesterday" mean `target=coffee` with temporal filter, NOT `segment_scope=coffee_topic`. Treating casual phrasing as segment scope will invoke the resolver and potentially return `[]` (empty), killing the query unexpectedly.

**Rule:** Segment resolver ONLY triggers on explicit scope keywords:
- "in topic/segment X"
- "within topic/segment X"
- "topic:X" or "segment:X" (command-style)

**NOT segment scope (these set target instead):**
- "when we discussed X" → target=X
- "our discussion about X" → target=X
- "about X" → target=X

**Patterns that trigger segment resolver:**

| Pattern | Examples |
|---------|----------|
| Explicit scope | "in the 'retrieval' segment", "in topic coffee-brewing" |
| Within scope | "within topic FTS migration", "within the segment about caching" |
| Command-style | "topic:coffee", "segment:retrieval" |

**Regex patterns:**
```python
# ONLY these patterns trigger segment resolver
SEGMENT_SCOPE_PATTERNS = [
    # Explicit "in topic/segment X" (quoted or unquoted)
    (r"\bin\s+(?:the\s+)?(?:segment|topic)\s+['\"]?([^'\",.]+)['\"]?", 'explicit_in'),
    
    # Explicit "within topic/segment X"
    (r"\bwithin\s+(?:the\s+)?(?:segment|topic)\s+['\"]?([^'\",.]+)['\"]?", 'explicit_within'),
    
    # Command-style "topic:X" or "segment:X"
    (r"\b(?:topic|segment):([^\s,]+)", 'command_style'),
]

# These are TARGET patterns, NOT segment scope
# They extract X as search target, not topic filter
TARGET_INTENT_PATTERNS = [
    # "when we discussed X" -> target=X (NOT segment scope)
    r"when\s+we\s+discussed\s+([^,]+?)(?:,|\s+(?:yesterday|last|on|between|$))",
    
    # "our discussion about X" -> target=X
    r"(?:our|the)\s+discussion\s+(?:about|of)\s+['\"]?([^'\",.]+)['\"]?",
    
    # "the thread where/about X" -> target=X
    r"the\s+thread\s+(?:where|about)\s+(.+?)(?:,|$)",
]
```

**Disambiguation policy:**

| User says | Interpretation | Rationale |
|-----------|----------------|----------|
| "in topic coffee" | segment_scope=resolve("coffee") | Explicit scope keyword |
| "when we discussed coffee" | target="coffee" | No explicit scope keyword |
| "in topic coffee, when we discussed grinders" | segment_scope=resolve("coffee"), target="grinders" | Both present |

### 3.4 Speaker Phrase Extraction

**CRITICAL: Require speech verbs to avoid false positives**

"your" alone is too broad — it appears in normal content ("your approach", "your file"). Only match when combined with speech verbs: said, recommended, responded, wrote, mentioned, suggested.

**Patterns to extract:**

| Pattern | Resolves to | Examples |
|---------|-------------|----------|
| First person + speech verb | "user" | "I said", "I asked", "what did I say" |
| Second person + speech verb | "assistant" | "you said", "you recommended", "what did you say" |
| Explicit restrict | From context | "only show what I said", "restrict to your responses" |
| Explicit role label | As stated | "user messages only", "assistant responses only" |

**Regex patterns:**
```python
# Speech verbs that indicate actual utterances
SPEECH_VERBS = r'(?:said|asked|mentioned|wrote|typed|recommended|responded|suggested|explained|told)'

SPEAKER_PATTERNS = [
    # First person + speech verb (user)
    (r'\bI\s+' + SPEECH_VERBS, 'user'),
    (r'\bwhat\s+(?:did\s+)?I\s+(?:say|ask|mention|write)', 'user'),
    (r'\bonly\s+(?:show\s+)?what\s+I\s+' + SPEECH_VERBS, 'user'),
    
    # Second person + speech verb (assistant) - REQUIRES VERB
    (r'\byou\s+' + SPEECH_VERBS, 'assistant'),
    (r'\bwhat\s+(?:did\s+)?you\s+(?:say|recommend|respond|suggest|explain)', 'assistant'),
    (r'\bonly\s+(?:show\s+)?what\s+you\s+' + SPEECH_VERBS, 'assistant'),
    
    # Explicit role labels (unambiguous)
    (r'\buser\s+messages?\s+only\b', 'user'),
    (r'\bassistant\s+(?:messages?|responses?)\s+only\b', 'assistant'),
    (r'\brestrict(?:ed)?\s+to\s+(?:my|user)\s+messages', 'user'),
    (r'\brestrict(?:ed)?\s+to\s+(?:your|assistant)\s+(?:messages|responses)', 'assistant'),
    
    # "my messages" / "your responses" - only at phrase boundaries
    (r'\bmy\s+(?:messages?|questions?)\b', 'user'),
    (r'\byour\s+(?:responses?|answers?|replies?)\b', 'assistant'),
]

# NOTE: Bare "your" without speech verb is NOT matched
# "your approach" -> no speaker scope
# "you said" -> speaker_scope = assistant
```

**Mode interaction:** If speaker scope is set and mode wasn't explicit, set mode=browse:
```python
if speaker_scope and mode == 'answer':  # default mode
    mode = 'browse'  # speaker-scoped queries are naturally browse
```

### 3.5 Target Extraction

After removing mode prefix, temporal phrases, segment phrases, and speaker phrases:

1. Strip extracted phrases from input
2. Remove filler words: "the", "a", "an", "and", "or", "about", "regarding"
3. Collapse whitespace
4. Result is `target`

**Example:**
```
Input: "When we discussed coffee yesterday, what did you say about grinders?"

Extracted:
  - temporal: "yesterday"
  - segment: "coffee" (from "discussed coffee")
  - speaker: "assistant" (from "you say")
  - mode: "answer" (default, question form)

Target: "grinders"
```

---

## 4. Temporal Resolver

### 4.1 Input/Output

**Input:** Extracted temporal phrase + user timezone  
**Output:** Half-open UTC interval `[start, end)` or `None`

### 4.2 Resolution Rules

**User timezone:** Configured in `RETRIEVAL_CONFIG["temporal"]["timezone"]` (e.g., "America/Chicago")

| Phrase | Start (local) | End (local) |
|--------|--------------|-------------|
| "today" | 00:00:00 today | 00:00:00 tomorrow |
| "yesterday" | 00:00:00 yesterday | 00:00:00 today |
| "this week" | 00:00:00 Monday | 00:00:00 next Monday |
| "last week" | 00:00:00 prev Monday | 00:00:00 this Monday |
| "last N days" | 00:00:00 N days ago | 00:00:00 tomorrow |
| "on January 24" | 00:00:00 Jan 24 | 00:00:00 Jan 25 |
| "between Jan 13 and Jan 20" | 00:00:00 Jan 13 | 00:00:00 Jan 21 |
| "this morning" | 00:00:00 today | 12:00:00 today |
| "this afternoon" | 12:00:00 today | 18:00:00 today |
| "this evening" | 18:00:00 today | 00:00:00 tomorrow |
| "last N messages" | N/A (deictic, not temporal) | N/A |

**Named day resolution:**
- "Monday" → most recent Monday (past or today)
- "last Monday" → the Monday before the most recent
- "this Monday" → the Monday of current week

**After local resolution:** Convert to UTC using configured timezone.

### 4.3 Implementation

```python
def resolve_temporal(
    phrase: str,
    user_tz: str,
    reference_time: datetime  # Usually "now"
) -> Optional[Tuple[datetime, datetime]]:
    """
    Resolve temporal phrase to UTC half-open interval.
    
    Returns None if phrase not recognized.
    Returns (start_utc, end_utc) where start <= t < end.
    """
```

### 4.4 Deictic Message Counts

"Last N messages/exchanges" is NOT temporal — it's a result limit applied post-retrieval:

```python
@dataclass
class QueryIntent:
    # ... existing fields ...
    deictic_limit: Optional[int]  # "last 10 messages" → 10
```

---

## 5. Segment Resolver

### 5.1 Input/Output

**Input:** Extracted segment/topic phrase  
**Output:** `List[str]` of node IDs, or `[]` if resolution fails

### 5.2 Resolution Strategy

1. **Exact match:** Topic name equals phrase (case-insensitive)
2. **Contains match:** Topic name contains phrase
3. **Fuzzy match:** Jaccard similarity on tokens > 0.5
4. **Semantic match:** Embedding similarity > threshold (uses cached topic embeddings)

**Combined scoring:**
```python
score = 0.4 * lexical_score + 0.6 * semantic_score
```

**Tiebreaker:** `(score DESC, topic.id DESC)` — prefer higher score, then more recent topic

**Threshold:** If best score < `segment_score_threshold` (default 0.3), return `[]`

### 5.3 Implementation

```python
def resolve_segment(
    conn: sqlite3.Connection,
    phrase: str,
    embedding_cache: Dict[int, List[float]]  # topic_id → embedding
) -> Optional[List[str]]:
    """
    Resolve segment phrase to node IDs.
    
    Returns None if phrase is empty/None (no segment scope requested).
    Returns [] if resolution attempted but failed (empty result).
    Returns [node_ids] if resolution succeeded.
    """
```

### 5.4 Ambiguity Handling

If multiple topics score above threshold and within 0.1 of each other:

1. Set `QueryIntent.ambiguity` to describe the options
2. Use the highest-scoring one
3. Example: `"Ambiguous: 'coffee' matches 'coffee-brewing' (0.85) and 'coffee-machines' (0.78)"`

---

## 6. Speaker Resolver

### 6.1 Input/Output

**Input:** Extracted speaker phrase  
**Output:** `"user"`, `"assistant"`, or `None`

### 6.2 Resolution

Direct mapping from extracted pattern (see 3.4). No complex resolution needed.

```python
def resolve_speaker(phrase: str) -> Optional[str]:
    """Returns 'user', 'assistant', or None."""
```

---

## 7. Full Extraction Pipeline

```python
def understand_query(
    query: str,
    conn: sqlite3.Connection,
    config: dict
) -> QueryIntent:
    """
    Parse natural language query into structured intent.
    
    Steps:
    1. Detect and remove mode prefix
    2. Extract temporal phrases
    3. Extract segment phrases
    4. Extract speaker phrases
    5. Remainder is target
    6. Resolve temporal → UTC interval
    7. Resolve segment → node IDs
    8. Resolve speaker → role string
    9. Return QueryIntent
    """
```

---

## 8. Examples

### Example 1: Simple

**Input:** `"coffee"`

**Extraction:**
- Mode: answer (default)
- Temporal: None
- Segment: None
- Speaker: None
- Target: "coffee"

**Intent:**
```python
QueryIntent(
    target="coffee",
    segment_scope=None,
    temporal_scope=None,
    speaker_scope=None,
    mode="answer",
    confidence=0.9
)
```

### Example 2: Temporal + Target (NOT segment scope)

**Input:** `"When we discussed coffee yesterday, what did we decide?"`

**Extraction:**
- Mode: answer (question form)
- Temporal: "yesterday"
- Segment: None ("discussed coffee" is target-intent, NOT segment scope)
- Speaker: None
- Target: "coffee decide" (extracted from discussion reference + question)

**Resolution:**
- Temporal: `[2026-01-24T06:00:00Z, 2026-01-25T06:00:00Z)` (CST→UTC)
- Segment: None (no explicit "in topic/segment" keyword)

**Intent:**
```python
QueryIntent(
    target="coffee decide",
    segment_scope=None,  # NOT resolved - no explicit scope keyword
    temporal_scope=(datetime(2026,1,24,6,0,0,tzinfo=UTC), datetime(2026,1,25,6,0,0,tzinfo=UTC)),
    speaker_scope=None,
    mode="answer",
    confidence=0.85
)
```

**Key point:** "when we discussed coffee" sets target, not segment_scope. This prevents silent empty results when there's no topic literally named "coffee".

### Example 2b: Explicit Segment Scope

**Input:** `"In topic coffee-brewing yesterday, what did we decide?"`

**Extraction:**
- Mode: answer
- Temporal: "yesterday"
- Segment: "coffee-brewing" (explicit "in topic" keyword)
- Target: "decide"

**Resolution:**
- Temporal: `[2026-01-24T06:00:00Z, 2026-01-25T06:00:00Z)`
- Segment: `["U1", "A1", "U2", "A2"]` (coffee-brewing topic nodes)

**Intent:**
```python
QueryIntent(
    target="decide",
    segment_scope=["U1", "A1", "U2", "A2"],  # Resolved because explicit "in topic"
    temporal_scope=(...),
    speaker_scope=None,
    mode="answer",
    confidence=0.9
)
```

### Example 3: Speaker Scope

**Input:** `"Only show what you said about BM25 and summarize it"`

**Extraction:**
- Mode: summarize (explicit)
- Temporal: None
- Segment: None
- Speaker: "assistant" (from "you said")
- Target: "BM25"

**Intent:**
```python
QueryIntent(
    target="BM25",
    segment_scope=None,
    temporal_scope=None,
    speaker_scope="assistant",
    mode="summarize",
    confidence=0.9
)
```

### Example 4: Mixed Scope

**Input:** `"In the segment about 'segment cache', during the last week, what did you say about effective_end invalidation?"`

**Extraction:**
- Mode: answer (question form)
- Temporal: "last week"
- Segment: "segment cache"
- Speaker: "assistant" (from "you say")
- Target: "effective_end invalidation"

**Resolution:**
- Temporal: `[2026-01-13T06:00:00Z, 2026-01-20T06:00:00Z)`
- Segment: `[...]` (nodes from segment-cache topic)

**Intent:**
```python
QueryIntent(
    target="effective_end invalidation",
    segment_scope=[...],
    temporal_scope=(...),
    speaker_scope="assistant",
    mode="answer",
    confidence=0.8
)
```

### Example 5: Deictic

**Input:** `"What were we just talking about?"`

**Extraction:**
- Mode: browse (deictic pattern)
- Temporal: None (but implies recency)
- Segment: None
- Speaker: None
- Target: "" (empty)
- Deictic: implied limit 1-3

**Intent:**
```python
QueryIntent(
    target="",
    segment_scope=None,
    temporal_scope=None,
    speaker_scope=None,
    mode="browse",
    confidence=0.95,
    deictic_limit=3
)
```

### Example 6: Explicit Message Count

**Input:** `"Summarize the last 10 messages"`

**Extraction:**
- Mode: summarize
- Temporal: None
- Segment: None
- Speaker: None
- Target: ""
- Deictic: 10

**Intent:**
```python
QueryIntent(
    target="",
    segment_scope=None,
    temporal_scope=None,
    speaker_scope=None,
    mode="summarize",
    confidence=0.95,
    deictic_limit=10
)
```

---

## 9. Confidence Scoring

Confidence reflects extraction quality:

| Factor | Impact |
|--------|--------|
| Explicit mode prefix | +0.1 |
| Clear temporal phrase | +0.05 |
| Quoted segment name | +0.1 |
| Ambiguous segment resolution | -0.15 |
| Multiple temporal phrases | -0.1 |
| Very short target (<3 chars) | -0.1 |
| No extractable scopes | baseline 0.7 |

**Usage:** Low confidence (<0.5) triggers clarification prompt instead of execution.

---

## 10. Debug/AUDIT Logging

**Add extraction trace for diagnosing "regex ate my target" bugs:**

```python
logger.debug("AUDIT: [extract] input=%r", query)
logger.debug("AUDIT: [extract] mode=%s by pattern %s", mode, mode_pattern)
logger.debug("AUDIT: [extract] temporal phrase=%r -> interval=%s", temporal_phrase, interval)
logger.debug("AUDIT: [extract] segment phrase=%r -> scope=%s", segment_phrase, scope_type)
logger.debug("AUDIT: [extract] speaker phrase=%r -> role=%s", speaker_phrase, role)
logger.debug("AUDIT: [extract] residual target=%r", target)
```

This trace should be emitted at DEBUG level on every query. It catches:
- Date regex matching non-dates ("segment 1")
- Segment resolver triggering unexpectedly
- Target becoming empty after stripping
- Wrong mode inference

---

## 11. Error Handling

### 10.1 Unparseable Temporal

If temporal phrase extracted but resolver fails:
- Set `temporal_scope = None`
- Add warning to `ambiguity`: "Could not parse temporal: '{phrase}'"
- Reduce confidence by 0.2

### 10.2 Failed Segment Resolution

If segment phrase extracted but no topic matches:
- Set `segment_scope = []` (triggers empty result per spec)
- Add to `ambiguity`: "No topic matching '{phrase}' found"

### 10.3 Conflicting Scopes

If multiple contradictory patterns match:
- Use the more specific one
- Log AUDIT warning

---

## 11. Configuration

```python
QUERY_UNDERSTANDING_CONFIG = {
    "segment_score_threshold": 0.3,
    "ambiguity_delta": 0.1,  # Topics within this of top score are ambiguous
    "min_confidence_to_execute": 0.5,
    "default_deictic_limit": 5,
    "timezone": "America/Chicago",
}
```

---

## 12. Module Structure

```
episodic/retrieval/
├── __init__.py
├── query_understanding/
│   ├── __init__.py
│   ├── extractor.py      # Pattern extraction
│   ├── temporal.py       # Temporal resolution
│   ├── segment.py        # Segment resolution (uses existing segment.py)
│   ├── speaker.py        # Speaker resolution
│   └── intent.py         # QueryIntent dataclass and understand_query()
├── pipeline.py           # Updated to accept QueryIntent
└── ...
```

---

## 13. Integration with CLI

The `/recall` command becomes a thin wrapper:

```python
def cmd_recall(query: str):
    intent = understand_query(query, conn, config)
    
    if intent.confidence < config["min_confidence_to_execute"]:
        print(f"I'm not sure I understood. {intent.ambiguity}")
        return
    
    results = retrieve(
        conn=conn,
        chroma=chroma,
        target=intent.target,
        segment_scope=intent.segment_scope,
        temporal=intent.temporal_scope,
        speaker=intent.speaker_scope,
        mode=intent.mode,
        max_results=intent.deictic_limit or config["max_results"],
        config=retrieval_config
    )
    
    format_and_display(results, intent.mode)
```

---

## 14. Test Cases

### Unit Tests for Extractor

```python
def test_mode_detection_explicit_prefix():
    assert extract_mode("Answer: coffee") == ("answer", "coffee")
    assert extract_mode("Browse: last week") == ("browse", "last week")
    assert extract_mode("Summarize: our discussion") == ("summarize", "our discussion")

def test_mode_detection_implicit():
    assert extract_mode("summarize our coffee discussion")[0] == "summarize"
    assert extract_mode("show the last 10 messages")[0] == "browse"
    assert extract_mode("what were we just talking about")[0] == "browse"
    assert extract_mode("what did we decide about coffee")[0] == "answer"

def test_temporal_extraction():
    assert extract_temporal("coffee yesterday") == ("yesterday", "coffee")
    assert extract_temporal("between Jan 13 and Jan 20 BM25") == ("between Jan 13 and Jan 20", "BM25")
    assert extract_temporal("last week segment caching") == ("last week", "segment caching")

def test_segment_extraction():
    assert extract_segment("in the 'retrieval' segment what") == ("retrieval", "what")
    assert extract_segment("when we discussed coffee, what") == ("coffee", "what")

def test_speaker_extraction():
    assert extract_speaker("what did you say about BM25") == ("assistant", "what about BM25")
    assert extract_speaker("only show what I said") == ("user", "only show")
```

### Unit Tests for Resolvers

```python
def test_temporal_resolver_yesterday(fx_base):
    # Reference: 2026-01-25 10:00 CST
    start, end = resolve_temporal("yesterday", "America/Chicago", reference)
    assert start == datetime(2026, 1, 24, 6, 0, 0, tzinfo=UTC)  # Midnight CST = 6am UTC
    assert end == datetime(2026, 1, 25, 6, 0, 0, tzinfo=UTC)

def test_segment_resolver_exact_match(fx_base):
    nodes = resolve_segment(conn, "coffee-brewing", cache)
    assert "U1" in nodes
    assert "A1" in nodes

def test_segment_resolver_fuzzy_match(fx_base):
    nodes = resolve_segment(conn, "coffee", cache)  # Partial match
    assert len(nodes) > 0

def test_segment_resolver_no_match(fx_base):
    nodes = resolve_segment(conn, "quantum physics", cache)
    assert nodes == []
```

### Integration Tests

```python
def test_full_query_understanding_simple():
    intent = understand_query("coffee", conn, config)
    assert intent.target == "coffee"
    assert intent.mode == "answer"
    assert intent.segment_scope is None

def test_full_query_understanding_complex():
    intent = understand_query(
        "When we discussed coffee yesterday, what did you say about grinders?",
        conn, config
    )
    assert intent.target == "grinders"
    assert intent.mode == "answer"
    assert intent.segment_scope is not None  # Resolved to coffee topic
    assert intent.temporal_scope is not None  # Yesterday
    assert intent.speaker_scope == "assistant"
```

---

## 15. Success Criteria

1. **Mode detection:** Correct mode for all example patterns
2. **Temporal extraction:** All listed temporal patterns recognized
3. **Temporal resolution:** Correct UTC intervals for relative and absolute dates
4. **Segment extraction:** Topic references extracted from various phrasings
5. **Segment resolution:** Exact, fuzzy, and semantic matching work
6. **Speaker extraction:** First/second person patterns recognized
7. **Target extraction:** Clean target after scope removal
8. **Confidence scoring:** Appropriate confidence for clear vs ambiguous queries
9. **Integration:** QueryIntent correctly drives retrieval pipeline
10. **Empty handling:** Failed resolution produces correct tri-state behavior
