# Query Understanding Specification v1.1
## Hybrid Two-Stage Parser for Episodic Retrieval

---

## 1. Overview

Natural language recall queries are parsed in two stages:

**Stage A (Deterministic):** Finite-state tokenizer handles explicit command forms. Zero ambiguity, 100% stable.

**Stage B (LLM Fallback):** Invoked only when Stage A leaves unparsed natural language. Strict schema validation; invalid parses rejected.

```
Input: "when we discussed coffee yesterday, what did you say?"
         │
         ▼
┌─────────────────────────────────────────┐
│           STAGE A: Tokenizer            │
│  - Scan for explicit prefixes           │
│  - Extract command-style scopes         │
│  - Extract known temporal tokens        │
│  - Compute coverage score               │
└─────────────────────────────────────────┘
         │
         ├── coverage >= 0.8 ──► Use Stage A result
         │
         ▼ coverage < 0.8
┌─────────────────────────────────────────┐
│         STAGE B: LLM Parser             │
│  - Temperature 0                        │
│  - Strict JSON schema                   │
│  - Validation against invariants        │
│  - Reject invalid → fall back to A      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Validated QueryIntent          │
└─────────────────────────────────────────┘
```

---

## 2. Output Schema

Both stages produce the same structure:

```python
@dataclass
class QueryIntent:
    mode: Literal["answer", "browse", "summarize"]
    target: str                                      # Search terms (may be empty for browse)
    temporal: Optional[TemporalSpec]                 # None = no time filter
    segment_scope_requested: bool                    # True only if explicit scope language
    segment_phrase: Optional[str]                    # Raw phrase for resolver (if requested)
    speaker: Optional[Literal["user", "assistant"]]  # None = both
    deictic_limit: Optional[int]                     # "last N messages" → N
    confidence: float                                # 0.0-1.0
    parse_stage: Literal["A", "B"]                   # Which stage produced this

@dataclass
class TemporalSpec:
    kind: Literal["relative_day", "relative_period", "named_day", "explicit_date", "date_range", "time_of_day"]
    raw: str                                         # Original phrase
    # Resolver converts to UTC half-open interval
```

---

## 3. Stage A: Deterministic Tokenizer

### 3.1 Token Types

```python
class TokenType(Enum):
    MODE_PREFIX = auto()      # "Browse:", "Answer:", "Summarize:"
    SCOPE_TOPIC = auto()      # "topic:X", "in topic X", "segment:X"
    SCOPE_SPEAKER = auto()    # "speaker:me", "speaker:you", "only what I said"
    TEMPORAL = auto()         # "yesterday", "last week", "between X and Y"
    DEICTIC = auto()          # "last 5 messages", "last exchange"
    TEXT = auto()             # Everything else (becomes target)
```

### 3.2 Tokenization Rules (Order Matters)

Process input left-to-right. First match wins. Matched text is consumed.

**Rule 1: Mode Prefix (must be at start)**
```
^Browse:\s*      → MODE_PREFIX("browse")
^Answer:\s*      → MODE_PREFIX("answer")  
^Summarize:\s*   → MODE_PREFIX("summarize")
^Show\s+         → MODE_PREFIX("browse")
^List\s+         → MODE_PREFIX("browse")
```

**Rule 2: Command-Style Scopes (anywhere)**
```
\btopic:(\S+)           → SCOPE_TOPIC($1)
\bsegment:(\S+)         → SCOPE_TOPIC($1)
\bspeaker:(me|you|user|assistant)\b → SCOPE_SPEAKER($1)
```

**Rule 3: Explicit Scope Phrases**
```
\bin\s+(?:the\s+)?(?:topic|segment)\s+"([^"]+)"  → SCOPE_TOPIC($1)
\bin\s+(?:the\s+)?(?:topic|segment)\s+(\S+)      → SCOPE_TOPIC($1)
\bwithin\s+(?:the\s+)?(?:topic|segment)\s+(\S+)  → SCOPE_TOPIC($1)
```

**Rule 4: Speaker Phrases (require speech verbs)**
```
\bonly\s+(?:show\s+)?what\s+I\s+(?:said|asked|wrote|mentioned)\b  → SCOPE_SPEAKER("user")
\bonly\s+(?:show\s+)?what\s+you\s+(?:said|recommended|wrote)\b   → SCOPE_SPEAKER("assistant")
\buser\s+messages?\s+only\b      → SCOPE_SPEAKER("user")
\bassistant\s+responses?\s+only\b → SCOPE_SPEAKER("assistant")
```

**Rule 5: Temporal Tokens**
```
# Relative days
\b(yesterday|today)\b                    → TEMPORAL(relative_day, $1)
\bthe\s+day\s+before\s+yesterday\b       → TEMPORAL(relative_day, "day_before_yesterday")

# Relative periods  
\b(last|this|past)\s+(week|month)\b      → TEMPORAL(relative_period, "$1 $2")
\b(last|past)\s+(\d+)\s+days?\b          → TEMPORAL(relative_period, "$1 $2 days")

# Named days (require prefix)
\b(last|this|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b
                                         → TEMPORAL(named_day, "$1 $2")

# Explicit dates (month names only - no "segment 1" false matches)
\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b
                                         → TEMPORAL(explicit_date, "$1 $2 $3")

# ISO dates
\b(\d{4}-\d{2}-\d{2})\b                  → TEMPORAL(explicit_date, $1)

# Date ranges
\bbetween\s+(jan\w*\s+\d+)\s+and\s+(jan\w*\s+\d+)\b  → TEMPORAL(date_range, "$1 to $2")
# (expand for all month names)

# Time of day
\b(earlier\s+today|this\s+morning|this\s+afternoon|this\s+evening)\b
                                         → TEMPORAL(time_of_day, $1)
```

**Rule 6: Deictic Message Counts**
```
\b(?:the\s+)?last\s+(\d+)\s+(?:messages?|exchanges?)\b → DEICTIC($1)
\b(?:the\s+)?last\s+exchange\b                         → DEICTIC(1)
\bjust\s+now\b                                         → DEICTIC(1)
\brecently\b                                           → DEICTIC(10)
```

**Rule 7: Everything Else**
```
\S+  → TEXT (accumulate into target)
```

### 3.3 Coverage Score

```python
def compute_coverage(tokens: List[Token], original: str) -> float:
    """
    Coverage = fraction of input consumed by non-TEXT tokens.
    High coverage = Stage A handled it.
    Low coverage = lots of unparsed natural language.
    """
    text_chars = sum(len(t.raw) for t in tokens if t.type == TokenType.TEXT)
    total_chars = len(original.replace(" ", ""))
    return 1.0 - (text_chars / total_chars) if total_chars > 0 else 1.0
```

### 3.4 Stage A Output

```python
def stage_a_parse(query: str) -> QueryIntent:
    tokens = tokenize(query)
    coverage = compute_coverage(tokens, query)
    
    # Extract fields
    mode = "answer"  # default
    for t in tokens:
        if t.type == TokenType.MODE_PREFIX:
            mode = t.value
            break
    
    # Mode inference from keywords if no prefix
    if not any(t.type == TokenType.MODE_PREFIX for t in tokens):
        text = query.lower()
        if any(w in text for w in ["show", "list", "browse", "display"]):
            mode = "browse"
        elif any(w in text for w in ["summarize", "summary"]):
            mode = "summarize"
        # "what were we" pattern → browse
        if re.match(r"^what\s+(?:were|was|did)\s+(?:we|I|you)\b", text):
            mode = "browse"
    
    temporal = None
    for t in tokens:
        if t.type == TokenType.TEMPORAL:
            temporal = TemporalSpec(kind=t.subtype, raw=t.value)
            break
    
    segment_requested = False
    segment_phrase = None
    for t in tokens:
        if t.type == TokenType.SCOPE_TOPIC:
            segment_requested = True
            segment_phrase = t.value
            break
    
    speaker = None
    for t in tokens:
        if t.type == TokenType.SCOPE_SPEAKER:
            speaker = "user" if t.value in ("me", "user") else "assistant"
            break
    
    deictic = None
    for t in tokens:
        if t.type == TokenType.DEICTIC:
            deictic = int(t.value)
            break
    
    target = " ".join(t.value for t in tokens if t.type == TokenType.TEXT)
    target = clean_target(target)  # Remove filler words
    
    return QueryIntent(
        mode=mode,
        target=target,
        temporal=temporal,
        segment_scope_requested=segment_requested,
        segment_phrase=segment_phrase,
        speaker=speaker,
        deictic_limit=deictic,
        confidence=coverage,
        parse_stage="A"
    )
```

### 3.5 Stage A Examples

| Input | Tokens | Coverage | Result |
|-------|--------|----------|--------|
| `Browse: last 10 messages` | MODE_PREFIX, DEICTIC | 1.0 | mode=browse, deictic=10 |
| `topic:coffee yesterday` | SCOPE_TOPIC, TEMPORAL | 1.0 | segment_requested=True, segment_phrase="coffee", temporal=yesterday |
| `what did you say about BM25 last week` | SCOPE_SPEAKER(?), TEMPORAL, TEXT, TEXT | 0.4 | Needs Stage B |
| `when we discussed coffee, what did we decide?` | TEXT, TEXT, TEXT... | 0.0 | Needs Stage B |

---

## 4. Stage B: LLM Parser

### 4.1 Invocation Criteria

```python
def needs_stage_b(stage_a_result: QueryIntent) -> bool:
    return stage_a_result.confidence < 0.8
```

### 4.2 Prompt

```python
STAGE_B_PROMPT = """
You are parsing a recall query for a conversation memory system.

Extract these fields from the user's query:
- mode: "answer" (find information), "browse" (show exchanges), or "summarize" (create summary)
- target: the search terms (what to look for)
- temporal_kind: one of [null, "yesterday", "today", "last_week", "this_week", "last_N_days", "explicit_date", "date_range", "time_of_day"] 
- temporal_raw: the exact temporal phrase from the query (or null)
- segment_scope_requested: true ONLY if user explicitly said "in topic/segment X" or "within topic X" or "topic:X"
- segment_phrase: the topic/segment name if explicitly requested (or null)
- speaker: "user" (I/me/my), "assistant" (you/your), or null (both)
- deictic_limit: integer if user said "last N messages" (or null)

CRITICAL RULES:
1. segment_scope_requested is TRUE only for explicit scope language:
   - "in topic coffee" → true
   - "within segment FTS" → true  
   - "topic:retrieval" → true
   - "when we discussed coffee" → FALSE (this is target, not scope)
   - "our discussion about X" → FALSE (this is target, not scope)

2. temporal_kind must be from the allowed list. If you can't map it, use null.

3. speaker requires speech verbs:
   - "you said", "you recommended" → "assistant"
   - "I said", "I asked" → "user"
   - "your approach" → null (no speech verb)

Respond with ONLY a JSON object, no other text.

Query: {query}
"""
```

### 4.3 Schema Validation

```python
ALLOWED_TEMPORAL_KINDS = {
    None, "yesterday", "today", "last_week", "this_week", 
    "last_N_days", "explicit_date", "date_range", "time_of_day"
}

def validate_llm_output(raw: dict) -> Optional[QueryIntent]:
    """
    Validate LLM JSON against invariants.
    Returns None if invalid (triggers fallback to Stage A).
    """
    try:
        # Required fields
        mode = raw.get("mode")
        if mode not in ("answer", "browse", "summarize"):
            return None
        
        target = raw.get("target", "")
        if not isinstance(target, str):
            return None
        
        # Temporal validation
        temporal_kind = raw.get("temporal_kind")
        if temporal_kind not in ALLOWED_TEMPORAL_KINDS:
            return None
        
        temporal = None
        if temporal_kind:
            temporal_raw = raw.get("temporal_raw", "")
            if not temporal_raw:
                return None  # Kind without raw phrase is invalid
            temporal = TemporalSpec(kind=temporal_kind, raw=temporal_raw)
        
        # Segment scope - must be explicit
        segment_requested = raw.get("segment_scope_requested", False)
        if not isinstance(segment_requested, bool):
            return None
        
        segment_phrase = raw.get("segment_phrase")
        if segment_requested and not segment_phrase:
            return None  # Requested but no phrase is invalid
        if not segment_requested and segment_phrase:
            # LLM tried to infer scope - reject this
            segment_phrase = None
        
        # Speaker validation
        speaker = raw.get("speaker")
        if speaker not in (None, "user", "assistant"):
            return None
        
        # Deictic
        deictic = raw.get("deictic_limit")
        if deictic is not None:
            if not isinstance(deictic, int) or deictic < 1:
                return None
        
        return QueryIntent(
            mode=mode,
            target=target,
            temporal=temporal,
            segment_scope_requested=segment_requested,
            segment_phrase=segment_phrase,
            speaker=speaker,
            deictic_limit=deictic,
            confidence=0.85,  # LLM parse, validated
            parse_stage="B"
        )
    
    except Exception:
        return None
```

### 4.4 LLM Call

```python
def stage_b_parse(query: str, stage_a_result: QueryIntent) -> QueryIntent:
    """
    Call LLM with temperature=0, validate output, fall back to Stage A if invalid.
    """
    prompt = STAGE_B_PROMPT.format(query=query)
    
    try:
        response = llm_call(
            model="gpt-4o-mini",  # or local model
            prompt=prompt,
            temperature=0,
            max_tokens=200
        )
        
        raw = json.loads(response)
        validated = validate_llm_output(raw)
        
        if validated:
            logger.debug("AUDIT: [stage_b] raw=%r validated=%r", raw, validated)
            return validated
        else:
            logger.debug("AUDIT: [stage_b] validation failed, raw=%r", raw)
            return stage_a_result
    
    except Exception as e:
        logger.debug("AUDIT: [stage_b] error=%s, falling back to stage_a", e)
        return stage_a_result
```

---

## 5. Combined Pipeline

```python
def understand_query(query: str, conn: sqlite3.Connection, config: dict) -> QueryIntent:
    """
    Two-stage parse with AUDIT logging.
    """
    logger.debug("AUDIT: [understand] input=%r", query)
    
    # Stage A: Deterministic
    stage_a = stage_a_parse(query)
    logger.debug("AUDIT: [stage_a] result=%r coverage=%.2f", stage_a, stage_a.confidence)
    
    # Check if Stage B needed
    if stage_a.confidence >= 0.8:
        return stage_a
    
    # Stage B: LLM with validation
    result = stage_b_parse(query, stage_a)
    logger.debug("AUDIT: [final] stage=%s result=%r", result.parse_stage, result)
    
    return result
```

---

## 6. Temporal Resolution

After parsing, temporal specs are resolved to UTC half-open intervals:

```python
def resolve_temporal(spec: TemporalSpec, user_tz: str, now: datetime) -> Tuple[datetime, datetime]:
    """
    Convert TemporalSpec to UTC [start, end) interval.
    """
    tz = pytz.timezone(user_tz)
    local_now = now.astimezone(tz)
    
    if spec.kind == "yesterday":
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        local_end = local_start + timedelta(days=1)
    
    elif spec.kind == "today":
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_end = local_start + timedelta(days=1)
    
    elif spec.kind == "last_week":
        # Monday of last week to Monday of this week
        days_since_monday = local_now.weekday()
        this_monday = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        local_start = this_monday - timedelta(days=7)
        local_end = this_monday
    
    elif spec.kind == "this_week":
        days_since_monday = local_now.weekday()
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        local_end = local_start + timedelta(days=7)
    
    elif spec.kind == "last_N_days":
        n = int(re.search(r'\d+', spec.raw).group())
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=n)
        local_end = local_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    elif spec.kind == "explicit_date":
        # Parse date from raw
        parsed = dateutil.parser.parse(spec.raw)
        local_start = tz.localize(parsed.replace(hour=0, minute=0, second=0, microsecond=0))
        local_end = local_start + timedelta(days=1)
    
    elif spec.kind == "date_range":
        # Parse "X to Y" from raw
        parts = spec.raw.split(" to ")
        start_parsed = dateutil.parser.parse(parts[0])
        end_parsed = dateutil.parser.parse(parts[1])
        local_start = tz.localize(start_parsed.replace(hour=0, minute=0, second=0, microsecond=0))
        local_end = tz.localize(end_parsed.replace(hour=0, minute=0, second=0, microsecond=0)) + timedelta(days=1)
    
    elif spec.kind == "time_of_day":
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        if "morning" in spec.raw:
            local_end = local_start.replace(hour=12)
        elif "afternoon" in spec.raw:
            local_start = local_start.replace(hour=12)
            local_end = local_start.replace(hour=18)
        elif "evening" in spec.raw:
            local_start = local_start.replace(hour=18)
            local_end = local_start + timedelta(days=1)
        else:  # "earlier today"
            local_end = local_now
    
    else:
        raise ValueError(f"Unknown temporal kind: {spec.kind}")
    
    # Convert to UTC
    return (local_start.astimezone(pytz.UTC), local_end.astimezone(pytz.UTC))
```

---

## 7. Segment Resolution

Only invoked when `segment_scope_requested=True`:

```python
def resolve_segment(
    conn: sqlite3.Connection,
    phrase: str,
    embedding_cache: Dict[int, List[float]]
) -> List[str]:
    """
    Resolve segment phrase to node IDs.
    
    Returns [] if no match (triggers empty result per tri-state semantics).
    """
    topics = get_all_topics(conn)
    
    # Score each topic
    scores = []
    phrase_lower = phrase.lower().strip()
    phrase_tokens = set(phrase_lower.split())
    
    for topic in topics:
        name_lower = topic['name'].lower()
        name_tokens = set(name_lower.split('-'))  # Handle slugs like "coffee-brewing"
        
        # Exact match
        if phrase_lower == name_lower or phrase_lower == name_lower.replace('-', ' '):
            scores.append((topic, 1.0))
            continue
        
        # Contains match
        if phrase_lower in name_lower or name_lower in phrase_lower:
            scores.append((topic, 0.8))
            continue
        
        # Jaccard similarity
        intersection = len(phrase_tokens & name_tokens)
        union = len(phrase_tokens | name_tokens)
        jaccard = intersection / union if union > 0 else 0
        
        if jaccard > 0.3:
            scores.append((topic, 0.4 + 0.4 * jaccard))
    
    if not scores:
        return []
    
    # Sort by score desc, then topic.id desc (prefer recent)
    scores.sort(key=lambda x: (-x[1], -x[0]['id']))
    
    best_topic, best_score = scores[0]
    if best_score < 0.3:
        return []
    
    # Get nodes for best topic
    nodes, _ = get_cached_segment_nodes(conn, best_topic['id'])
    return nodes
```

---

## 8. Full Integration

```python
def parse_and_resolve(
    query: str,
    conn: sqlite3.Connection,
    config: dict
) -> ResolvedQuery:
    """
    Parse query and resolve all scopes to concrete values.
    """
    intent = understand_query(query, conn, config)
    
    # Resolve temporal
    temporal_interval = None
    if intent.temporal:
        temporal_interval = resolve_temporal(
            intent.temporal,
            config["timezone"],
            datetime.now(pytz.UTC)
        )
    
    # Resolve segment (only if explicitly requested)
    segment_nodes = None  # None = no filter (tri-state)
    if intent.segment_scope_requested:
        segment_nodes = resolve_segment(conn, intent.segment_phrase, {})
        # Note: [] means requested but not found → empty results
    
    return ResolvedQuery(
        mode=intent.mode,
        target=intent.target,
        temporal=temporal_interval,
        segment_scope=segment_nodes,  # None, [], or [ids]
        speaker=intent.speaker,
        deictic_limit=intent.deictic_limit,
        confidence=intent.confidence,
        parse_stage=intent.parse_stage
    )
```

---

## 9. Examples

### Example 1: Fully Explicit (Stage A Only)

**Input:** `Browse: topic:coffee yesterday`

**Stage A Tokens:**
- MODE_PREFIX("browse")
- SCOPE_TOPIC("coffee")
- TEMPORAL(relative_day, "yesterday")

**Coverage:** 1.0 → Stage A sufficient

**Result:**
```python
QueryIntent(
    mode="browse",
    target="",
    temporal=TemporalSpec("relative_day", "yesterday"),
    segment_scope_requested=True,
    segment_phrase="coffee",
    speaker=None,
    deictic_limit=None,
    confidence=1.0,
    parse_stage="A"
)
```

### Example 2: Natural Language (Stage B)

**Input:** `when we discussed coffee yesterday, what did you say about grinders?`

**Stage A:**
- TEMPORAL(relative_day, "yesterday")
- TEXT: "when", "we", "discussed", "coffee", "what", "did", "you", "say", "about", "grinders"

**Coverage:** ~0.1 → Needs Stage B

**Stage B LLM Output:**
```json
{
    "mode": "answer",
    "target": "coffee grinders",
    "temporal_kind": "yesterday",
    "temporal_raw": "yesterday",
    "segment_scope_requested": false,
    "segment_phrase": null,
    "speaker": "assistant",
    "deictic_limit": null
}
```

**Validation:** ✓ Passes (segment_scope_requested correctly false)

**Result:**
```python
QueryIntent(
    mode="answer",
    target="coffee grinders",
    temporal=TemporalSpec("yesterday", "yesterday"),
    segment_scope_requested=False,  # NOT inferred from "discussed coffee"
    segment_phrase=None,
    speaker="assistant",
    deictic_limit=None,
    confidence=0.85,
    parse_stage="B"
)
```

### Example 3: Explicit Segment Scope

**Input:** `In topic coffee-brewing, what did we decide about water ratio?`

**Stage A:**
- SCOPE_TOPIC("coffee-brewing")
- TEXT: "what", "did", "we", "decide", "about", "water", "ratio"

**Coverage:** ~0.25 → Needs Stage B

**Stage B LLM Output:**
```json
{
    "mode": "answer",
    "target": "water ratio",
    "temporal_kind": null,
    "temporal_raw": null,
    "segment_scope_requested": true,
    "segment_phrase": "coffee-brewing",
    "speaker": null,
    "deictic_limit": null
}
```

**Result:**
```python
QueryIntent(
    mode="answer",
    target="water ratio",
    segment_scope_requested=True,  # Explicit "in topic"
    segment_phrase="coffee-brewing",
    ...
)
```

### Example 4: LLM Tries to Infer Scope (Rejected)

**Input:** `what did we talk about regarding coffee last week?`

**Stage B LLM Output (Invalid):**
```json
{
    "mode": "answer",
    "target": "coffee",
    "segment_scope_requested": false,
    "segment_phrase": "coffee"  // ← INVALID: phrase without requested=true
}
```

**Validation:** ✗ Fails (segment_phrase set but segment_scope_requested=false)

**Result:** Falls back to Stage A with target="coffee last week"

---

## 10. Configuration

```python
QUERY_UNDERSTANDING_CONFIG = {
    "stage_a_coverage_threshold": 0.8,
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0,
    "llm_max_tokens": 200,
    "segment_score_threshold": 0.3,
    "timezone": "America/Chicago",
    "default_deictic_limit": 10,
}
```

---

## 11. Test Cases

```python
# Stage A tests
def test_explicit_browse_with_deictic():
    result = stage_a_parse("Browse: last 10 messages")
    assert result.mode == "browse"
    assert result.deictic_limit == 10
    assert result.confidence >= 0.8

def test_command_style_scopes():
    result = stage_a_parse("topic:coffee yesterday")
    assert result.segment_scope_requested == True
    assert result.segment_phrase == "coffee"
    assert result.temporal.kind == "relative_day"

def test_explicit_speaker():
    result = stage_a_parse("only what you said about BM25")
    assert result.speaker == "assistant"

# Stage B validation tests
def test_validation_rejects_inferred_scope():
    raw = {"mode": "answer", "target": "coffee", 
           "segment_scope_requested": False, "segment_phrase": "coffee"}
    assert validate_llm_output(raw) is None

def test_validation_accepts_explicit_scope():
    raw = {"mode": "answer", "target": "grinders",
           "segment_scope_requested": True, "segment_phrase": "coffee"}
    result = validate_llm_output(raw)
    assert result is not None
    assert result.segment_scope_requested == True

def test_validation_rejects_bad_temporal_kind():
    raw = {"mode": "answer", "target": "test", "temporal_kind": "sometime_last_year"}
    assert validate_llm_output(raw) is None

# Integration tests
def test_natural_language_does_not_infer_segment():
    result = understand_query("when we discussed coffee yesterday", conn, config)
    assert result.segment_scope_requested == False
    assert "coffee" in result.target

def test_explicit_in_topic_triggers_segment():
    result = understand_query("in topic coffee, what did we decide?", conn, config)
    assert result.segment_scope_requested == True
    assert result.segment_phrase == "coffee"
```

---

## 12. Success Criteria

1. **Stage A handles explicit forms:** Coverage >= 0.8 for command-style queries
2. **Stage B validates strictly:** Invalid LLM output falls back to Stage A
3. **Segment scope only explicit:** "when we discussed X" → target, not scope
4. **Temporal whitelist enforced:** Only allowed kinds pass validation
5. **Speaker requires speech verbs:** "your approach" → no speaker scope
6. **AUDIT logging:** Every parse traced at DEBUG level
7. **Deterministic Stage A:** Same input → same output, always
8. **Graceful degradation:** LLM failure → Stage A result, never crash
