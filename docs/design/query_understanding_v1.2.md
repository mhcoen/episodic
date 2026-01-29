# Query Understanding Specification v1.2
## Hybrid Parser with Command Form Recognition

---

## 1. Design Principles

1. **Stage A is a command recognizer, not an English parser.** It handles ~12 explicit command forms. Everything else goes to Stage B.
2. **Stage B is the default.** If Stage A doesn't match a command form exactly, route to LLM immediately.
3. **Fail closed.** Invalid LLM output → empty result with explanation, not silent broadening.
4. **Audit everything.** Parse traces for every query.

---

## 2. Output Schema

```python
@dataclass
class QueryIntent:
    mode: Literal["answer", "browse", "summarize"]
    target: str                                      # Search terms (empty only allowed for browse)
    temporal: Optional[TemporalSpec]                 # None = no time filter
    segment_scope_requested: bool                    # True ONLY for explicit "in topic/segment X"
    segment_phrase: Optional[str]                    # Raw phrase for resolver (if requested)
    speaker: Optional[Literal["user", "assistant"]]  # None = both
    deictic_limit: Optional[int]                     # "last N messages" → N
    parse_stage: Literal["A", "B"]                   # Which stage produced this
    parse_rule: Optional[str]                        # Stage A: rule ID; Stage B: "llm"

@dataclass  
class TemporalSpec:
    kind: Literal[
        "yesterday", "today", "day_before_yesterday",
        "last_week", "this_week", "last_month", "this_month",
        "last_N_days", "named_day", "explicit_date", "date_range",
        "this_morning", "this_afternoon", "this_evening", "earlier_today"
    ]
    raw: str
    n: Optional[int] = None  # For "last_N_days"
```

---

## 3. Canonical Command Forms (Stage A)

Stage A recognizes **only** these command templates. Anything else → Stage B.

### 3.1 Command Catalog

| ID | Template | Mode | Speaker | Segment | Temporal | Target From |
|----|----------|------|---------|---------|----------|-------------|
| B1 | `Browse: <target>` | browse | - | - | - | after prefix |
| B2 | `Show me where we talked about <target>` | browse | - | - | - | after "about" |
| B3 | `last <N> messages` | browse | - | - | - | empty |
| B4 | `last <N> exchanges` | browse | - | - | - | empty |
| A1 | `Answer: <target>` | answer | - | - | - | after prefix |
| A2 | `What did you say about <target>` | answer | assistant | - | - | after "about" |
| A3 | `What did we conclude about <target>` | answer | - | - | - | after "about" |
| S1 | `Summarize: <target>` | summarize | - | - | - | after prefix |
| S2 | `Summarize our discussion about <target>` | summarize | - | - | - | after "about" |
| SP1 | `Did I ever mention <target>` | browse | user | - | - | after "mention" |
| SP2 | `Did you ever say <target>` | browse | assistant | - | - | after "say" |
| T1 | `<base> yesterday` | (from base) | (from base) | - | yesterday | (from base) |
| T2 | `<base> last week` | (from base) | (from base) | - | last_week | (from base) |
| T3 | `<base> on <month> <day>` | (from base) | (from base) | - | explicit_date | (from base) |
| SEG1 | `<base> in topic <phrase>` | (from base) | (from base) | yes | (from base) | (from base) |
| SEG2 | `<base> in segment <phrase>` | (from base) | (from base) | yes | (from base) | (from base) |
| SEG3 | `topic:<phrase> <rest>` | (infer) | - | yes | - | rest |
| SEG4 | `segment:<phrase> <rest>` | (infer) | - | yes | - | rest |

### 3.2 Stage A Matching Algorithm

```python
COMMAND_PATTERNS = [
    # Explicit prefixes (highest priority)
    CommandPattern(
        id="B1",
        regex=r"^Browse:\s*(.*)$",
        mode="browse",
        target_group=1
    ),
    CommandPattern(
        id="A1",
        regex=r"^Answer:\s*(.*)$",
        mode="answer",
        target_group=1
    ),
    CommandPattern(
        id="S1",
        regex=r"^Summarize:\s*(.*)$",
        mode="summarize",
        target_group=1
    ),
    
    # Deictic patterns
    CommandPattern(
        id="B3",
        regex=r"^(?:the\s+)?last\s+(\d+)\s+messages?$",
        mode="browse",
        target="",
        deictic_group=1
    ),
    CommandPattern(
        id="B4",
        regex=r"^(?:the\s+)?last\s+(\d+)\s+exchanges?$",
        mode="browse",
        target="",
        deictic_group=1
    ),
    
    # Browse patterns
    CommandPattern(
        id="B2",
        regex=r"^(?:show\s+me\s+)?where\s+we\s+talked\s+about\s+(.+)$",
        mode="browse",
        target_group=1
    ),
    
    # Answer patterns with speaker
    CommandPattern(
        id="A2",
        regex=r"^what\s+did\s+you\s+say\s+about\s+(.+)$",
        mode="answer",
        speaker="assistant",
        target_group=1
    ),
    CommandPattern(
        id="A3",
        regex=r"^what\s+did\s+we\s+(?:conclude|decide)\s+about\s+(.+)$",
        mode="answer",
        target_group=1
    ),
    
    # Summarize patterns
    CommandPattern(
        id="S2",
        regex=r"^summarize\s+(?:our\s+)?discussion\s+(?:about|of)\s+(.+)$",
        mode="summarize",
        target_group=1
    ),
    
    # Speaker patterns
    CommandPattern(
        id="SP1",
        regex=r"^did\s+I\s+ever\s+(?:mention|say|ask)\s+(.+)$",
        mode="browse",
        speaker="user",
        target_group=1
    ),
    CommandPattern(
        id="SP2",
        regex=r"^did\s+you\s+ever\s+(?:say|mention|recommend)\s+(.+)$",
        mode="browse",
        speaker="assistant",
        target_group=1
    ),
]

# Temporal suffixes (applied after base pattern match)
TEMPORAL_SUFFIXES = [
    TemporalSuffix(
        id="T1",
        regex=r"\s+yesterday$",
        kind="yesterday"
    ),
    TemporalSuffix(
        id="T2", 
        regex=r"\s+last\s+week$",
        kind="last_week"
    ),
    TemporalSuffix(
        id="T3",
        regex=r"\s+this\s+week$",
        kind="this_week"
    ),
    TemporalSuffix(
        id="T4",
        regex=r"\s+last\s+month$",
        kind="last_month"
    ),
    TemporalSuffix(
        id="T5",
        regex=r"\s+last\s+(\d+)\s+days?$",
        kind="last_N_days",
        n_group=1
    ),
    TemporalSuffix(
        id="T6",
        regex=r"\s+on\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:,?\s+\d{4})?)$",
        kind="explicit_date",
        raw_group=1
    ),
    TemporalSuffix(
        id="T7",
        regex=r"\s+between\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})\s+and\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})$",
        kind="date_range",
        raw_group=(1, 2)
    ),
]

# Segment scope prefixes/suffixes
SEGMENT_PATTERNS = [
    SegmentPattern(
        id="SEG1",
        regex=r"\s+in\s+topic\s+['\"]?([^'\"]+?)['\"]?$",
        phrase_group=1
    ),
    SegmentPattern(
        id="SEG2",
        regex=r"\s+in\s+segment\s+['\"]?([^'\"]+?)['\"]?$",
        phrase_group=1
    ),
    SegmentPattern(
        id="SEG3",
        regex=r"^topic:(\S+)\s+(.*)$",
        phrase_group=1,
        remainder_group=2
    ),
    SegmentPattern(
        id="SEG4",
        regex=r"^segment:(\S+)\s+(.*)$",
        phrase_group=1,
        remainder_group=2
    ),
]


def stage_a_parse(query: str) -> Optional[QueryIntent]:
    """
    Attempt to match query against command catalog.
    Returns None if no command form matches → route to Stage B.
    """
    q = query.strip()
    q_lower = q.lower()
    
    # 1. Check for segment prefix commands first (topic:X, segment:X)
    for seg in SEGMENT_PATTERNS:
        if seg.remainder_group:  # Prefix pattern
            m = re.match(seg.regex, q_lower)
            if m:
                phrase = m.group(seg.phrase_group)
                remainder = m.group(seg.remainder_group)
                # Recursively parse remainder
                inner = stage_a_parse(remainder)
                if inner:
                    inner.segment_scope_requested = True
                    inner.segment_phrase = phrase
                    inner.parse_rule = f"{seg.id}+{inner.parse_rule}"
                    return inner
                # No inner match → Stage B with segment hint
                return None
    
    # 2. Try main command patterns
    for cmd in COMMAND_PATTERNS:
        m = re.match(cmd.regex, q_lower, re.IGNORECASE)
        if m:
            target_raw = m.group(cmd.target_group) if cmd.target_group else cmd.target
            deictic = int(m.group(cmd.deictic_group)) if cmd.deictic_group else None
            
            # 3. Check for temporal suffix on target
            temporal = None
            for tsuf in TEMPORAL_SUFFIXES:
                tm = re.search(tsuf.regex, target_raw, re.IGNORECASE)
                if tm:
                    target_raw = target_raw[:tm.start()]
                    if tsuf.kind == "last_N_days":
                        n = int(tm.group(tsuf.n_group))
                        temporal = TemporalSpec(kind=tsuf.kind, raw=tm.group(0).strip(), n=n)
                    elif tsuf.kind == "date_range":
                        raw = f"{tm.group(tsuf.raw_group[0])} to {tm.group(tsuf.raw_group[1])}"
                        temporal = TemporalSpec(kind=tsuf.kind, raw=raw)
                    else:
                        temporal = TemporalSpec(kind=tsuf.kind, raw=tm.group(0).strip())
                    break
            
            # 4. Check for segment suffix on target
            segment_requested = False
            segment_phrase = None
            for seg in SEGMENT_PATTERNS:
                if not seg.remainder_group:  # Suffix pattern
                    sm = re.search(seg.regex, target_raw, re.IGNORECASE)
                    if sm:
                        target_raw = target_raw[:sm.start()]
                        segment_requested = True
                        segment_phrase = sm.group(seg.phrase_group)
                        break
            
            target = target_raw.strip()
            
            return QueryIntent(
                mode=cmd.mode,
                target=target,
                temporal=temporal,
                segment_scope_requested=segment_requested,
                segment_phrase=segment_phrase,
                speaker=cmd.speaker,
                deictic_limit=deictic,
                parse_stage="A",
                parse_rule=cmd.id
            )
    
    # No command form matched
    return None
```

### 3.3 Stage A Examples

| Input | Matched | Rule | Result |
|-------|---------|------|--------|
| `Browse: coffee` | ✓ | B1 | mode=browse, target="coffee" |
| `last 10 messages` | ✓ | B3 | mode=browse, deictic=10 |
| `What did you say about BM25 yesterday` | ✓ | A2+T1 | mode=answer, speaker=assistant, target="BM25", temporal=yesterday |
| `topic:coffee what did we decide` | ✓ | SEG3+A3 | mode=answer, segment_requested=True, segment_phrase="coffee", target="decide" |
| `when we discussed coffee yesterday` | ✗ | - | → Stage B |
| `pull up the coffee thing` | ✗ | - | → Stage B |

---

## 4. Stage B: LLM Parser

### 4.1 Invocation

```python
def understand_query(query: str, conn, config) -> QueryIntent:
    # Stage A: Try command recognition
    stage_a_result = stage_a_parse(query)
    
    if stage_a_result is not None:
        logger.debug("AUDIT: [stage_a] matched rule=%s result=%r", 
                     stage_a_result.parse_rule, stage_a_result)
        return stage_a_result
    
    # Stage B: LLM parse (default path)
    logger.debug("AUDIT: [stage_a] no match, routing to stage_b")
    return stage_b_parse(query, config)
```

### 4.2 LLM Prompt

```python
STAGE_B_PROMPT = '''You are parsing a recall query for a conversation memory system.

Given the user's query, extract a JSON object with exactly these fields:

{
  "mode": "answer" | "browse" | "summarize",
  "target": "<search terms>",
  "temporal_kind": null | "yesterday" | "today" | "last_week" | "this_week" | "last_month" | "last_N_days" | "explicit_date" | "date_range" | "this_morning" | "this_afternoon" | "this_evening",
  "temporal_raw": "<exact temporal phrase from query>" | null,
  "temporal_n": <integer for last_N_days> | null,
  "segment_scope_requested": true | false,
  "segment_phrase": "<topic/segment name>" | null,
  "speaker": "user" | "assistant" | null,
  "deictic_limit": <integer> | null
}

RULES:

1. mode:
   - "browse" = show exchanges (triggered by: "show me", "when did we", "where did we", "list", "display")
   - "answer" = find specific information (triggered by: "what did we decide", "what was the conclusion")
   - "summarize" = create summary (triggered by: "summarize", "summary of")
   - Default: "answer" for questions, "browse" for imperatives

2. target:
   - The thing being searched for
   - For browse mode, may be empty (means "show recent")
   - For answer/summarize, must NOT be empty

3. temporal:
   - Only set if user mentioned time
   - temporal_kind must be from the allowed list
   - temporal_raw is the exact phrase from the query
   - temporal_n is only for "last_N_days"

4. segment_scope_requested:
   - TRUE only if user explicitly said "in topic X", "in segment X", "within topic X", "topic:X"
   - FALSE for casual phrases like "when we discussed X", "our conversation about X"
   - This is critical: casual phrasing means search FOR X, not search IN topic named X

5. speaker:
   - "user" if query asks about what the human said ("I said", "did I mention", "my messages")
   - "assistant" if query asks about what AI said ("you said", "did you recommend", "your response")
   - null if query asks about both or doesn't specify

6. deictic_limit:
   - Set if user says "last N messages/exchanges"
   - null otherwise

Respond with ONLY the JSON object. No other text.

Query: {query}
'''
```

### 4.3 Validation

```python
ALLOWED_TEMPORAL_KINDS = frozenset([
    None, "yesterday", "today", "day_before_yesterday",
    "last_week", "this_week", "last_month", "this_month",
    "last_N_days", "named_day", "explicit_date", "date_range",
    "this_morning", "this_afternoon", "this_evening", "earlier_today"
])

ALLOWED_MODES = frozenset(["answer", "browse", "summarize"])
ALLOWED_SPEAKERS = frozenset([None, "user", "assistant"])


def validate_llm_output(raw: dict) -> Tuple[Optional[QueryIntent], Optional[str]]:
    """
    Validate LLM JSON against invariants.
    Returns (intent, None) on success or (None, error_message) on failure.
    """
    errors = []
    
    # Mode: required, must be in enum
    mode = raw.get("mode")
    if mode not in ALLOWED_MODES:
        errors.append(f"invalid mode: {mode}")
    
    # Target: required string
    target = raw.get("target")
    if not isinstance(target, str):
        errors.append(f"target must be string, got {type(target)}")
    
    # Empty target validation by mode
    if mode in ("answer", "summarize") and not target:
        errors.append(f"target cannot be empty for mode={mode}")
    
    # Temporal: kind must be in whitelist
    temporal_kind = raw.get("temporal_kind")
    if temporal_kind not in ALLOWED_TEMPORAL_KINDS:
        errors.append(f"invalid temporal_kind: {temporal_kind}")
    
    # Temporal: raw required if kind set
    temporal_raw = raw.get("temporal_raw")
    if temporal_kind and not temporal_raw:
        errors.append("temporal_raw required when temporal_kind is set")
    
    # Temporal: n required for last_N_days
    temporal_n = raw.get("temporal_n")
    if temporal_kind == "last_N_days" and not isinstance(temporal_n, int):
        errors.append("temporal_n required for last_N_days")
    
    # Segment: consistency check
    segment_requested = raw.get("segment_scope_requested", False)
    segment_phrase = raw.get("segment_phrase")
    
    if not isinstance(segment_requested, bool):
        errors.append("segment_scope_requested must be boolean")
    
    if segment_requested and not segment_phrase:
        errors.append("segment_phrase required when segment_scope_requested=true")
    
    if not segment_requested and segment_phrase:
        # LLM tried to infer scope - reject
        errors.append("segment_phrase set but segment_scope_requested=false (scope inference not allowed)")
    
    # Speaker: must be in enum
    speaker = raw.get("speaker")
    if speaker not in ALLOWED_SPEAKERS:
        errors.append(f"invalid speaker: {speaker}")
    
    # Deictic: must be positive int if set
    deictic = raw.get("deictic_limit")
    if deictic is not None and (not isinstance(deictic, int) or deictic < 1):
        errors.append(f"deictic_limit must be positive int, got {deictic}")
    
    # Reject unknown keys
    allowed_keys = {"mode", "target", "temporal_kind", "temporal_raw", "temporal_n",
                    "segment_scope_requested", "segment_phrase", "speaker", "deictic_limit"}
    unknown = set(raw.keys()) - allowed_keys
    if unknown:
        errors.append(f"unknown keys: {unknown}")
    
    if errors:
        return None, "; ".join(errors)
    
    # Build validated intent
    temporal = None
    if temporal_kind:
        temporal = TemporalSpec(kind=temporal_kind, raw=temporal_raw, n=temporal_n)
    
    return QueryIntent(
        mode=mode,
        target=target or "",
        temporal=temporal,
        segment_scope_requested=segment_requested,
        segment_phrase=segment_phrase,
        speaker=speaker,
        deictic_limit=deictic,
        parse_stage="B",
        parse_rule="llm"
    ), None


def stage_b_parse(query: str, config: dict) -> QueryIntent:
    """
    Parse via LLM with strict validation.
    On failure: return empty-result intent (fail closed).
    """
    prompt = STAGE_B_PROMPT.format(query=query)
    
    try:
        response = llm_call(
            model=config.get("llm_model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"}  # Force JSON
        )
        
        raw = json.loads(response)
        logger.debug("AUDIT: [stage_b] llm_raw=%r", raw)
        
        intent, error = validate_llm_output(raw)
        
        if intent:
            logger.debug("AUDIT: [stage_b] validated=%r", intent)
            return intent
        else:
            logger.warning("AUDIT: [stage_b] validation_failed=%s", error)
            return _fail_closed_intent(query, error)
    
    except json.JSONDecodeError as e:
        logger.warning("AUDIT: [stage_b] json_error=%s", e)
        return _fail_closed_intent(query, f"JSON parse error: {e}")
    
    except Exception as e:
        logger.warning("AUDIT: [stage_b] error=%s", e)
        return _fail_closed_intent(query, str(e))


def _fail_closed_intent(query: str, error: str) -> QueryIntent:
    """
    Return a safe intent that will produce empty results.
    """
    return QueryIntent(
        mode="answer",
        target=query,  # Use whole query as target (best effort)
        temporal=None,
        segment_scope_requested=False,
        segment_phrase=None,
        speaker=None,
        deictic_limit=None,
        parse_stage="B",
        parse_rule=f"fallback:{error[:50]}"
    )
```

---

## 5. Temporal Resolution (DST-Safe)

```python
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta


def resolve_temporal(
    spec: TemporalSpec,
    user_tz: str,
    reference: datetime  # UTC
) -> Tuple[datetime, datetime]:
    """
    Convert TemporalSpec to UTC half-open interval [start, end).
    
    Uses zoneinfo for DST-safe local midnight computation.
    """
    tz = ZoneInfo(user_tz)
    local_ref = reference.astimezone(tz)
    
    # Compute local midnight (start of day)
    def local_midnight(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    today_midnight = local_midnight(local_ref)
    
    if spec.kind == "yesterday":
        local_start = today_midnight - timedelta(days=1)
        local_end = today_midnight
    
    elif spec.kind == "today":
        local_start = today_midnight
        local_end = today_midnight + timedelta(days=1)
    
    elif spec.kind == "day_before_yesterday":
        local_start = today_midnight - timedelta(days=2)
        local_end = today_midnight - timedelta(days=1)
    
    elif spec.kind == "last_week":
        # Monday of last week to Monday of this week
        days_since_monday = local_ref.weekday()
        this_monday = today_midnight - timedelta(days=days_since_monday)
        local_start = this_monday - timedelta(days=7)
        local_end = this_monday
    
    elif spec.kind == "this_week":
        days_since_monday = local_ref.weekday()
        local_start = today_midnight - timedelta(days=days_since_monday)
        local_end = local_start + timedelta(days=7)
    
    elif spec.kind == "last_month":
        # First of last month to first of this month
        first_of_this_month = local_ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_ref.month == 1:
            first_of_last_month = first_of_this_month.replace(year=local_ref.year - 1, month=12)
        else:
            first_of_last_month = first_of_this_month.replace(month=local_ref.month - 1)
        local_start = first_of_last_month
        local_end = first_of_this_month
    
    elif spec.kind == "this_month":
        first_of_this_month = local_ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_ref.month == 12:
            first_of_next_month = first_of_this_month.replace(year=local_ref.year + 1, month=1)
        else:
            first_of_next_month = first_of_this_month.replace(month=local_ref.month + 1)
        local_start = first_of_this_month
        local_end = first_of_next_month
    
    elif spec.kind == "last_N_days":
        n = spec.n
        local_start = today_midnight - timedelta(days=n)
        local_end = today_midnight + timedelta(days=1)  # Include today
    
    elif spec.kind == "explicit_date":
        # Parse date from raw (e.g., "Jan 24" or "January 24, 2026")
        from dateutil import parser as dateparser
        parsed = dateparser.parse(spec.raw, default=local_ref)
        local_start = tz.localize(parsed.replace(hour=0, minute=0, second=0, microsecond=0))
        local_end = local_start + timedelta(days=1)
    
    elif spec.kind == "date_range":
        from dateutil import parser as dateparser
        parts = spec.raw.split(" to ")
        start_parsed = dateparser.parse(parts[0], default=local_ref)
        end_parsed = dateparser.parse(parts[1], default=local_ref)
        local_start = tz.localize(start_parsed.replace(hour=0, minute=0, second=0, microsecond=0))
        local_end = tz.localize(end_parsed.replace(hour=0, minute=0, second=0, microsecond=0)) + timedelta(days=1)
    
    elif spec.kind == "this_morning":
        local_start = today_midnight
        local_end = today_midnight.replace(hour=12)
    
    elif spec.kind == "this_afternoon":
        local_start = today_midnight.replace(hour=12)
        local_end = today_midnight.replace(hour=18)
    
    elif spec.kind == "this_evening":
        local_start = today_midnight.replace(hour=18)
        local_end = today_midnight + timedelta(days=1)
    
    elif spec.kind == "earlier_today":
        local_start = today_midnight
        local_end = local_ref
    
    else:
        raise ValueError(f"Unknown temporal kind: {spec.kind}")
    
    # Convert to UTC
    utc = ZoneInfo("UTC")
    return (local_start.astimezone(utc), local_end.astimezone(utc))
```

### 5.1 DST Test Cases

```python
def test_yesterday_across_spring_forward():
    """March 10, 2024 at 3am EDT (spring forward happened at 2am)"""
    tz = "America/New_York"
    ref = datetime(2024, 3, 10, 15, 0, tzinfo=ZoneInfo("UTC"))  # 11am EDT
    spec = TemporalSpec(kind="yesterday", raw="yesterday")
    
    start, end = resolve_temporal(spec, tz, ref)
    
    # March 9 was still EST (UTC-5)
    # Local midnight March 9 = 05:00 UTC
    # Local midnight March 10 = 04:00 UTC (EDT, UTC-4)
    assert start.hour == 5  # March 9 00:00 EST
    assert end.hour == 4    # March 10 00:00 EDT

def test_yesterday_across_fall_back():
    """November 3, 2024 at 3am EST (fall back happened at 2am)"""
    tz = "America/New_York"
    ref = datetime(2024, 11, 3, 15, 0, tzinfo=ZoneInfo("UTC"))  # 10am EST
    spec = TemporalSpec(kind="yesterday", raw="yesterday")
    
    start, end = resolve_temporal(spec, tz, ref)
    
    # Nov 2 was EDT (UTC-4), Nov 3 is EST (UTC-5)
    assert start.hour == 4  # Nov 2 00:00 EDT
    assert end.hour == 5    # Nov 3 00:00 EST
```

---

## 6. Segment Resolution

```python
def resolve_segment(
    conn: sqlite3.Connection,
    phrase: str
) -> List[str]:
    """
    Resolve segment phrase to node IDs.
    Returns [] if no match (→ empty results per tri-state).
    """
    topics = get_all_topics(conn)
    phrase_lower = phrase.lower().strip()
    phrase_tokens = set(phrase_lower.replace('-', ' ').split())
    
    candidates = []
    
    for topic in topics:
        name = topic['name'].lower()
        name_normalized = name.replace('-', ' ')
        name_tokens = set(name_normalized.split())
        
        # Exact match (highest score)
        if phrase_lower == name or phrase_lower == name_normalized:
            candidates.append((topic, 1.0, "exact"))
            continue
        
        # Contains match
        if phrase_lower in name_normalized or name_normalized in phrase_lower:
            candidates.append((topic, 0.8, "contains"))
            continue
        
        # Token overlap (Jaccard)
        intersection = phrase_tokens & name_tokens
        union = phrase_tokens | name_tokens
        jaccard = len(intersection) / len(union) if union else 0
        
        if jaccard >= 0.3:
            candidates.append((topic, 0.3 + 0.4 * jaccard, "jaccard"))
    
    if not candidates:
        logger.debug("AUDIT: [segment] phrase=%r no_matches", phrase)
        return []
    
    # Sort: score desc, then topic.id desc (prefer recent)
    candidates.sort(key=lambda x: (-x[1], -x[0]['id']))
    
    best, score, match_type = candidates[0]
    logger.debug("AUDIT: [segment] phrase=%r best=%s score=%.2f type=%s",
                 phrase, best['name'], score, match_type)
    
    if score < 0.3:
        return []
    
    nodes, _ = get_cached_segment_nodes(conn, best['id'])
    return nodes
```

---

## 7. Full Pipeline

```python
@dataclass
class ResolvedQuery:
    mode: str
    target: str
    temporal: Optional[Tuple[datetime, datetime]]  # UTC half-open
    segment_scope: Optional[List[str]]              # None, [], or [node_ids]
    speaker: Optional[str]
    deictic_limit: Optional[int]
    parse_stage: str
    parse_rule: str


def parse_and_resolve(
    query: str,
    conn: sqlite3.Connection,
    config: dict
) -> ResolvedQuery:
    """
    Parse query and resolve all scopes.
    """
    intent = understand_query(query, conn, config)
    
    # Resolve temporal
    temporal = None
    if intent.temporal:
        temporal = resolve_temporal(
            intent.temporal,
            config["timezone"],
            datetime.now(ZoneInfo("UTC"))
        )
    
    # Resolve segment (only if explicitly requested)
    segment_scope = None
    if intent.segment_scope_requested:
        segment_scope = resolve_segment(conn, intent.segment_phrase)
        # [] means requested but not found → empty results
    
    return ResolvedQuery(
        mode=intent.mode,
        target=intent.target,
        temporal=temporal,
        segment_scope=segment_scope,
        speaker=intent.speaker,
        deictic_limit=intent.deictic_limit,
        parse_stage=intent.parse_stage,
        parse_rule=intent.parse_rule
    )
```

---

## 8. Configuration

```python
QUERY_UNDERSTANDING_CONFIG = {
    # LLM settings
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0,
    "llm_max_tokens": 300,
    
    # Resolution
    "timezone": "America/Chicago",
    "segment_score_threshold": 0.3,
    
    # Defaults
    "default_deictic_limit": 10,
}
```

---

## 9. AUDIT Logging

Every parse produces a trace:

```
AUDIT: [understand] input="when we discussed coffee yesterday, what did you say?"
AUDIT: [stage_a] no match, routing to stage_b
AUDIT: [stage_b] llm_raw={"mode": "answer", "target": "coffee", ...}
AUDIT: [stage_b] validated=QueryIntent(mode='answer', target='coffee', ...)
AUDIT: [segment] phrase=None (not requested)
AUDIT: [temporal] spec=yesterday -> [2026-01-24T06:00Z, 2026-01-25T06:00Z)
AUDIT: [resolved] mode=answer target="coffee" temporal=[...] segment=None speaker=assistant
```

---

## 10. Test Suite

```python
# Stage A: Command form recognition
class TestStageA:
    def test_explicit_browse_prefix(self):
        result = stage_a_parse("Browse: coffee grinders")
        assert result.mode == "browse"
        assert result.target == "coffee grinders"
        assert result.parse_rule == "B1"
    
    def test_deictic_messages(self):
        result = stage_a_parse("last 10 messages")
        assert result.mode == "browse"
        assert result.deictic_limit == 10
        assert result.target == ""
    
    def test_speaker_with_temporal(self):
        result = stage_a_parse("What did you say about BM25 yesterday")
        assert result.mode == "answer"
        assert result.speaker == "assistant"
        assert result.target == "BM25"
        assert result.temporal.kind == "yesterday"
    
    def test_topic_prefix_command(self):
        result = stage_a_parse("topic:coffee what did we decide")
        assert result.segment_scope_requested == True
        assert result.segment_phrase == "coffee"
    
    def test_no_match_routes_to_stage_b(self):
        result = stage_a_parse("when we discussed coffee yesterday")
        assert result is None  # → Stage B


# Stage B: LLM validation
class TestStageB:
    def test_rejects_inferred_segment_scope(self):
        raw = {
            "mode": "answer",
            "target": "coffee",
            "segment_scope_requested": False,
            "segment_phrase": "coffee-brewing"  # Invalid: phrase without request
        }
        intent, error = validate_llm_output(raw)
        assert intent is None
        assert "scope inference not allowed" in error
    
    def test_rejects_empty_target_for_answer(self):
        raw = {
            "mode": "answer",
            "target": "",
            "segment_scope_requested": False
        }
        intent, error = validate_llm_output(raw)
        assert intent is None
        assert "cannot be empty" in error
    
    def test_accepts_empty_target_for_browse(self):
        raw = {
            "mode": "browse",
            "target": "",
            "segment_scope_requested": False
        }
        intent, error = validate_llm_output(raw)
        assert intent is not None
    
    def test_rejects_unknown_temporal_kind(self):
        raw = {
            "mode": "answer",
            "target": "coffee",
            "temporal_kind": "sometime_last_year"
        }
        intent, error = validate_llm_output(raw)
        assert intent is None
    
    def test_rejects_unknown_keys(self):
        raw = {
            "mode": "answer",
            "target": "coffee",
            "extra_field": "bad"
        }
        intent, error = validate_llm_output(raw)
        assert intent is None
        assert "unknown keys" in error


# Temporal resolution
class TestTemporal:
    def test_yesterday_cst(self):
        tz = "America/Chicago"
        ref = datetime(2026, 1, 25, 18, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="yesterday", raw="yesterday")
        start, end = resolve_temporal(spec, tz, ref)
        # Jan 24 00:00 CST = Jan 24 06:00 UTC
        assert start == datetime(2026, 1, 24, 6, 0, tzinfo=ZoneInfo("UTC"))
        assert end == datetime(2026, 1, 25, 6, 0, tzinfo=ZoneInfo("UTC"))
    
    def test_last_5_days(self):
        tz = "America/Chicago"
        ref = datetime(2026, 1, 25, 18, 0, tzinfo=ZoneInfo("UTC"))
        spec = TemporalSpec(kind="last_N_days", raw="last 5 days", n=5)
        start, end = resolve_temporal(spec, tz, ref)
        # Jan 20 00:00 CST to Jan 26 00:00 CST
        assert start == datetime(2026, 1, 20, 6, 0, tzinfo=ZoneInfo("UTC"))
        assert end == datetime(2026, 1, 26, 6, 0, tzinfo=ZoneInfo("UTC"))


# Integration
class TestIntegration:
    def test_natural_language_no_segment_inference(self):
        result = understand_query("when we discussed coffee yesterday", conn, config)
        assert result.segment_scope_requested == False
        assert "coffee" in result.target
    
    def test_explicit_in_topic_triggers_segment(self):
        result = understand_query("in topic coffee what did we decide", conn, config)
        assert result.segment_scope_requested == True
```

---

## 11. Success Criteria

1. **Stage A recognizes only command forms.** ~12 patterns, not general English.
2. **Stage B is the default.** Any non-command input routes to LLM.
3. **Validation rejects invalid LLM output.** Including segment inference, empty answer targets, unknown keys.
4. **Fail closed.** Invalid parse → fallback intent, not broadened search.
5. **DST-safe temporal.** Uses zoneinfo, tested across transitions.
6. **AUDIT traces every parse.** Stage, rule/error, resolved values.
7. **Empty target allowed only for browse.** Enforced in validation.
8. **Segment scope only explicit.** "when we discussed X" → target, not scope.
