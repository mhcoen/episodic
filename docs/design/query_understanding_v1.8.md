# Episodic Query Understanding (MQL) — v1.7

**Status:** Implementation spec for the query-understanding front-end (lexer → parser → resolver) that drives the Episodic retrieval pipeline.

This version is a corrective rewrite of v1.6. It addresses specification gaps and inconsistencies identified in external review, particularly around token definitions, punctuation handling, multiword phrase parsing, speaker mapping, temporal/deictic disambiguation, span conventions, FreeText fallback behavior, discussion-query coverage, and topic resolution ambiguity.

---

## 0. Changelog (v1.6 → v1.7)

1. **WORD token definition fixed** (Review Issue #1)
   - Explicit character class: `[A-Za-z0-9]([A-Za-z0-9_'-]*[A-Za-z0-9])?|[A-Za-z0-9]`
   - Allows internal hyphens, underscores, and apostrophes (but not at start/end of multi-char words)
   - Single alphanumeric characters are valid WORDs
   - Added fixtures for `don't`, `Ralph-Wiggum`, `Fenway's`

2. **Punctuation tokenization explicit** (Review Issue #2)
   - Lexer emits distinct token kinds: `COLON`, `COMMA`, `QUESTION`, `LPAREN`, `RPAREN`, `DASH`
   - No generic `PUNCT` token exists; removed ambiguity
   - Grammar explicitly uses `COLON` in segment productions (e.g., `topic: foo`)

3. **Multiword phrase parsing clarified** (Review Issue #3)
   - Lexer emits single-token keywords (`KW_WHEN`, `KW_DISCUSS`, `KW_SPEAKER`, etc.)
   - Multiword phrases like "when we discussed X" are matched by **parser productions with explicit lookahead**, NOT by lexer phrase recognition
   - No trie-based or phrase-level lexing; the lexer is strictly word-at-a-time

4. **SpeakerSpec.BOTH → speaker=None mapping** (Review Issue #4)
   - `SpeakerSpec(role="both")` (from "we/us/our") resolves to `speaker=None` in ResolvedQuery
   - This means "no speaker restriction" → both retrieval channels enabled
   - Explicit mapping table added in Section 6.7

5. **Temporal vs deictic "last time" disambiguation** (Review Issue #5)
   - Explicit disambiguation rule: parser checks for `last time` sequence BEFORE attempting temporal parse
   - `last` + `time` (or `last time we...`) → DEICTIC parse path
   - Temporal `last` only continues if followed by `week`/`month`/`year`/NUMBER/`days`

6. **Span convention pinned** (Review Issue #6)
   - All spans are codepoint offsets into `s_norm` (the normalized string)
   - `s_norm` is the canonical audited artifact and MUST be logged
   - `s_raw` is also logged for reference but spans do NOT index into it
   - No raw→normalized offset mapping is required; the normalization audit records what transformations occurred

7. **Temporal boundary type pinned** (Review Issue #7)
   - `temporal` field in ResolvedQuery is `Optional[Tuple[datetime, datetime]]`
   - Datetimes are timezone-aware UTC (`datetime` objects with `tzinfo=ZoneInfo("UTC")`)
   - Contract: the retrieval pipeline is responsible for formatting to ISO8601 strings for SQLite comparison

8. **FreeText resolver behavior explicit** (Review Issue #8)
   - FreeText input produces a fully-specified ResolvedQuery:
     - `mode="answer"`
     - `target=s_norm` (the full normalized input)
     - `segment_explicit=False`
     - `segment_query=None`
     - `segment_resolved_ids=None`
     - `temporal=None`
     - `speaker=None`
     - `deictic=None`

9. **"Have we" discussion-query forms added** (Review Issue #9)
   - Added `KW_HAVE` token for "have/has"
   - Added grammar production for "have we (ever)? discussed/talked/mentioned (about)? X"
   - Added trailing discourse markers: "before", "previously", "already" treated as broadness cues

10. **Topic resolver disambiguation explicit** (Review Issue #10)
    - Exact match → single result
    - Contains match with multiple candidates → return `AmbiguousSegment` with all candidates
    - Deterministic tie-breaking: `topic.id ASC` when multiple exact matches (shouldn't happen) or as display order for ambiguous
    - Audit logging required when ambiguity detected

11. **"before/previously/already/ever" are broadness cues, NOT temporal** (Review Issue #11)
    - These words in discussion-query context are discourse markers indicating "at any prior point"
    - They do NOT produce temporal filters
    - They MAY influence mode (→ browse) but not temporal range
    - Only anchored forms ("before yesterday", "before January") would be temporal (NOT supported in v1.7)

12. **DiscussionQuery as distinct AST node** (Review Issue #12)
    - Discussion-query forms produce `DiscussionQuery` AST node, not `MQLCommand` with side-effects
    - Makes the "discussion-query → browse, no segment" rule explicit and testable
    - Golden fixtures can assert `ast_kind: DiscussionQuery`

13. **Enhanced span provenance in AST** (Review Issue #13)
    - Every AST node derived from user text includes:
      - `span: Tuple[int, int]` — raw span [start, end) into s_norm
      - `source_tokens: List[int]` — indices into token stream
    - Enables full audit trail from resolved query back to input

---

## 1. User-Facing Mental Model

The user can type either:

**A) A structured "memory query command" (MQL)**, such as:
- `browse when we discussed coffee yesterday`
- `summarize in topic: weapons balance last week`
- `did you say 'Fenway'?`
- `have we talked about research before?`

**B) Free text** that is not reliably parseable as MQL.
- The system produces `FreeText(input)` and the resolver converts it to a ResolvedQuery with `mode="answer"` and `target=s_norm`, subject to the contract in Section 8.

**Key safety rule:** A discussion-query ("when we discussed X", "have we talked about X", "mentioned X") is NOT segment scope. Segment scope only occurs with explicit segment syntax.

---

## 2. Pipeline Overview

```
Input (raw user text)
  → Normalizer (Unicode + whitespace normalization)
  → Lexer (token stream with spans into s_norm)
  → Parser (AST: MQLCommand | DiscussionQuery | FreeText)
  → Resolver (ResolvedQuery: concrete mode/scope/target, plus audit)
  → Retrieval pipeline (out of scope here)
```

---

## 3. Normalization (Pre-Lex)

### 3.1 Purpose

Normalization reduces fragile parsing outcomes caused by Unicode punctuation variants and spacing differences, while preserving auditability.

Normalization is **deterministic** and MUST be applied before tokenization.

### 3.2 Transformations

Given raw input string `s_raw`, produce `s_norm` by applying:

```python
def normalize(s_raw: str) -> Tuple[str, NormalizationAudit]:
    """
    Apply Unicode and whitespace normalization.
    Returns (s_norm, audit_record).
    """
    s = s_raw
    changes = []
    
    # 1. Unicode punctuation normalization
    PUNCT_MAP = {
        '"': '"',  # Left double quote
        '"': '"',  # Right double quote
        ''': "'",  # Left single quote
        ''': "'",  # Right single quote
        '—': '-',  # Em dash
        '–': '-',  # En dash
        '\u00A0': ' ',  # Non-breaking space
        '\u2002': ' ',  # En space
        '\u2003': ' ',  # Em space
        '\u2009': ' ',  # Thin space
    }
    
    for old, new in PUNCT_MAP.items():
        if old in s:
            changes.append(f"replaced {repr(old)} with {repr(new)}")
            s = s.replace(old, new)
    
    # 2. Whitespace normalization
    import re
    s_collapsed = re.sub(r'\s+', ' ', s)
    if s_collapsed != s:
        changes.append("collapsed whitespace")
    s = s_collapsed
    
    # 3. Trim leading/trailing whitespace
    s_trimmed = s.strip()
    if s_trimmed != s:
        changes.append("trimmed whitespace")
    s = s_trimmed
    
    # NOTE: Do NOT lowercase (would break quoted target fidelity)
    # Case-insensitive matching is handled at lexer via keyword map
    
    return s, NormalizationAudit(
        raw=s_raw,
        normalized=s,
        changes=changes
    )


@dataclass(frozen=True)
class NormalizationAudit:
    raw: str
    normalized: str
    changes: List[str]
```

### 3.3 Span Convention (CRITICAL)

**All token spans are `(start, end)` codepoint offsets into `s_norm` (post-normalization).**

- `s_norm` is the **canonical audited artifact** and MUST be included in all audit output.
- `s_raw` is logged for reference but spans do NOT index into it.
- No raw→normalized offset mapping is maintained; the `NormalizationAudit.changes` list records what transformations occurred for forensic purposes.

This design choice prioritizes simplicity: spans are always valid against the string the lexer/parser actually operated on.

---

## 4. Lexical Specification

### 4.1 Token Types

```python
class TokenKind(Enum):
    # Literals (never reinterpreted)
    QUOTED = auto()       # "..." or '...' (quotes stripped in value)
    ISO_DATE = auto()     # YYYY-MM-DD (strict)
    NUMBER = auto()       # Sequence of digits
    
    # Punctuation (distinct tokens, NOT a generic PUNCT)
    COLON = auto()        # :
    COMMA = auto()        # ,
    QUESTION = auto()     # ?
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    DASH = auto()         # - (standalone, not internal to word)
    
    # Keywords (soft - may be reinterpreted as WORD by parser)
    KW_MODE = auto()      # browse, summarize, answer
    KW_SEGMENT = auto()   # topic, segment, topics, segments
    KW_SPEAKER = auto()   # i, me, my, you, your, we, us, our, user, assistant
    KW_TIME_REL = auto()  # yesterday, today, last, this, week, month, year, ago, days
    KW_DISCUSS = auto()   # discussed, talked, mention, mentioned, brought, bring, said, say, asked, ask
    KW_WHEN = auto()      # when, where
    KW_DID = auto()       # did
    KW_HAVE = auto()      # have, has (NEW)
    KW_EVER = auto()      # ever
    KW_IN = auto()        # in, within
    KW_ON = auto()        # on
    KW_ABOUT = auto()     # about
    KW_DEICTIC = auto()   # earlier, previous
    KW_TIME = auto()      # time (for "last time" disambiguation)
    KW_DISCOURSE = auto() # before, previously, already (broadness cues, NOT temporal)
    
    # Default
    WORD = auto()         # Any other word (see 4.5 for character class)
    
    # Special
    EOF = auto()
    LEX_ERROR = auto()    # Unclosed quote, invalid character


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str                 # Original substring from s_norm
    span: Tuple[int, int]       # (start, end) codepoint offsets into s_norm
    normalized: Optional[str]   # Canonical form (for keywords)
    index: int                  # Position in token stream (for provenance)
    
    def as_word(self) -> 'Token':
        """Reinterpret as WORD (soft keyword handling)."""
        return Token(TokenKind.WORD, self.lexeme, self.span, self.lexeme.lower(), self.index)
    
    def to_dict(self) -> dict:
        """Stable serialization."""
        return {
            "kind": self.kind.name,
            "lexeme": self.lexeme,
            "span": list(self.span),
            "normalized": self.normalized,
            "index": self.index
        }
```

### 4.2 Keyword Normalization Map

```python
KEYWORD_MAP: Dict[str, Tuple[TokenKind, str]] = {
    # Mode
    "browse": (TokenKind.KW_MODE, "browse"),
    "show": (TokenKind.KW_MODE, "browse"),
    "list": (TokenKind.KW_MODE, "browse"),
    "display": (TokenKind.KW_MODE, "browse"),
    "summarize": (TokenKind.KW_MODE, "summarize"),
    "summary": (TokenKind.KW_MODE, "summarize"),
    "answer": (TokenKind.KW_MODE, "answer"),
    
    # Segment
    "topic": (TokenKind.KW_SEGMENT, "topic"),
    "topics": (TokenKind.KW_SEGMENT, "topic"),
    "segment": (TokenKind.KW_SEGMENT, "segment"),
    "segments": (TokenKind.KW_SEGMENT, "segment"),
    
    # Speaker
    "i": (TokenKind.KW_SPEAKER, "user"),
    "me": (TokenKind.KW_SPEAKER, "user"),
    "my": (TokenKind.KW_SPEAKER, "user"),
    "you": (TokenKind.KW_SPEAKER, "assistant"),
    "your": (TokenKind.KW_SPEAKER, "assistant"),
    "we": (TokenKind.KW_SPEAKER, "both"),
    "us": (TokenKind.KW_SPEAKER, "both"),
    "our": (TokenKind.KW_SPEAKER, "both"),
    "user": (TokenKind.KW_SPEAKER, "user"),
    "assistant": (TokenKind.KW_SPEAKER, "assistant"),
    
    # Time relative
    "yesterday": (TokenKind.KW_TIME_REL, "yesterday"),
    "today": (TokenKind.KW_TIME_REL, "today"),
    "last": (TokenKind.KW_TIME_REL, "last"),
    "this": (TokenKind.KW_TIME_REL, "this"),
    "week": (TokenKind.KW_TIME_REL, "week"),
    "month": (TokenKind.KW_TIME_REL, "month"),
    "year": (TokenKind.KW_TIME_REL, "year"),
    "ago": (TokenKind.KW_TIME_REL, "ago"),
    "days": (TokenKind.KW_TIME_REL, "days"),
    "day": (TokenKind.KW_TIME_REL, "day"),
    
    # Time noun (for deictic disambiguation)
    "time": (TokenKind.KW_TIME, "time"),
    
    # Discussion verbs
    "discussed": (TokenKind.KW_DISCUSS, "discussed"),
    "discuss": (TokenKind.KW_DISCUSS, "discussed"),
    "talked": (TokenKind.KW_DISCUSS, "talked"),
    "talk": (TokenKind.KW_DISCUSS, "talked"),
    "mentioned": (TokenKind.KW_DISCUSS, "mentioned"),
    "mention": (TokenKind.KW_DISCUSS, "mentioned"),
    "brought": (TokenKind.KW_DISCUSS, "brought"),
    "bring": (TokenKind.KW_DISCUSS, "brought"),
    "said": (TokenKind.KW_DISCUSS, "said"),
    "say": (TokenKind.KW_DISCUSS, "said"),
    "asked": (TokenKind.KW_DISCUSS, "asked"),
    "ask": (TokenKind.KW_DISCUSS, "asked"),
    
    # Query openers
    "when": (TokenKind.KW_WHEN, "when"),
    "where": (TokenKind.KW_WHEN, "where"),
    "did": (TokenKind.KW_DID, "did"),
    "have": (TokenKind.KW_HAVE, "have"),
    "has": (TokenKind.KW_HAVE, "have"),
    "ever": (TokenKind.KW_EVER, "ever"),
    
    # Scope
    "in": (TokenKind.KW_IN, "in"),
    "within": (TokenKind.KW_IN, "in"),
    "on": (TokenKind.KW_ON, "on"),
    "about": (TokenKind.KW_ABOUT, "about"),
    
    # Deictic (anaphoric pointers)
    "earlier": (TokenKind.KW_DEICTIC, "earlier"),
    "previous": (TokenKind.KW_DEICTIC, "previous"),
    
    # Discourse markers (broadness cues, NOT temporal filters)
    "before": (TokenKind.KW_DISCOURSE, "before"),
    "previously": (TokenKind.KW_DISCOURSE, "previously"),
    "already": (TokenKind.KW_DISCOURSE, "already"),
}
```

### 4.3 Quoted Phrases

- Quoted strings are atomic `QUOTED` tokens.
- Quoted content is used verbatim for target extraction.
- **Unclosed quote:** Lexer MUST emit `LEX_ERROR` token with `lex_error_unclosed_quote`, which forces the parser to return `FreeText`.

```python
def _scan_quoted(self, quote_char: str) -> Token:
    start = self.pos
    self.pos += 1  # Skip opening quote
    
    while self.pos < len(self.s_norm):
        if self.s_norm[self.pos] == quote_char:
            self.pos += 1  # Skip closing quote
            lexeme = self.s_norm[start:self.pos]
            value = lexeme[1:-1]  # Strip quotes
            return Token(TokenKind.QUOTED, lexeme, (start, self.pos), value, self._next_index())
        self.pos += 1
    
    # Unclosed quote
    lexeme = self.s_norm[start:self.pos]
    return Token(TokenKind.LEX_ERROR, lexeme, (start, self.pos), "lex_error_unclosed_quote", self._next_index())
```

### 4.4 ISO Date Recognition

Strict `YYYY-MM-DD` format only. Month names are NOT supported in v1.7.

```python
ISO_DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')

def _try_scan_iso_date(self) -> Optional[Token]:
    match = ISO_DATE_PATTERN.match(self.s_norm, self.pos)
    if match:
        lexeme = match.group(0)
        self.pos = match.end()
        return Token(TokenKind.ISO_DATE, lexeme, (match.start(), match.end()), lexeme, self._next_index())
    return None
```

### 4.5 WORD Token Character Class (CORRECTED)

The WORD token allows internal hyphens, underscores, and apostrophes, but NOT at word boundaries.

**Regex:** `[A-Za-z0-9]([A-Za-z0-9_'-]*[A-Za-z0-9])?|[A-Za-z0-9]`

This matches:
- Single alphanumeric: `a`, `5`
- Multi-char with internal punctuation: `don't`, `Ralph-Wiggum`, `Fenway's`, `foo_bar`
- Does NOT match: `-foo`, `foo-`, `'bar`, `bar'`

```python
WORD_PATTERN = re.compile(r"[A-Za-z0-9]([A-Za-z0-9_'\-]*[A-Za-z0-9])?|[A-Za-z0-9]")

def _is_word_start(self, c: str) -> bool:
    """Word must start with alphanumeric."""
    return c.isalnum()

def _scan_word(self) -> Token:
    start = self.pos
    match = WORD_PATTERN.match(self.s_norm, self.pos)
    if match:
        lexeme = match.group(0)
        self.pos = match.end()
        lower = lexeme.lower()
        
        if lower in KEYWORD_MAP:
            kind, normalized = KEYWORD_MAP[lower]
            return Token(kind, lexeme, (start, self.pos), normalized, self._next_index())
        else:
            return Token(TokenKind.WORD, lexeme, (start, self.pos), lower, self._next_index())
    
    # Fallback: single character (shouldn't happen if _is_word_start is correct)
    self.pos += 1
    lexeme = self.s_norm[start:self.pos]
    return Token(TokenKind.WORD, lexeme, (start, self.pos), lexeme.lower(), self._next_index())
```

### 4.6 Punctuation Tokenization (EXPLICIT)

The lexer emits **distinct token kinds** for each punctuation character. There is no generic `PUNCT` token.

```python
def _scan_punct(self) -> Optional[Token]:
    """Scan single-character punctuation. Returns None if not punctuation."""
    start = self.pos
    char = self.s_norm[self.pos]
    
    PUNCT_TOKENS = {
        ':': TokenKind.COLON,
        ',': TokenKind.COMMA,
        '?': TokenKind.QUESTION,
        '(': TokenKind.LPAREN,
        ')': TokenKind.RPAREN,
    }
    
    if char in PUNCT_TOKENS:
        self.pos += 1
        return Token(PUNCT_TOKENS[char], char, (start, self.pos), char, self._next_index())
    
    # Standalone dash (not internal to word)
    if char == '-':
        # Check if this is standalone (not followed by alphanumeric that would make it word-internal)
        if self.pos + 1 >= len(self.s_norm) or not self.s_norm[self.pos + 1].isalnum():
            self.pos += 1
            return Token(TokenKind.DASH, char, (start, self.pos), char, self._next_index())
    
    return None
```

### 4.7 Lexer Implementation

```python
@dataclass
class LexResult:
    tokens: List[Token]
    s_norm: str
    has_error: bool
    error_code: Optional[str]


class Lexer:
    def __init__(self, s_norm: str):
        self.s_norm = s_norm
        self.pos = 0
        self.tokens: List[Token] = []
        self.has_error = False
        self.error_code = None
        self._token_index = 0
    
    def _next_index(self) -> int:
        idx = self._token_index
        self._token_index += 1
        return idx
    
    def tokenize(self) -> LexResult:
        while self.pos < len(self.s_norm):
            self._skip_whitespace()
            if self.pos >= len(self.s_norm):
                break
            self._scan_token()
        
        self.tokens.append(Token(TokenKind.EOF, "", (self.pos, self.pos), None, self._next_index()))
        return LexResult(
            tokens=self.tokens,
            s_norm=self.s_norm,
            has_error=self.has_error,
            error_code=self.error_code
        )
    
    def _skip_whitespace(self):
        while self.pos < len(self.s_norm) and self.s_norm[self.pos] == ' ':
            self.pos += 1
    
    def _scan_token(self):
        char = self.s_norm[self.pos]
        
        # Quoted string
        if char in ('"', "'"):
            tok = self._scan_quoted(char)
            self.tokens.append(tok)
            if tok.kind == TokenKind.LEX_ERROR:
                self.has_error = True
                self.error_code = tok.normalized
            return
        
        # Punctuation (distinct tokens)
        punct_tok = self._scan_punct()
        if punct_tok:
            self.tokens.append(punct_tok)
            return
        
        # ISO date (before number, since dates start with digits)
        date_tok = self._try_scan_iso_date()
        if date_tok:
            self.tokens.append(date_tok)
            return
        
        # Number
        if char.isdigit():
            self._scan_number()
            return
        
        # Word or keyword
        if self._is_word_start(char):
            tok = self._scan_word()
            self.tokens.append(tok)
            return
        
        # Unknown character - MUST emit token for auditability (never silently skip)
        lexeme = self.s_norm[self.pos]
        self.tokens.append(Token(
            TokenKind.LEX_ERROR, lexeme, (self.pos, self.pos + 1),
            "unknown_char", self._next_index()
        ))
        self.has_error = True
        if not self.error_code:
            self.error_code = "unknown_char"
        self.pos += 1
    
    def _scan_number(self):
        start = self.pos
        while self.pos < len(self.s_norm) and self.s_norm[self.pos].isdigit():
            self.pos += 1
        lexeme = self.s_norm[start:self.pos]
        self.tokens.append(Token(TokenKind.NUMBER, lexeme, (start, self.pos), lexeme, self._next_index()))
```

---

## 5. Grammar (Recursive Descent)

The parser is **deterministic** and must not backtrack exponentially.
If parsing fails at any required point, return `FreeText(s_norm)`, along with the token stream and an error code.

**INVARIANT: Any LEX_ERROR forces FreeText.**

If the lexer emits ANY `LEX_ERROR` token (unclosed quote, unknown character, etc.), the parser MUST return `FreeText`. It is NOT acceptable to parse a structured command containing lex errors—this would produce silent mis-parses on malformed input, which is worse than conservative fallback to FreeText.

### 5.1 Multiword Phrase Handling (CLARIFIED)

**The lexer emits single-token keywords.** Multiword phrases like "when we discussed X" or "have we talked about X" are matched by **parser productions with explicit lookahead**, NOT by lexer phrase recognition.

Example: "have we talked about research before"
- Lexer emits: `[KW_HAVE, KW_SPEAKER("both"), KW_DISCUSS, KW_ABOUT, WORD("research"), KW_DISCOURSE, EOF]`
- Parser matches the sequence via `_try_have_we_query()` production

This design:
- Keeps the lexer simple (word-at-a-time, no trie)
- Allows soft keywords to be reinterpreted as WORD when not in expected positions
- Prevents false positives from keywords appearing inside targets

### 5.2 AST Types

```python
class Mode(Enum):
    BROWSE = "browse"
    SUMMARIZE = "summarize"
    ANSWER = "answer"


@dataclass(frozen=True)
class SpanInfo:
    """Provenance information for any AST node derived from user text."""
    span: Tuple[int, int]        # [start, end) into s_norm
    source_tokens: Tuple[int, ...]  # Indices into token stream


@dataclass(frozen=True)
class SegmentSpec:
    explicit: bool
    query: Optional[str]  # Raw segment query text
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class SpeakerSpec:
    role: str  # "user", "assistant", or "both"
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class TemporalSpec:
    kind: str  # "yesterday", "last_week", "iso_date", etc.
    raw: str
    iso_date: Optional[str] = None  # For ISO dates
    n: Optional[int] = None         # For "N days ago"
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class DeicticSpec:
    kind: str  # "earlier", "previous", "last_time"
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class TargetSpec:
    text: str
    spans: Tuple[Tuple[int, int], ...]
    source_tokens: Tuple[int, ...]  # Token indices for full provenance


@dataclass(frozen=True)
class AuditInfo:
    s_raw: str                   # Original input (for reference)
    s_norm: str                  # Normalized input (canonical for spans)
    tokens: List[dict]           # Serialized token stream
    rule_path: List[str]         # Parser rules taken
    decisions: List[str]         # Soft-keyword reinterpretations


@dataclass(frozen=True)
class MQLCommand:
    """Standard MQL command (explicit mode, segment, etc.)."""
    mode: Mode
    segment: SegmentSpec
    speaker: Optional[SpeakerSpec]
    temporal: Optional[TemporalSpec]
    deictic: Optional[DeicticSpec]
    target: Optional[TargetSpec]
    audit: AuditInfo
    
    def to_dict(self) -> dict:
        """Canonical serialization (sorted keys)."""
        return {
            "ast_kind": "MQLCommand",
            "deictic": {"kind": self.deictic.kind} if self.deictic else None,
            "mode": self.mode.value,
            "segment": {"explicit": self.segment.explicit, "query": self.segment.query},
            "speaker": {"role": self.speaker.role} if self.speaker else None,
            "target": self.target.text if self.target else None,
            "temporal": {"kind": self.temporal.kind, "raw": self.temporal.raw} if self.temporal else None,
        }


@dataclass(frozen=True)
class DiscussionQuery:
    """
    Distinct AST node for discussion-query forms.
    
    This makes the "discussion-query → browse mode, no segment scope" rule
    explicit and testable. Golden fixtures can assert ast_kind: DiscussionQuery.
    
    Forms: "when we discussed X", "have we talked about X", "did I mention X", etc.
    """
    target: Optional[TargetSpec]
    speaker: Optional[SpeakerSpec]      # From "did I/you/we say"
    temporal: Optional[TemporalSpec]    # From trailing "yesterday", "last week"
    has_broadness_cue: bool             # True if "before/previously/already/ever" present
    query_form: str                     # "when_we", "have_we", "did_speaker", etc.
    audit: AuditInfo
    
    # These are ALWAYS fixed for DiscussionQuery:
    # - mode = BROWSE (implicit)
    # - segment.explicit = False (guaranteed)
    
    def to_dict(self) -> dict:
        return {
            "ast_kind": "DiscussionQuery",
            "has_broadness_cue": self.has_broadness_cue,
            "query_form": self.query_form,
            "speaker": {"role": self.speaker.role} if self.speaker else None,
            "target": self.target.text if self.target else None,
            "temporal": {"kind": self.temporal.kind, "raw": self.temporal.raw} if self.temporal else None,
        }


@dataclass(frozen=True)
class FreeText:
    text: str
    parse_error: str
    audit: AuditInfo
    
    def to_dict(self) -> dict:
        return {
            "ast_kind": "FreeText",
            "parse_error": self.parse_error,
            "text": self.text,
        }
```

### 5.3 Precedence and Ordering (Critical)

Parsing priority is structural, not "pattern order":

1. **Mode phrase** (if present; else default depends on subsequent forms)
2. **Discussion-query check** (when/have/did forms → DiscussionQuery AST)
3. **Explicit segment modifier** (topic/segment syntax only)
4. **Explicit speaker modifier** (did-I-say / did-you-say / did-we-say)
5. **Temporal modifier** (but see 5.10 for deictic disambiguation)
6. **Deictic modifier**
7. **Discourse markers** (before/previously/already → broadness cue, NOT temporal)
8. **Target extraction**

**Discussion-query forms produce `DiscussionQuery` AST nodes, which ALWAYS resolve to:**
- `mode = BROWSE`
- `segment.explicit = False`

### 5.4 Mode Phrases

Accepted explicit mode prefixes:
- `browse ...`
- `summarize ...`
- `answer ...`

If no explicit mode phrase:
- Discussion-query forces `BROWSE` (via DiscussionQuery node)
- Otherwise default is `ANSWER`

### 5.5 Segment Modifiers (Explicit Only)

**Accepted explicit segment forms** (`segment.explicit = True`):

1. `in topic: <segment_query>`
2. `in topic <segment_query>` — requires `in` prefix
3. `topic: <segment_query>` — colon required without `in`
4. `topic <segment_query>` — **NOT accepted** (requires colon or `in`)
5. `segment: <segment_query>`
6. `in segment <segment_query>`

`segment_query` is:
- A `QUOTED` literal, OR
- A sequence of WORD tokens up to the next recognized modifier boundary

**NOT accepted as segment modifiers** (these produce DiscussionQuery):
- `when we discussed X`
- `have we talked about X`
- `we mentioned X`

### 5.6 Discussion-Query Forms (DiscussionQuery AST)

These forms produce `DiscussionQuery` AST nodes with:
- Implicit `mode = BROWSE`
- `segment.explicit = False` (guaranteed by AST type)
- `target` extracted as X

**"When/Where" forms:**
- `when we discussed <X>`
- `when we talked about <X>`
- `when we mentioned <X>`
- `when did we discuss <X>`
- `where did we discuss <X>`

**"Have we" forms (NEW):**
- `have we discussed <X>`
- `have we (ever) discussed <X>`
- `have we talked about <X>`
- `have we (ever) talked about <X>`
- `have we mentioned <X>`
- `has anyone mentioned <X>` (→ speaker=None)

**"Did speaker" forms:**
- `did I say <X>`
- `did you mention <X>`
- `did we ever discuss <X>`

**Trailing discourse markers:**
- `have we talked about X before` → `has_broadness_cue = True`
- `did we discuss X previously` → `has_broadness_cue = True`
- `have we ever mentioned X` → `has_broadness_cue = True` (ever counts)

**This rule is NON-NEGOTIABLE: DiscussionQuery semantics NEVER imply segment scope.**

### 5.7 Discourse Markers: "before/previously/already/ever" (CRITICAL)

**These are broadness cues, NOT temporal filters.**

In the context of discussion-queries:
- "Have we talked about research **before**?" — "before" means "at any prior point"
- "Did we **ever** discuss this?" — "ever" means "at any point in history"
- "Have we **already** covered this?" — "already" means "prior to now"
- "We **previously** mentioned this" — "previously" means "in an earlier conversation"

**Behavior:**
- Set `DiscussionQuery.has_broadness_cue = True`
- Do NOT produce a temporal filter
- The word is consumed and not included in target

**Anchored temporal forms (NOT supported in v1.7):**
- "before yesterday" — would be temporal if supported
- "before January" — would be temporal if supported
- These fall to FreeText in v1.7

### 5.8 Speaker Forms

**Symmetric speaker constructions:**

| Form | SpeakerSpec.role | ResolvedQuery.speaker |
|------|------------------|----------------------|
| `did I say <X>` | "user" | "user" |
| `did I ask <X>` | "user" | "user" |
| `have I mentioned <X>` | "user" | "user" |
| `did you say <X>` | "assistant" | "assistant" |
| `did you mention <X>` | "assistant" | "assistant" |
| `have you said <X>` | "assistant" | "assistant" |
| `did we say <X>` | "both" | None |
| `have we discussed <X>` | "both" | None |
| `my messages` / `my responses` | "user" | "user" |
| `your messages` / `your responses` | "assistant" | "assistant" |

**Note:** "we/us/our" produces `SpeakerSpec(role="both")` which the resolver maps to `speaker=None` (no restriction).

### 5.9 Temporal Forms

**Supported in v1.7:**

| Form | Kind |
|------|------|
| `yesterday` | yesterday |
| `today` | today |
| `last week` | last_week |
| `this week` | this_week |
| `last month` | last_month |
| `this month` | this_month |
| `last year` | last_year |
| `last N days` | last_n_days |
| `N days ago` | n_days_ago |
| `on YYYY-MM-DD` | iso_date |
| `YYYY-MM-DD` | iso_date |

**NOT supported in v1.7 (falls to FreeText or becomes target):**
- `on Jan 5` — Month names not parsed as temporal
- `January 5` — Month names not parsed as temporal
- `before yesterday` — Anchored temporal not supported

This is explicitly fixture-covered to lock behavior.

### 5.10 Temporal vs Deictic Disambiguation (CRITICAL)

**Problem:** `last` appears in both temporal (`last week`) and deictic (`last time`) contexts.

**Resolution rule:** Parser checks for `last time` sequence BEFORE attempting temporal parse.

```python
def _parse_temporal_or_deictic(self):
    """
    Disambiguation: check for 'last time' (deictic) before 'last week' (temporal).
    """
    if self._at_normalized("last"):
        # Lookahead: is next token 'time'?
        if self._peek(1).kind == TokenKind.KW_TIME:
            # This is deictic "last time", NOT temporal
            self._match_normalized("last")
            self._match(TokenKind.KW_TIME)
            self.deictic = DeicticSpec(kind="last_time")
            self.rule_path.append("deictic:last_time_disambiguated")
            return
        
        # Otherwise, continue to temporal parse (last week/month/etc.)
    
    self._parse_temporal_modifier()
    self._parse_deictic_modifier()
```

### 5.11 Deictic Forms

**Supported in v1.7:**
- `earlier` → `DeicticSpec(kind="earlier")`
- `previous` → `DeicticSpec(kind="previous")`
- `last time` → `DeicticSpec(kind="last_time")` (see 5.10 for disambiguation)
- `the last time` → `DeicticSpec(kind="last_time")`

**NOT supported (falls to FreeText):**
- `that one`
- `the earlier answer`

**Note:** `before/previously/already` are discourse markers (broadness cues), NOT deictic. See 5.7.

### 5.12 Parser Implementation

```python
class Parser:
    def __init__(self, lex_result: LexResult, s_raw: str):
        self.tokens = lex_result.tokens
        self.s_norm = lex_result.s_norm
        self.s_raw = s_raw
        self.pos = 0
        self.rule_path: List[str] = []
        self.decisions: List[str] = []
        
        # Parse state
        self.mode: Optional[Mode] = None
        self.segment = SegmentSpec(explicit=False, query=None)
        self.speaker: Optional[SpeakerSpec] = None
        self.temporal: Optional[TemporalSpec] = None
        self.deictic: Optional[DeicticSpec] = None
        self.target_tokens: List[Token] = []
        self.has_broadness_cue: bool = False
        self.discussion_query_form: Optional[str] = None
        
        # Check for lex error
        self.lex_error = lex_result.has_error
        self.lex_error_code = lex_result.error_code
    
    def parse(self) -> Union[MQLCommand, DiscussionQuery, FreeText]:
        # INVARIANT: Any lex error forces FreeText - no partial parsing allowed.
        # This prevents silent mis-parses on malformed input.
        if self.lex_error:
            return self._make_freetext(f"lex_error:{self.lex_error_code}")
        
        try:
            # Check for discussion-query forms FIRST
            discussion = self._try_discussion_query()
            if discussion:
                return discussion
            
            # Standard MQL parsing
            self._parse_mode_phrase()
            self._parse_segment_modifier()
            self._parse_speaker_modifier()
            self._parse_temporal_or_deictic()
            self._parse_target()
            self._skip_trailing_punct()
            return self._build_command()
        except ParseError as e:
            return self._make_freetext(str(e))
    
    def _try_discussion_query(self) -> Optional[DiscussionQuery]:
        """
        Try to match discussion-query forms. Returns DiscussionQuery or None.
        
        Forms:
        - when/where (did)? (we|I|you)? discuss/talked/mentioned (about)? <X> (before|previously)?
        - have/has (we|I|you) (ever)? discuss/talked/mentioned (about)? <X> (before|previously)?
        - did (I|you|we) (ever)? discuss/talked/mentioned (about)? <X> (before|previously)?
        """
        saved = self._save()
        
        # Try "when/where" forms
        if self._at(TokenKind.KW_WHEN):
            result = self._parse_when_discussion_query()
            if result:
                return result
            self._restore(saved)
        
        # Try "have/has" forms
        if self._at(TokenKind.KW_HAVE):
            result = self._parse_have_discussion_query()
            if result:
                return result
            self._restore(saved)
        
        # Try "did" forms (speaker-specific)
        if self._at(TokenKind.KW_DID):
            result = self._parse_did_discussion_query()
            if result:
                return result
            self._restore(saved)
        
        return None
    
    def _parse_when_discussion_query(self) -> Optional[DiscussionQuery]:
        """Parse: when/where (did)? (we|I|you)? discuss/talked/mentioned (about)? <X>"""
        self._match(TokenKind.KW_WHEN)  # consume when/where
        self._match(TokenKind.KW_DID)   # optional did
        
        speaker_tok = self._match(TokenKind.KW_SPEAKER)  # optional speaker
        if speaker_tok:
            self.speaker = SpeakerSpec(
                role=speaker_tok.normalized,
                provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
            )
        
        self._match(TokenKind.KW_DISCUSS)  # optional verb
        self._match(TokenKind.KW_ABOUT)    # optional about
        
        # Parse target and trailing modifiers
        self._parse_discussion_target_and_modifiers()
        
        return self._build_discussion_query("when_we")
    
    def _parse_have_discussion_query(self) -> Optional[DiscussionQuery]:
        """Parse: have/has (we|I|you) (ever)? discuss/talked/mentioned (about)? <X> (before)?"""
        self._match(TokenKind.KW_HAVE)  # consume have/has
        
        speaker_tok = self._match(TokenKind.KW_SPEAKER)
        if not speaker_tok:
            return None  # "have" without speaker doesn't match
        
        self.speaker = SpeakerSpec(
            role=speaker_tok.normalized,
            provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
        )
        
        # "ever" is a broadness cue
        if self._match(TokenKind.KW_EVER):
            self.has_broadness_cue = True
        
        # Must have a discussion verb
        if not self._match(TokenKind.KW_DISCUSS):
            return None
        
        self._match(TokenKind.KW_ABOUT)  # optional about
        
        # Parse target and trailing modifiers
        self._parse_discussion_target_and_modifiers()
        
        return self._build_discussion_query("have_we")
    
    def _parse_did_discussion_query(self) -> Optional[DiscussionQuery]:
        """Parse: did (I|you|we) (ever)? discuss/talked/mentioned (about)? <X>"""
        self._match(TokenKind.KW_DID)  # consume did
        
        speaker_tok = self._match(TokenKind.KW_SPEAKER)
        if not speaker_tok:
            return None
        
        speaker = SpeakerSpec(
            role=speaker_tok.normalized,
            provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
        )
        
        if self._match(TokenKind.KW_EVER):
            self.has_broadness_cue = True
        
        if not self._match(TokenKind.KW_DISCUSS):
            return None
        
        self._match(TokenKind.KW_ABOUT)
        
        self._parse_discussion_target_and_modifiers()
        self.speaker = speaker
        
        return self._build_discussion_query("did_speaker")
    
    def _parse_discussion_target_and_modifiers(self):
        """Parse target, temporal, and trailing discourse markers for discussion queries."""
        # Collect target tokens until we hit temporal, discourse marker, or end
        while not self._at_end():
            tok = self._peek()
            
            # Stop at temporal keywords (but not "last" which might be in target)
            if tok.kind == TokenKind.KW_TIME_REL and tok.normalized in ("yesterday", "today"):
                break
            if tok.kind == TokenKind.KW_TIME_REL and tok.normalized == "last":
                # Check if "last week/month/year" follows
                if self._peek(1).normalized in ("week", "month", "year", "days", "day"):
                    break
                if self._peek(1).kind == TokenKind.NUMBER:
                    break
            
            # Stop at discourse markers
            if tok.kind == TokenKind.KW_DISCOURSE:
                break
            
            # Stop at punctuation
            if tok.kind in (TokenKind.QUESTION, TokenKind.COMMA, TokenKind.EOF):
                break
            
            # Collect token
            word_tok = self._accept_wordish()
            if word_tok:
                self.target_tokens.append(word_tok)
            elif tok.kind == TokenKind.QUOTED:
                self.target_tokens.append(self._match(TokenKind.QUOTED))
            else:
                break
        
        # Parse trailing temporal
        self._parse_temporal_modifier()
        
        # Parse trailing discourse markers (broadness cues)
        while self._at(TokenKind.KW_DISCOURSE):
            self._match(TokenKind.KW_DISCOURSE)
            self.has_broadness_cue = True
        
        self._skip_trailing_punct()
    
    def _build_discussion_query(self, form: str) -> DiscussionQuery:
        """Build DiscussionQuery AST node."""
        target = None
        if self.target_tokens:
            text = " ".join(t.normalized or t.lexeme for t in self.target_tokens)
            spans = tuple(t.span for t in self.target_tokens)
            indices = tuple(t.index for t in self.target_tokens)
            target = TargetSpec(text=text, spans=spans, source_tokens=indices)
        
        return DiscussionQuery(
            target=target,
            speaker=self.speaker,
            temporal=self.temporal,
            has_broadness_cue=self.has_broadness_cue,
            query_form=form,
            audit=AuditInfo(
                s_raw=self.s_raw,
                s_norm=self.s_norm,
                tokens=[t.to_dict() for t in self.tokens],
                rule_path=self.rule_path + [f"discussion_query:{form}"],
                decisions=self.decisions
            )
        )
    
    def _make_freetext(self, error: str) -> FreeText:
        return FreeText(
            text=self.s_norm,
            parse_error=error,
            audit=AuditInfo(
                s_raw=self.s_raw,
                s_norm=self.s_norm,
                tokens=[t.to_dict() for t in self.tokens],
                rule_path=self.rule_path,
                decisions=self.decisions
            )
        )
    
    def _build_command(self) -> MQLCommand:
        target = None
        if self.target_tokens:
            text = " ".join(t.normalized or t.lexeme for t in self.target_tokens)
            spans = tuple(t.span for t in self.target_tokens)
            indices = tuple(t.index for t in self.target_tokens)
            target = TargetSpec(text=text, spans=spans, source_tokens=indices)
        
        return MQLCommand(
            mode=self.mode or Mode.ANSWER,
            segment=self.segment,
            speaker=self.speaker,
            temporal=self.temporal,
            deictic=self.deictic,
            target=target,
            audit=AuditInfo(
                s_raw=self.s_raw,
                s_norm=self.s_norm,
                tokens=[t.to_dict() for t in self.tokens],
                rule_path=self.rule_path,
                decisions=self.decisions
            )
        )
    
    # --- Token helpers ---
    
    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]
    
    def _at(self, *kinds: TokenKind) -> bool:
        return self._peek().kind in kinds
    
    def _at_normalized(self, *values: str) -> bool:
        tok = self._peek()
        return tok.normalized in values
    
    def _match(self, *kinds: TokenKind) -> Optional[Token]:
        if self._at(*kinds):
            tok = self._peek()
            self.pos += 1
            return tok
        return None
    
    def _match_normalized(self, *values: str) -> Optional[Token]:
        if self._at_normalized(*values):
            tok = self._peek()
            self.pos += 1
            return tok
        return None
    
    def _at_end(self) -> bool:
        return self._at(TokenKind.EOF)
    
    def _save(self) -> int:
        return self.pos
    
    def _restore(self, pos: int):
        self.pos = pos
    
    def _accept_wordish(self) -> Optional[Token]:
        """Accept WORD or soft keyword as word."""
        tok = self._peek()
        if tok.kind == TokenKind.WORD:
            self.pos += 1
            return tok
        if tok.kind.name.startswith("KW_"):
            self.decisions.append(f"reinterpreted {tok.kind.name} as WORD at span {tok.span}")
            self.pos += 1
            return tok.as_word()
        return None
    
    # --- Mode parsing ---
    
    def _parse_mode_phrase(self):
        self.rule_path.append("mode_phrase")
        
        if self._at(TokenKind.KW_MODE):
            tok = self._match(TokenKind.KW_MODE)
            self.mode = Mode(tok.normalized)
            self.rule_path.append(f"explicit_mode:{self.mode.value}")
    
    # --- Segment parsing ---
    
    def _parse_segment_modifier(self):
        self.rule_path.append("segment_modifier")
        saved = self._save()
        
        # "topic:" or "segment:" (colon required without "in")
        if self._at(TokenKind.KW_SEGMENT):
            seg_tok = self._match(TokenKind.KW_SEGMENT)
            if self._match(TokenKind.COLON):
                query, provenance = self._parse_segment_query()
                if query:
                    self.segment = SegmentSpec(explicit=True, query=query, provenance=provenance)
                    self.rule_path.append(f"explicit_segment:{query}")
                    return
            self._restore(saved)
        
        # "in topic X" or "in segment X"
        if self._match_normalized("in"):
            if self._at(TokenKind.KW_SEGMENT):
                self._match(TokenKind.KW_SEGMENT)
                self._match(TokenKind.COLON)  # optional colon after "in topic"
                query, provenance = self._parse_segment_query()
                if query:
                    self.segment = SegmentSpec(explicit=True, query=query, provenance=provenance)
                    self.rule_path.append(f"explicit_segment_in:{query}")
                    return
            self._restore(saved)
    
    def _parse_segment_query(self) -> Tuple[Optional[str], Optional[SpanInfo]]:
        """Parse segment query: QUOTED or words until modifier boundary."""
        if self._at(TokenKind.QUOTED):
            tok = self._match(TokenKind.QUOTED)
            return tok.normalized, SpanInfo(tok.span, (tok.index,))
        
        words = []
        tokens_used = []
        start_span = None
        end_span = None
        
        for _ in range(5):  # Max 5 words
            tok = self._peek()
            # Stop at modifier boundaries - but NOT KW_SPEAKER, which should be
            # reinterpreted as WORD in segment names (e.g., "my-project", "our-research")
            if tok.kind in (TokenKind.KW_TIME_REL, TokenKind.KW_DEICTIC,
                           TokenKind.KW_TIME, TokenKind.KW_DISCOURSE, TokenKind.ISO_DATE, 
                           TokenKind.COLON, TokenKind.EOF):
                break
            if tok.normalized in ("yesterday", "today", "last", "this", "on"):
                break
            
            word_tok = self._accept_wordish()
            if word_tok:
                words.append(word_tok.normalized or word_tok.lexeme)
                tokens_used.append(word_tok.index)
                if start_span is None:
                    start_span = word_tok.span[0]
                end_span = word_tok.span[1]
            else:
                break
        
        if words:
            query = " ".join(words)
            provenance = SpanInfo((start_span, end_span), tuple(tokens_used))
            return query, provenance
        return None, None
    
    # --- Speaker parsing ---
    
    def _parse_speaker_modifier(self):
        self.rule_path.append("speaker_modifier")
        saved = self._save()
        
        # "my messages" / "your messages"
        if self._at_normalized("my"):
            tok = self._match_normalized("my")
            if self._match_normalized("messages", "responses"):
                self.speaker = SpeakerSpec(role="user", provenance=SpanInfo(tok.span, (tok.index,)))
                self.rule_path.append("speaker_my_messages")
                return
            self._restore(saved)
        
        if self._at_normalized("your"):
            tok = self._match_normalized("your")
            if self._match_normalized("messages", "responses"):
                self.speaker = SpeakerSpec(role="assistant", provenance=SpanInfo(tok.span, (tok.index,)))
                self.rule_path.append("speaker_your_messages")
                return
            self._restore(saved)
    
    # --- Temporal/Deictic parsing with disambiguation ---
    
    def _parse_temporal_or_deictic(self):
        """Combined temporal/deictic parsing with 'last time' disambiguation."""
        self.rule_path.append("temporal_or_deictic")
        
        # CRITICAL: Check for "last time" (deictic) BEFORE "last week" (temporal)
        if self._at_normalized("last"):
            if self._peek(1).kind == TokenKind.KW_TIME:
                last_tok = self._match_normalized("last")
                time_tok = self._match(TokenKind.KW_TIME)
                self.deictic = DeicticSpec(
                    kind="last_time",
                    provenance=SpanInfo((last_tok.span[0], time_tok.span[1]), (last_tok.index, time_tok.index))
                )
                self.rule_path.append("deictic:last_time_disambiguated")
                return
        
        # "the last time" variant
        if self._at_normalized("the"):
            saved = self._save()
            self._match_normalized("the")
            if self._at_normalized("last") and self._peek(1).kind == TokenKind.KW_TIME:
                last_tok = self._match_normalized("last")
                time_tok = self._match(TokenKind.KW_TIME)
                self.deictic = DeicticSpec(
                    kind="last_time",
                    provenance=SpanInfo((last_tok.span[0], time_tok.span[1]), (last_tok.index, time_tok.index))
                )
                self.rule_path.append("deictic:the_last_time")
                return
            self._restore(saved)
        
        self._parse_temporal_modifier()
        self._parse_deictic_modifier()
    
    def _parse_temporal_modifier(self):
        if self._at_normalized("yesterday"):
            tok = self._match_normalized("yesterday")
            self.temporal = TemporalSpec(
                kind="yesterday", raw="yesterday",
                provenance=SpanInfo(tok.span, (tok.index,))
            )
            return
        
        if self._at_normalized("today"):
            tok = self._match_normalized("today")
            self.temporal = TemporalSpec(
                kind="today", raw="today",
                provenance=SpanInfo(tok.span, (tok.index,))
            )
            return
        
        if self._at_normalized("last"):
            saved = self._save()
            last_tok = self._match_normalized("last")
            
            if self._at_normalized("week"):
                week_tok = self._match_normalized("week")
                self.temporal = TemporalSpec(
                    kind="last_week", raw="last week",
                    provenance=SpanInfo((last_tok.span[0], week_tok.span[1]), (last_tok.index, week_tok.index))
                )
                return
            if self._at_normalized("month"):
                month_tok = self._match_normalized("month")
                self.temporal = TemporalSpec(
                    kind="last_month", raw="last month",
                    provenance=SpanInfo((last_tok.span[0], month_tok.span[1]), (last_tok.index, month_tok.index))
                )
                return
            if self._at_normalized("year"):
                year_tok = self._match_normalized("year")
                self.temporal = TemporalSpec(
                    kind="last_year", raw="last year",
                    provenance=SpanInfo((last_tok.span[0], year_tok.span[1]), (last_tok.index, year_tok.index))
                )
                return
            
            num_tok = self._match(TokenKind.NUMBER)
            if num_tok:
                days_tok = self._match_normalized("days", "day")
                if days_tok:
                    n = int(num_tok.lexeme)
                    self.temporal = TemporalSpec(
                        kind="last_n_days", raw=f"last {n} days", n=n,
                        provenance=SpanInfo((last_tok.span[0], days_tok.span[1]), (last_tok.index, num_tok.index, days_tok.index))
                    )
                    return
            
            self._restore(saved)
        
        if self._at_normalized("this"):
            saved = self._save()
            this_tok = self._match_normalized("this")
            
            if self._at_normalized("week"):
                week_tok = self._match_normalized("week")
                self.temporal = TemporalSpec(
                    kind="this_week", raw="this week",
                    provenance=SpanInfo((this_tok.span[0], week_tok.span[1]), (this_tok.index, week_tok.index))
                )
                return
            if self._at_normalized("month"):
                month_tok = self._match_normalized("month")
                self.temporal = TemporalSpec(
                    kind="this_month", raw="this month",
                    provenance=SpanInfo((this_tok.span[0], month_tok.span[1]), (this_tok.index, month_tok.index))
                )
                return
            
            self._restore(saved)
        
        if self._at(TokenKind.NUMBER):
            saved = self._save()
            num_tok = self._match(TokenKind.NUMBER)
            days_tok = self._match_normalized("days", "day")
            if days_tok:
                ago_tok = self._match_normalized("ago")
                if ago_tok:
                    n = int(num_tok.lexeme)
                    self.temporal = TemporalSpec(
                        kind="n_days_ago", raw=f"{n} days ago", n=n,
                        provenance=SpanInfo((num_tok.span[0], ago_tok.span[1]), (num_tok.index, days_tok.index, ago_tok.index))
                    )
                    return
            self._restore(saved)
        
        if self._match_normalized("on"):
            if self._at(TokenKind.ISO_DATE):
                date_tok = self._match(TokenKind.ISO_DATE)
                self.temporal = TemporalSpec(
                    kind="iso_date", raw=f"on {date_tok.lexeme}", iso_date=date_tok.lexeme,
                    provenance=SpanInfo(date_tok.span, (date_tok.index,))
                )
                return
        
        if self._at(TokenKind.ISO_DATE):
            date_tok = self._match(TokenKind.ISO_DATE)
            self.temporal = TemporalSpec(
                kind="iso_date", raw=date_tok.lexeme, iso_date=date_tok.lexeme,
                provenance=SpanInfo(date_tok.span, (date_tok.index,))
            )
    
    def _parse_deictic_modifier(self):
        if self._at(TokenKind.KW_DEICTIC):
            tok = self._match(TokenKind.KW_DEICTIC)
            self.deictic = DeicticSpec(
                kind=tok.normalized,
                provenance=SpanInfo(tok.span, (tok.index,))
            )
            self.rule_path.append(f"deictic:{tok.normalized}")
    
    # --- Target parsing ---
    
    def _parse_target(self):
        self.rule_path.append("target")
        
        self._match_normalized("about")
        
        while not self._at_end():
            tok = self._peek()
            
            if tok.kind in (TokenKind.QUESTION, TokenKind.COMMA):
                break
            
            if tok.kind == TokenKind.QUOTED:
                self.target_tokens.append(self._match(TokenKind.QUOTED))
            elif tok.kind == TokenKind.NUMBER:
                self.target_tokens.append(self._match(TokenKind.NUMBER))
            else:
                word_tok = self._accept_wordish()
                if word_tok:
                    self.target_tokens.append(word_tok)
                else:
                    break
    
    def _skip_trailing_punct(self):
        while self._at(TokenKind.QUESTION, TokenKind.COMMA):
            self.pos += 1


class ParseError(Exception):
    pass
```

---

## 6. Resolver

### 6.1 Inputs

- `ast: MQLCommand | DiscussionQuery | FreeText`
- `now_utc: datetime` — injected for determinism (timezone-aware UTC)
- `user_tz: str` — from config (default `America/Chicago`)
- `conn: sqlite3.Connection` — for segment lookup

### 6.2 ResolvedQuery

```python
@dataclass(frozen=True)
class ResolvedQuery:
    mode: str                                        # "browse", "answer", "summarize"
    target: Optional[str]                            # May be None/empty
    segment_explicit: bool                           # True if explicitly requested
    segment_query: Optional[str]                     # Raw query text
    segment_resolved_ids: Optional[List[str]]        # None | [] | [ids]
    segment_ambiguous: bool                          # True if multiple candidates found
    segment_candidates: Optional[List[dict]]         # Candidate topics if ambiguous
    temporal: Optional[Tuple[datetime, datetime]]    # UTC half-open [start, end), timezone-aware
    speaker: Optional[str]                           # None | "user" | "assistant"
    deictic: Optional[str]                           # Kind if present
    has_broadness_cue: bool                          # True if before/previously/ever present
    audit_trace: str                                 # Canonical JSON
    
    def to_dict(self) -> dict:
        return {
            "deictic": self.deictic,
            "has_broadness_cue": self.has_broadness_cue,
            "mode": self.mode,
            "segment_ambiguous": self.segment_ambiguous,
            "segment_candidates": self.segment_candidates,
            "segment_explicit": self.segment_explicit,
            "segment_query": self.segment_query,
            "segment_resolved_ids": self.segment_resolved_ids,
            "speaker": self.speaker,
            "target": self.target,
            "temporal": [t.isoformat() for t in self.temporal] if self.temporal else None,
        }
```

### 6.3 Temporal Resolution (DST-Safe)

**Output type:** `Optional[Tuple[datetime, datetime]]` where both datetimes are timezone-aware UTC.

The retrieval pipeline is responsible for formatting these to ISO8601 strings for SQLite comparison.

```python
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta


def resolve_temporal(
    spec: TemporalSpec,
    now_utc: datetime,
    user_tz: str
) -> Optional[Tuple[datetime, datetime]]:
    """
    Resolve temporal spec to UTC half-open [start, end).
    Uses zoneinfo for DST-safe computation.
    
    Returns timezone-aware datetime objects (tzinfo=UTC).
    """
    tz = ZoneInfo(user_tz)
    utc = ZoneInfo("UTC")
    local_now = now_utc.astimezone(tz)
    
    def midnight(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    today_local = midnight(local_now)
    
    if spec.kind == "yesterday":
        start = today_local - timedelta(days=1)
        end = today_local
    
    elif spec.kind == "today":
        start = today_local
        end = today_local + timedelta(days=1)
    
    elif spec.kind == "last_week":
        days_to_monday = local_now.weekday()
        this_monday = today_local - timedelta(days=days_to_monday)
        start = this_monday - timedelta(days=7)
        end = this_monday
    
    elif spec.kind == "this_week":
        days_to_monday = local_now.weekday()
        start = today_local - timedelta(days=days_to_monday)
        end = start + timedelta(days=7)
    
    elif spec.kind == "last_month":
        first_of_month = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_now.month == 1:
            start = first_of_month.replace(year=local_now.year - 1, month=12)
        else:
            start = first_of_month.replace(month=local_now.month - 1)
        end = first_of_month
    
    elif spec.kind == "this_month":
        start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_now.month == 12:
            end = start.replace(year=local_now.year + 1, month=1)
        else:
            end = start.replace(month=local_now.month + 1)
    
    elif spec.kind == "last_year":
        start = local_now.replace(year=local_now.year - 1, month=1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)
        end = local_now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    elif spec.kind == "last_n_days":
        start = today_local - timedelta(days=spec.n)
        end = today_local + timedelta(days=1)
    
    elif spec.kind == "n_days_ago":
        start = today_local - timedelta(days=spec.n)
        end = start + timedelta(days=1)
    
    elif spec.kind == "iso_date":
        from datetime import date
        d = date.fromisoformat(spec.iso_date)
        start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz)
        end = start + timedelta(days=1)
    
    else:
        return None
    
    return (start.astimezone(utc), end.astimezone(utc))
```

### 6.4 Segment Resolution with Disambiguation (CRITICAL)

Segment resolution uses exact match first, then contains fallback. **Ambiguity is explicitly handled.**

```python
@dataclass
class SegmentResolutionResult:
    """Result of segment resolution with disambiguation information."""
    normalized_query: str
    node_ids: List[str]
    is_ambiguous: bool
    candidates: Optional[List[dict]]  # All matching topics if ambiguous
    audit_notes: List[str]


def resolve_segment(
    conn: sqlite3.Connection,
    query: str
) -> SegmentResolutionResult:
    """
    Resolve segment query to node IDs with explicit disambiguation.
    
    Resolution order:
    1. Normalized exact match → single result
    2. Normalized contains match:
       - Single match → use it
       - Multiple matches → return ambiguous with all candidates
       - No matches → return empty
    
    Tie-breaking (for ambiguous display): topic.id ASC (deterministic)
    """
    norm_query = query.lower().replace('-', ' ').replace('_', ' ').strip()
    audit_notes = []
    
    topics = get_all_topics(conn)
    
    # Phase 1: Exact match
    exact_matches = []
    for topic in topics:
        norm_name = topic['name'].lower().replace('-', ' ').replace('_', ' ')
        if norm_name == norm_query:
            exact_matches.append(topic)
    
    if len(exact_matches) == 1:
        topic = exact_matches[0]
        nodes, _ = get_cached_segment_nodes(conn, topic['id'])
        audit_notes.append(f"exact_match:topic_id={topic['id']}")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=nodes,
            is_ambiguous=False,
            candidates=None,
            audit_notes=audit_notes
        )
    
    if len(exact_matches) > 1:
        # Multiple exact matches (rare, but handle deterministically)
        audit_notes.append(f"multiple_exact_matches:count={len(exact_matches)}")
        # Sort by id ASC for deterministic selection
        exact_matches.sort(key=lambda t: t['id'])
        topic = exact_matches[0]
        nodes, _ = get_cached_segment_nodes(conn, topic['id'])
        audit_notes.append(f"tie_break:selected_topic_id={topic['id']}")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=nodes,
            is_ambiguous=True,
            candidates=[{'id': t['id'], 'name': t['name']} for t in exact_matches],
            audit_notes=audit_notes
        )
    
    # Phase 2: Contains match
    contains_matches = []
    for topic in topics:
        if norm_query in topic['name'].lower():
            contains_matches.append(topic)
    
    if len(contains_matches) == 0:
        audit_notes.append("no_match")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=[],
            is_ambiguous=False,
            candidates=None,
            audit_notes=audit_notes
        )
    
    if len(contains_matches) == 1:
        topic = contains_matches[0]
        nodes, _ = get_cached_segment_nodes(conn, topic['id'])
        audit_notes.append(f"contains_match:topic_id={topic['id']}")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=nodes,
            is_ambiguous=False,
            candidates=None,
            audit_notes=audit_notes
        )
    
    # Multiple contains matches → AMBIGUOUS
    audit_notes.append(f"ambiguous_contains:count={len(contains_matches)}")
    contains_matches.sort(key=lambda t: t['id'])  # Deterministic order
    
    return SegmentResolutionResult(
        normalized_query=norm_query,
        node_ids=[],  # Empty because ambiguous - retrieval should not proceed
        is_ambiguous=True,
        candidates=[{'id': t['id'], 'name': t['name']} for t in contains_matches],
        audit_notes=audit_notes
    )
```

### 6.5 FreeText Resolver Behavior (EXPLICIT)

When the parser returns `FreeText`, the resolver produces a fully-specified `ResolvedQuery`:

```python
def _resolve_freetext(self, ast: FreeText) -> ResolvedQuery:
    """
    FreeText → well-defined ResolvedQuery with explicit defaults.
    """
    return ResolvedQuery(
        mode="answer",
        target=ast.text,  # s_norm
        segment_explicit=False,
        segment_query=None,
        segment_resolved_ids=None,
        segment_ambiguous=False,
        segment_candidates=None,
        temporal=None,
        speaker=None,
        deictic=None,
        has_broadness_cue=False,
        audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
    )
```

### 6.6 DiscussionQuery Resolver Behavior

When the parser returns `DiscussionQuery`, the resolver produces:

```python
def _resolve_discussion_query(self, ast: DiscussionQuery) -> ResolvedQuery:
    """
    DiscussionQuery → ResolvedQuery with BROWSE mode, no segment scope.
    """
    target = ast.target.text if ast.target else None
    
    temporal = None
    if ast.temporal:
        temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)
    
    # Speaker: map "both" → None
    speaker = None
    if ast.speaker:
        if ast.speaker.role == "both":
            speaker = None
        else:
            speaker = ast.speaker.role
    
    return ResolvedQuery(
        mode="browse",  # ALWAYS browse for discussion queries
        target=target,
        segment_explicit=False,  # NEVER segment scope for discussion queries
        segment_query=None,
        segment_resolved_ids=None,
        segment_ambiguous=False,
        segment_candidates=None,
        temporal=temporal,
        speaker=speaker,
        deictic=None,
        has_broadness_cue=ast.has_broadness_cue,
        audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
    )
```

### 6.7 Speaker Resolution Mapping (EXPLICIT)

The AST `SpeakerSpec.role` maps to `ResolvedQuery.speaker` as follows:

| SpeakerSpec.role | ResolvedQuery.speaker | Retrieval behavior |
|------------------|----------------------|-------------------|
| "user" | "user" | Restrict to user messages only |
| "assistant" | "assistant" | Restrict to assistant messages only |
| "both" | None | No speaker restriction (both channels enabled) |
| (not present) | None | No speaker restriction |

**Rationale:** The retrieval pipeline contract specifies `speaker: None | "user" | "assistant"`. The "both" value from "we/us/our" pronouns means "no restriction", which is represented as `None`.

### 6.8 Full Resolver

```python
class Resolver:
    def __init__(self, conn: sqlite3.Connection, now_utc: datetime, user_tz: str):
        self.conn = conn
        self.now_utc = now_utc
        self.user_tz = user_tz
    
    def resolve(self, ast: Union[MQLCommand, DiscussionQuery, FreeText]) -> ResolvedQuery:
        if isinstance(ast, FreeText):
            return self._resolve_freetext(ast)
        
        if isinstance(ast, DiscussionQuery):
            return self._resolve_discussion_query(ast)
        
        # MQLCommand
        target = ast.target.text if ast.target else None
        
        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)
        
        # Segment (explicit gate with disambiguation)
        segment_query = None
        segment_resolved_ids = None
        segment_ambiguous = False
        segment_candidates = None
        
        if ast.segment.explicit:
            segment_query = ast.segment.query
            result = resolve_segment(self.conn, segment_query)
            segment_resolved_ids = result.node_ids
            segment_ambiguous = result.is_ambiguous
            segment_candidates = result.candidates
        
        # Speaker (map "both" → None)
        speaker = None
        if ast.speaker:
            if ast.speaker.role == "both":
                speaker = None
            else:
                speaker = ast.speaker.role
        
        deictic = ast.deictic.kind if ast.deictic else None
        
        return ResolvedQuery(
            mode=ast.mode.value,
            target=target,
            segment_explicit=ast.segment.explicit,
            segment_query=segment_query,
            segment_resolved_ids=segment_resolved_ids,
            segment_ambiguous=segment_ambiguous,
            segment_candidates=segment_candidates,
            temporal=temporal,
            speaker=speaker,
            deictic=deictic,
            has_broadness_cue=False,  # Only set by DiscussionQuery
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
        )
    
    def _resolve_freetext(self, ast: FreeText) -> ResolvedQuery:
        return ResolvedQuery(
            mode="answer",
            target=ast.text,
            segment_explicit=False,
            segment_query=None,
            segment_resolved_ids=None,
            segment_ambiguous=False,
            segment_candidates=None,
            temporal=None,
            speaker=None,
            deictic=None,
            has_broadness_cue=False,
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
        )
    
    def _resolve_discussion_query(self, ast: DiscussionQuery) -> ResolvedQuery:
        target = ast.target.text if ast.target else None
        
        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)
        
        speaker = None
        if ast.speaker:
            if ast.speaker.role == "both":
                speaker = None
            else:
                speaker = ast.speaker.role
        
        return ResolvedQuery(
            mode="browse",
            target=target,
            segment_explicit=False,
            segment_query=None,
            segment_resolved_ids=None,
            segment_ambiguous=False,
            segment_candidates=None,
            temporal=temporal,
            speaker=speaker,
            deictic=None,
            has_broadness_cue=ast.has_broadness_cue,
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
        )
```

---

## 7. User-Level CLI Examples

| Input | AST Kind | mode | target | segment.explicit | speaker | temporal | broadness_cue |
|-------|----------|------|--------|------------------|---------|----------|---------------|
| `when we discussed coffee yesterday` | DiscussionQuery | browse | coffee | false | - | yesterday | false |
| `have we talked about research before` | DiscussionQuery | browse | research | false | - | - | true |
| `did you ever mention Fenway` | DiscussionQuery | browse | fenway | false | assistant | - | true |
| `browse in topic: weapon balance last week` | MQLCommand | browse | - | true | - | last_week | - |
| `did you say 'Fenway'?` | DiscussionQuery | browse | Fenway | false | assistant | - | false |
| `did we say 'Fenway'?` | DiscussionQuery | browse | Fenway | false | None | - | false |
| `summarize topic: episodic retrieval` | MQLCommand | summarize | - | true | - | - | - |
| `browse on 2025-01-24 coffee` | MQLCommand | browse | coffee | false | - | iso_date | - |
| `do you remember the thing from before` | FreeText | answer | (full text) | - | - | - | - |
| `the last time we discussed coffee` | DiscussionQuery | browse | coffee | false | - | - | false (deictic) |

---

## 8. Integration Contract to Retrieval Pipeline

### 8.1 Field Semantics

| Field | Semantics |
|-------|-----------|
| `mode` | One of {answer, browse, summarize} |
| `target` | May be None/empty; retrieval decides handling |
| `segment_resolved_ids=None` | No segment scope requested → search globally |
| `segment_resolved_ids=[]` | Segment scope requested but no match OR ambiguous → return empty/prompt |
| `segment_resolved_ids=[ids]` | Apply segment scope to both channels |
| `segment_ambiguous=True` | Multiple candidates found; `segment_candidates` contains options |
| `temporal` | None, or half-open UTC interval [start, end) as timezone-aware datetimes |
| `speaker` | None (both channels), or "user"/"assistant" (restricts to one channel) |
| `deictic` | May be present; retrieval may ignore but must not reinterpret |
| `has_broadness_cue` | True if before/previously/ever present; informational only |

### 8.2 Temporal Type Contract

The `temporal` field contains `Optional[Tuple[datetime, datetime]]` where:
- Both datetimes are timezone-aware with `tzinfo=ZoneInfo("UTC")`
- The interval is half-open: `[start, end)`
- **The retrieval pipeline is responsible for formatting to ISO8601 strings** for SQLite comparison

### 8.3 Speaker Restriction Contract

| ResolvedQuery.speaker | Retrieval behavior |
|----------------------|-------------------|
| None | Enable both semantic and lexical channels |
| "user" | Disable semantic retrieval; lexical only on user messages |
| "assistant" | Disable semantic retrieval; lexical only on assistant messages |

### 8.4 Segment Ambiguity Contract

When `segment_ambiguous=True`:
- `segment_resolved_ids` is `[]` (empty)
- `segment_candidates` contains all matching topics
- Retrieval should NOT proceed automatically
- UI should prompt user to disambiguate

### 8.5 FreeText Contract

When parser produces `FreeText`, the resolver guarantees:
- `mode="answer"`
- `target=s_norm` (full normalized input)
- `segment_explicit=False`, `segment_resolved_ids=None`
- `temporal=None`, `speaker=None`, `deictic=None`
- `has_broadness_cue=False`

### 8.6 DiscussionQuery Contract

When parser produces `DiscussionQuery`, the resolver guarantees:
- `mode="browse"` (always)
- `segment_explicit=False` (always)
- `segment_resolved_ids=None` (always)
- `has_broadness_cue` reflects presence of before/previously/ever

### 8.7 Prohibition: Implicit Politeness Stripping

**No downstream stage may rewrite the user's text** (e.g., remove "please", remove "do you remember") unless it is an explicitly specified transformation in the normalizer/lexer/parser.

This preserves auditability.

---

## 9. Golden Fixtures

### 9.1 Fixture Format

```yaml
- id: <unique_id>
  input_raw: "<raw user input>"
  input_norm: "<after normalization>"
  tokens:
    - {kind: "KW_WHEN", lexeme: "when", span: [0, 4], normalized: "when", index: 0}
    - ...
  parse:
    ast_kind: MQLCommand | DiscussionQuery | FreeText
    rule_path: ["discussion_query:when_we", ...]
    mode: browse
    segment: {explicit: false, query: null}
    speaker: null
    temporal: {kind: "yesterday", raw: "yesterday"}
    deictic: null
    target: "coffee"
    has_broadness_cue: false
  resolve:
    now_utc: "2026-01-25T12:00:00Z"
    user_tz: "America/Chicago"
    mode: browse
    target: "coffee"
    segment_explicit: false
    segment_resolved_ids: null
    segment_ambiguous: false
    temporal_utc: ["2026-01-24T06:00:00Z", "2026-01-25T06:00:00Z"]
    speaker: null
    deictic: null
    has_broadness_cue: false
```

### 9.2 Discussion Query Fixtures (NEW)

```yaml
# "Have we" forms
- id: have_we_discussed
  input_raw: "have we discussed research"
  parse:
    ast_kind: DiscussionQuery
    query_form: "have_we"
    target: "research"
    has_broadness_cue: false
  resolve:
    mode: browse
    segment_explicit: false
    target: "research"

- id: have_we_ever_talked_about_before
  input_raw: "have we ever talked about research before"
  parse:
    ast_kind: DiscussionQuery
    query_form: "have_we"
    target: "research"
    has_broadness_cue: true  # "ever" and "before"
  resolve:
    mode: browse
    segment_explicit: false
    has_broadness_cue: true

- id: have_we_mentioned_previously
  input_raw: "have we mentioned coffee previously"
  parse:
    ast_kind: DiscussionQuery
    target: "coffee"
    has_broadness_cue: true
  resolve:
    has_broadness_cue: true

# "When we" forms
- id: when_we_discussed
  input_raw: "when we discussed coffee"
  parse:
    ast_kind: DiscussionQuery
    query_form: "when_we"
    segment: {explicit: false}
```

### 9.3 Minimal Pairs (Prevent Scope Widening)

```yaml
# Discussion-query vs explicit topic
- id: discussion_query_no_segment
  input_raw: "when we discussed coffee"
  parse:
    ast_kind: DiscussionQuery
    segment: {explicit: false}

- id: explicit_topic_has_segment
  input_raw: "in topic: coffee"
  parse:
    ast_kind: MQLCommand
    segment: {explicit: true, query: "coffee"}

# Speaker symmetric forms
- id: did_you_say
  input_raw: "did you say 'x'"
  parse:
    ast_kind: DiscussionQuery
    speaker: {role: "assistant"}
  resolve:
    speaker: "assistant"

- id: did_i_say
  input_raw: "did I say 'x'"
  parse:
    ast_kind: DiscussionQuery
    speaker: {role: "user"}
  resolve:
    speaker: "user"

- id: did_we_say
  input_raw: "did we say 'x'"
  parse:
    ast_kind: DiscussionQuery
    speaker: {role: "both"}
  resolve:
    speaker: null  # "both" maps to None
```

### 9.4 Discourse Marker Fixtures (NEW)

```yaml
# "before" is broadness cue, NOT temporal
- id: before_is_broadness_cue
  input_raw: "have we talked about this before"
  parse:
    ast_kind: DiscussionQuery
    temporal: null  # NOT a temporal filter
    has_broadness_cue: true
  resolve:
    temporal: null
    has_broadness_cue: true

# "ever" is broadness cue
- id: ever_is_broadness_cue
  input_raw: "did we ever discuss research"
  parse:
    ast_kind: DiscussionQuery
    has_broadness_cue: true
  resolve:
    has_broadness_cue: true

# "previously" is broadness cue
- id: previously_is_broadness_cue
  input_raw: "we previously mentioned this"
  parse:
    # May fall to FreeText if no verb match, or parse if recognized
    has_broadness_cue: true
```

### 9.5 Segment Ambiguity Fixtures (NEW)

```yaml
# Assume topics: "research-methodology", "research-results", "meeting-notes"
- id: segment_ambiguous_contains
  input_raw: "in topic: research"
  parse:
    ast_kind: MQLCommand
    segment: {explicit: true, query: "research"}
  resolve:
    segment_explicit: true
    segment_resolved_ids: []  # Empty because ambiguous
    segment_ambiguous: true
    segment_candidates:
      - {id: "t1", name: "research-methodology"}
      - {id: "t2", name: "research-results"}

- id: segment_exact_match
  input_raw: "in topic: research-methodology"
  resolve:
    segment_resolved_ids: ["node1", "node2", ...]
    segment_ambiguous: false
    segment_candidates: null
```

### 9.6 DST Boundary Fixtures (America/Chicago)

```yaml
# Spring forward: March 10, 2024 (2am becomes 3am)
- id: dst_spring_forward_yesterday
  input_raw: "yesterday"
  resolve:
    now_utc: "2024-03-10T15:00:00Z"
    user_tz: "America/Chicago"
    temporal_utc: ["2024-03-09T06:00:00Z", "2024-03-10T05:00:00Z"]

# Fall back: November 3, 2024 (2am repeats)
- id: dst_fall_back_yesterday
  input_raw: "yesterday"
  resolve:
    now_utc: "2024-11-03T15:00:00Z"
    user_tz: "America/Chicago"
    temporal_utc: ["2024-11-02T05:00:00Z", "2024-11-03T06:00:00Z"]
```

### 9.7 Quoting and Punctuation Fixtures

```yaml
# Smart quotes normalize to ASCII
- id: smart_quotes
  input_raw: ""coffee""
  input_norm: '"coffee"'
  tokens:
    - {kind: "QUOTED", lexeme: '"coffee"', normalized: "coffee"}

# Apostrophe in word
- id: apostrophe_word
  input_raw: "don't"
  tokens:
    - {kind: "WORD", lexeme: "don't", normalized: "don't"}

# Hyphenated name
- id: hyphenated_name
  input_raw: "Ralph-Wiggum"
  tokens:
    - {kind: "WORD", lexeme: "Ralph-Wiggum", normalized: "ralph-wiggum"}

# Possessive with apostrophe
- id: possessive_apostrophe
  input_raw: "Fenway's"
  tokens:
    - {kind: "WORD", lexeme: "Fenway's", normalized: "fenway's"}
```

### 9.8 Temporal vs Deictic Disambiguation

```yaml
# "last time" is deictic, NOT temporal
- id: last_time_is_deictic
  input_raw: "the last time we discussed coffee"
  parse:
    ast_kind: DiscussionQuery
    temporal: null
    deictic: {kind: "last_time"}

# "last week" is temporal, NOT deictic
- id: last_week_is_temporal
  input_raw: "last week"
  parse:
    temporal: {kind: "last_week", raw: "last week"}
    deictic: null
```

### 9.9 FreeText Fallback Fixtures

```yaml
- id: freetext_fallback
  input_raw: "do you remember the thing from before"
  parse:
    ast_kind: FreeText
    parse_error: "unrecognized_pattern"
  resolve:
    mode: "answer"
    target: "do you remember the thing from before"
    segment_explicit: false
    segment_resolved_ids: null
    has_broadness_cue: false
```

### 9.10 Lex Error Forces FreeText Fixtures (CRITICAL)

```yaml
# Unknown character in modifier region - MUST NOT parse as structured
- id: lex_error_unknown_char_forces_freetext
  input_raw: "browse in topic: foo@bar"
  tokens:
    - {kind: "KW_MODE", lexeme: "browse", ...}
    - {kind: "KW_IN", lexeme: "in", ...}
    - {kind: "KW_SEGMENT", lexeme: "topic", ...}
    - {kind: "COLON", lexeme: ":", ...}
    - {kind: "WORD", lexeme: "foo", ...}
    - {kind: "LEX_ERROR", lexeme: "@", normalized: "unknown_char", ...}
    - {kind: "WORD", lexeme: "bar", ...}
  parse:
    ast_kind: FreeText  # NOT MQLCommand - lex error forces fallback
    parse_error: "lex_error:unknown_char"

# Leading special character - MUST NOT silently drop
- id: lex_error_leading_special_char
  input_raw: "@browse coffee"
  tokens:
    - {kind: "LEX_ERROR", lexeme: "@", normalized: "unknown_char", ...}
    - {kind: "KW_MODE", lexeme: "browse", ...}
    - {kind: "WORD", lexeme: "coffee", ...}
  parse:
    ast_kind: FreeText
    parse_error: "lex_error:unknown_char"

# Unclosed quote - already covered but explicit
- id: lex_error_unclosed_quote
  input_raw: 'browse "unclosed'
  parse:
    ast_kind: FreeText
    parse_error: "lex_error:lex_error_unclosed_quote"
```

---

## 10. Unit Test Plan

### 10.1 Normalizer Tests

- Smart quotes → ASCII quotes
- Em/en dash → hyphen
- NBSP → space
- Whitespace collapse
- Trim leading/trailing
- Audit record includes both s_raw and s_norm

### 10.2 Lexer Tests

- QUOTED with spans and token indices
- ISO_DATE strict matching (YYYY-MM-DD)
- Apostrophes stay in WORD: `don't` → single WORD token
- Hyphens stay in WORD: `Ralph-Wiggum` → single WORD token
- Possessives: `Fenway's` → single WORD token
- Unclosed quote → LEX_ERROR
- Punctuation emits distinct tokens: COLON, COMMA, QUESTION, LPAREN, RPAREN
- KW_HAVE for "have/has"
- KW_DISCOURSE for "before/previously/already"
- Unknown characters emit LEX_ERROR, never silently dropped

### 10.3 Parser Tests

- Mode parsing (explicit prefix)
- **DiscussionQuery for "when we discussed X"**
- **DiscussionQuery for "have we talked about X"**
- **DiscussionQuery for "did I/you/we say X"**
- Explicit segment modifiers produce MQLCommand with segment.explicit=True
- DiscussionQuery NEVER has segment.explicit=True
- Speaker symmetric forms (did I/you/we say)
- Temporal forms (relative + ISO date)
- **"last time" disambiguation**: parses as deictic, NOT temporal
- **"before/previously/already/ever" set has_broadness_cue, NOT temporal**
- **Any LEX_ERROR forces FreeText** (invariant)
- Failure paths → FreeText with parse_error
- Token indices preserved in AST for provenance

### 10.4 Resolver Tests

- Temporal → UTC half-open [start, end) as timezone-aware datetimes
- DST fixtures pass
- Segment gating: called iff segment.explicit
- **Segment disambiguation: single match → ids, multiple → ambiguous with candidates**
- Target propagation unchanged
- **Speaker "both" → None mapping**
- **DiscussionQuery → mode="browse", segment_explicit=False always**
- **FreeText produces well-defined ResolvedQuery**
- **has_broadness_cue propagated from DiscussionQuery**

---

## 11. Integration Test Plan (Pipeline Success Criteria)

This section maps each pipeline success criterion to concrete tests and golden fixture IDs.

**Conventions:**
- Unit tests (pure): `tests/unit/test_lexer.py`, `test_parser.py`, `test_resolver.py`
- Integration tests (SQLite + pipeline wiring): `tests/integration/test_pipeline.py`, `test_migration.py`
- Golden fixtures: `tests/fixtures/golden/*.json`
- Each fixture records: input, now_utc, tokens (type, span, lexeme, normalized), AST (or FreeText), resolved query (mode/target/speaker/segment tri-state/temporal interval)
- Fixture IDs are filenames like `G001.json`

### 11.1 Migration Idempotence (FTS rebuild + triggers + exclusive lock)

**Tests:**

1. `test_migration_fts5_idempotent_triggers_present`
   - Type: integration (SQLite on temp file)
   - Setup: create nodes table + minimal rows; run `migrate_fts5` twice on a migration connection (`isolation_level=None`)
   - Assert:
     - `sqlite_master` contains exactly one each of `nodes_fts_ai/ad/au` triggers (no duplicates)
     - `nodes_fts` exists and is fts5

2. `test_migration_fts5_rebuild_matches_nodes_content`
   - Type: integration
   - Setup: insert N nodes before migration; run `migrate_fts5`; query `nodes_fts` count and a few rowid/content pairs
   - Assert: `nodes_fts(rowid,content)` matches `nodes(rowid,content)` for all rows (or for a sampled join)

3. `test_migration_begin_exclusive_enforced_blocks_concurrent_writer`
   - Type: integration (two connections)
   - Setup: `conn_mig` begins exclusive; `conn_writer` attempts INSERT into nodes
   - Assert: writer fails/blocks with `SQLITE_BUSY` (or raises `OperationalError`). Release lock; writer succeeds.

**Golden fixtures:** none (migration is not text-derived)

### 11.2 Connection Passing (no get_connection in query path; temp tables visible; cleanup)

**Tests:**

1. `test_no_get_connection_calls_in_query_path`
   - Type: unit (monkeypatch)
   - Setup: monkeypatch `get_connection` to raise if called; run pipeline entry that executes lexical search with filters
   - Assert: no call occurs (test passes)

2. `test_temp_table_filter_uses_same_connection_and_cleans_up`
   - Type: integration
   - Setup: build `SegmentFilter` `PENDING_IDS` with > `in_clause_max` to force temp table; execute lexical; afterwards query `sqlite_temp_master` for `seg_filter_*`
   - Assert:
     - query succeeded (so temp table was visible)
     - no `seg_filter_*` tables remain after call (cleanup)

3. `test_row_factory_invariant_not_mutated`
   - Type: unit/integration hybrid
   - Setup: create connection with `row_factory=sqlite3.Row`; run lexical search; check `conn.row_factory` is unchanged

**Golden fixtures:** none

### 11.3 BM25 Orientation (negated; sort and fusion treat larger as better)

**Tests:**

1. `test_fts_bm25_negated_orders_desc`
   - Type: integration
   - Setup: small DB with nodes containing two distinct terms; run `execute_lexical_search` for a target
   - Assert: best match appears first when ordering by `bm25_score DESC` (where `bm25_score` is `-bm25()`)

2. `test_normalization_invert_flags_correct_for_bm25_and_distance`
   - Type: unit
   - Setup: provide lexical results with `bm25_score` values and semantic results with `distance` values
   - Assert:
     - `normalize_scores(invert=False)` gives higher norm for larger `bm25_score`
     - `normalize_scores(invert=True)` gives higher norm for lower `distance`

3. `test_fusion_prefers_higher_bm25_when_semantic_missing`
   - Type: unit
   - Setup: lexical-only candidate set; semantic empty
   - Assert: fused ordering matches `bm25_score` ordering (after normalization), and missing semantic channel contributes 0

**Golden fixtures:** none (score behavior is numeric/unit)

### 11.4 Segment Scoping Tri-State (None searches all; [] returns empty)

**Tests:**

1. `test_segment_scope_none_does_not_apply_segment_filter`
   - Type: integration
   - Setup: `segment_scope=None` → `SegmentFilter` NONE; run lexical
   - Assert: results include nodes from multiple segments (or at least are not restricted)

2. `test_segment_scope_empty_returns_empty_without_retrieval`
   - Type: integration with spies/mocks OR unit on pipeline routing
   - Setup: `segment_scope=[]` → `SegmentFilter` EMPTY
   - Assert:
     - pipeline returns `[]` immediately
     - retrieval functions not called (if you implement short-circuit); otherwise called with filter EMPTY and returns `[]`

3. `test_build_segment_filter_dedupes_preserving_order`
   - Type: unit
   - Setup: `segment_node_ids` with duplicates
   - Assert: resulting `SegmentFilter` has stable-order deduped `node_ids`

**Golden fixtures:**
- `G201.json`: explicit segment requested but no match → resolved segment scope is `[]` (not None)
- `G202.json`: no segment mentioned → resolved segment scope is None

### 11.5 Ongoing Segment Cache Invalidation by effective_end Changes

**Tests:**

1. `test_segment_cache_hit_same_effective_end`
   - Type: integration
   - Setup: create topic with `end_node_id` NULL; set head to H1; call `get_cached_segment_nodes` twice
   - Assert: second call does not recompute (instrument `compute_segment_nodes` with counter or patch)

2. `test_segment_cache_invalidate_on_head_advance`
   - Type: integration
   - Setup: topic end NULL; head H1 then insert new node making head H2; call `get_cached_segment_nodes` again
   - Assert: recompute occurs and `cached.effective_end` updates from H1 to H2

3. `test_effective_end_none_fails_safe`
   - Type: integration
   - Setup: empty DB (no head); call `get_cached_segment_nodes` on an ongoing topic (or topic exists but head missing)
   - Assert: returns `([], set())` and emits AUDIT log indicating no effective end

**Golden fixtures:** none (cache behavior is structural)

### 11.6 Speaker Scope Disables Semantic; Browse Still Shows Full Exchange

**Tests:**

1. `test_speaker_scope_disables_semantic_channel`
   - Type: integration with mock chroma client
   - Setup: `ResolvedQuery.speaker` set to `'user'` or `'assistant'`; run pipeline
   - Assert: `chroma.query` not called; lexical called with role filter

2. `test_browse_mode_displays_full_exchange_under_speaker_scope`
   - Type: integration
   - Setup: fixture DB containing a user+assistant pair; query with `speaker='user'` and `mode=browse`
   - Assert: browse output includes both user and assistant turns for returned exchanges (even though matching was only on one role)

**Golden fixtures:**
- `G301.json`: "Did I ever say coffee?" → AST `speaker=user` (or explicit), resolved `speaker=user`, `mode=browse`; pipeline routes lexical-only
- `G302.json`: same but with quotes/punctuation variants to ensure stability

### 11.7 Display Consistency (metadata.assistant_id forces correct pairing when valid)

**Tests:**

1. `test_display_uses_metadata_assistant_id_when_valid`
   - Type: integration
   - Setup: create user node U with two assistant children A1 (on current ancestry) and A2 (alternate). Provide result dict with `metadata.assistant_id=A2`
   - Assert: `get_exchange_for_display` returns `assistant=A2` (and validates role+parent_id)

2. `test_display_invalid_assistant_id_falls_back_and_audits`
   - Type: integration
   - Setup: `metadata.assistant_id` points to node with wrong role or wrong `parent_id`
   - Assert: fallback selection is used; AUDIT log emitted

3. `test_fallback_prefers_current_ancestry_then_created_at`
   - Type: integration
   - Setup: multiple assistant children; only one on current ancestry
   - Assert: picks ancestry one; if none, picks earliest `created_at`

**Golden fixtures:** none (display pairing is structural, not parse-derived)

### 11.8 Temporal Filtering Half-Open; Boundary Correctness; Missing Timestamp Drops

**Tests:**

1. `test_temporal_half_open_includes_start_excludes_end`
   - Type: unit
   - Setup: `start_utc`, `end_utc`; semantic results with timestamps exactly at start and exactly at end
   - Assert: start included, end excluded

2. `test_temporal_missing_timestamp_drops_with_audit`
   - Type: unit
   - Setup: semantic result missing `metadata.timestamp`
   - Assert: dropped; AUDIT log emitted

3. `test_temporal_dst_safe_yesterday_boundaries_chicago`
   - Type: unit with zoneinfo
   - Setup: freeze `now_utc` near DST transition dates; resolve "yesterday"
   - Assert: produced UTC interval corresponds to local midnight boundaries; still half-open

**Golden fixtures:**
- `G401.json`: "yesterday coffee" with frozen `now_utc` (non-DST day)
- `G402.json`: "yesterday coffee" with frozen `now_utc` on DST start weekend
- `G403.json`: "yesterday coffee" with frozen `now_utc` on DST end weekend

### 11.9 Determinism (same query → identical ordering)

**Tests:**

1. `test_prepare_for_fusion_deterministic_sort_tiebreakers`
   - Type: unit
   - Setup: semantic results with equal distance; lexical with equal `bm25_score`
   - Assert: ordering breaks ties by `exchange_id` exactly as specified

2. `test_fusion_deterministic_end_to_end`
   - Type: unit/integration hybrid
   - Setup: fixed semantic+lexical inputs; run `fuse_results` twice
   - Assert: identical list of `exchange_id`s in same order; identical `final_score` values

3. `test_pipeline_repeatability_with_fixed_fixtures`
   - Type: integration
   - Setup: fixed SQLite fixture DB + mocked chroma returning fixed ordering; run pipeline twice
   - Assert: identical outputs (including grouping if browse)

**Golden fixtures:**
- `G501.json`: ambiguous query that produces ties to ensure tie-breaking is exercised (include fixed chroma distances and bm25 scores in the fixture's "expected pipeline inputs" if you snapshot at that layer)

### 11.10 Empty Target Handling (browse returns recent; answer/summarize empty)

**Tests:**

1. `test_empty_target_browse_returns_recent_exchanges`
   - Type: integration
   - Setup: `mode=browse`, target empty/whitespace, DB with several exchanges
   - Assert: returns most recent exchanges by `created_at DESC` (respecting segment/temporal if supplied)

2. `test_empty_target_answer_returns_empty_without_llm`
   - Type: integration with llm mock
   - Setup: `mode=answer`, target empty/whitespace
   - Assert: returns empty (or the exact empty-retrieval string if that's the spec at this layer) and does not call LLM

3. `test_empty_target_summarize_returns_empty_without_llm`
   - Type: integration with llm mock
   - Setup: `mode=summarize`, target empty/whitespace
   - Assert: "No conversations found to summarize." and no LLM call

**Golden fixtures:**
- `G601.json`: "/browse" (or equivalent) producing empty target
- `G602.json`: "summarize" with no target text
- `G603.json`: whitespace-only input routed to browse

### 11.11 Major-Bug Safety Net Fixtures

These guard the exact failure modes that cause production issues:

**Golden fixtures:**
- `G701.json`: "-foo" ensures dash is not silently dropped; parser returns FreeText (or a deliberate AST) but never a partially structured lie
- `G702.json`: "/recall topic: we research" ensures `KW_SPEAKER` doesn't truncate segment query; `segment.explicit=true`; `segment_query` captures full phrase including "we"

### 11.12 CI Enforcement Strategy

To make this mechanically enforceable in CI:

1. **Single enumerating test**: `test_golden_fixtures_all_match`
   - Enumerates all `tests/fixtures/golden/*.json` files
   - For each fixture, asserts token stream, AST, and resolution match exactly
   - Fails fast on any mismatch with diff output

2. **Criterion-specific tests**: Each test in §11.1-11.10 can be smaller and only assert criterion-specific postconditions, relying on the golden fixture enumeration for full coverage.

---

## 12. Open Extension Points (Non-goals in v1.8)

**Explicitly excluded** (reserved for future versions):

- Month-name date parsing ("Jan 5", "January 5")
- Nested quoting and escape sequences
- Rich deictic resolution ("that one", "the earlier answer")
- Multi-command chaining ("browse X; summarize")
- Anchored temporal "before X" forms ("before yesterday", "before January")
- Scored topic matching (Jaccard + embedding similarity)
- Raw→normalized offset mapping

These are explicitly fixture-covered to lock current behavior.

---

## 13. Implementation Checklist

- [ ] Implement normalizer as pure function returning `(s_norm, NormalizationAudit)`
- [ ] Lexer emits tokens with spans into s_norm AND token indices; on lex error, mark stream invalid
- [ ] Lexer emits distinct punctuation tokens (COLON, COMMA, etc.), NOT generic PUNCT
- [ ] Lexer WORD pattern allows internal hyphens/apostrophes/underscores
- [ ] Lexer emits KW_TIME for "time" to enable deictic disambiguation
- [ ] Lexer emits KW_HAVE for "have/has"
- [ ] Lexer emits KW_DISCOURSE for "before/previously/already"
- [ ] Parser produces MQLCommand, DiscussionQuery, or FreeText
- [ ] Parser handles "have we" discussion-query forms
- [ ] Parser handles "when we" discussion-query forms
- [ ] Parser handles "did I/you/we" discussion-query forms
- [ ] Parser treats before/previously/already/ever as broadness cues, NOT temporal
- [ ] Parser disambiguates "last time" (deictic) from "last week" (temporal)
- [ ] DiscussionQuery AST node guarantees mode=BROWSE, segment.explicit=False
- [ ] All AST nodes include SpanInfo provenance with source_tokens
- [ ] Resolver requires injected now_utc (timezone-aware)
- [ ] Resolver resolves temporal to UTC half-open as timezone-aware datetimes
- [ ] Resolver gates segment lookup on segment.explicit
- [ ] Resolver handles segment ambiguity with candidates
- [ ] Resolver maps SpeakerSpec(role="both") → speaker=None
- [ ] Resolver produces well-defined ResolvedQuery for FreeText input
- [ ] Resolver produces well-defined ResolvedQuery for DiscussionQuery input
- [ ] AuditInfo includes both s_raw and s_norm
- [ ] Golden fixtures frozen and run in CI
- [ ] Minimal pairs + DST + punctuation + disambiguation + ambiguity fixtures included
