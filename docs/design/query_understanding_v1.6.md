# Episodic Query Understanding (MQL) — v1.6

**Status:** Implementation spec for the query-understanding front-end (lexer → parser → resolver) that drives the Episodic retrieval pipeline.

This version is a corrective rewrite of v1.5. It tightens ambiguity control, specifies normalization and span conventions, expands temporal coverage, and upgrades golden fixtures into a regression-grade suite. It remains intentionally scoped to query understanding; retrieval is an external consumer with an explicit contract.

---

## 0. Changelog (v1.5 → v1.6)

1. **Lexer normalization and span conventions**
   - Added Unicode punctuation normalization (smart quotes, dashes, NBSP) prior to lexing.
   - Declared span convention: codepoint offsets into the post-normalization string.

2. **Grammar/precedence hardening (ambiguity control)**
   - Locked down "discussion-query" forms (e.g., "when we discussed X") to NEVER imply segment scope.
   - Added symmetric speaker constructions: "did I say …", "did you say …", "did we say …".
   - Declared deterministic precedence ordering for modifiers (mode → segment → speaker → temporal → deictic → target).

3. **Temporal surface-form specification**
   - Explicitly supports relative time (yesterday/last week/etc.) and ISO dates (YYYY-MM-DD), with optional "on".
   - Explicitly does NOT support natural-language month names in v1.6 (falls to FreeText), fixture-covered.
   - Resolver emits half-open UTC intervals [start, end).

4. **Deictic handling**
   - Introduced explicit Deictic AST nodes for a small, auditable subset: "earlier", "before", "previous", "last time".
   - Everything else deictic remains FreeText (fixture-covered), with a clear retrieval contract.

5. **Golden fixtures upgraded**
   - Added minimal pairs to prevent accidental scope widening.
   - Added DST-boundary fixtures for America/Chicago.
   - Added punctuation/quoting fixtures (smart quotes, apostrophes, hyphens).
   - Added canonical JSON serialization requirements for stable golden files.

6. **Integration contract hardened**
   - Prohibited "UX politeness stripping" unless it is part of the normalizer/lexer.
   - Defined exact allowed None/empty behaviors per field.

---

## 1. User-Facing Mental Model

The user can type either:

**A) A structured "memory query command" (MQL)**, such as:
- `browse when we discussed coffee yesterday`
- `summarize in topic: weapons balance last week`
- `did you say 'Fenway'?`

**B) Free text** that is not reliably parseable as MQL.
- The system produces `FreeText(input)` and retrieval treats it as an unscoped target string, subject to the contract in Section 8.

**Key safety rule:** A discussion-query ("when we discussed X", "talked about X", "mentioned X") is NOT segment scope. Segment scope only occurs with explicit segment syntax.

---

## 2. Pipeline Overview

```
Input (raw user text)
  → Normalizer (Unicode + whitespace normalization)
  → Lexer (token stream with spans into s_norm)
  → Parser (AST: MQLCommand | FreeText)
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

### 3.3 Span Convention

**All token spans are `(start, end)` codepoint offsets into `s_norm` (post-normalization).**

Audit output MUST include `s_norm` so spans are interpretable.

---

## 4. Lexical Specification

### 4.1 Token Types

```python
class TokenKind(Enum):
    # Literals (never reinterpreted)
    QUOTED = auto()       # "..." or '...' (quotes stripped in value)
    ISO_DATE = auto()     # YYYY-MM-DD (strict)
    NUMBER = auto()       # Sequence of digits
    
    # Punctuation
    COLON = auto()        # :
    COMMA = auto()        # ,
    QUESTION = auto()     # ?
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    DASH = auto()         # - (standalone, not in word)
    
    # Keywords (soft - may be reinterpreted as WORD)
    KW_MODE = auto()      # browse, summarize, answer
    KW_SEGMENT = auto()   # topic, segment, topics, segments
    KW_SPEAKER = auto()   # i, me, my, you, your, we, us, our, user, assistant
    KW_TIME_REL = auto()  # yesterday, today, last, this, week, month, year, ago, days
    KW_DISCUSS = auto()   # discussed, talked, mention, mentioned, brought, bring, said, say, asked, ask
    KW_WHEN = auto()      # when, where
    KW_DID = auto()       # did
    KW_EVER = auto()      # ever
    KW_IN = auto()        # in, within
    KW_ON = auto()        # on
    KW_ABOUT = auto()     # about
    KW_DEICTIC = auto()   # earlier, before, previous
    
    # Default
    WORD = auto()         # Any other alphanumeric sequence
    
    # Special
    EOF = auto()
    LEX_ERROR = auto()    # Unclosed quote, invalid character


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str                 # Original substring from s_norm
    span: Tuple[int, int]       # (start, end) into s_norm
    normalized: Optional[str]   # Canonical form (for keywords)
    
    def as_word(self) -> 'Token':
        """Reinterpret as WORD (soft keyword handling)."""
        return Token(TokenKind.WORD, self.lexeme, self.span, self.lexeme.lower())
    
    def to_dict(self) -> dict:
        """Stable serialization."""
        return {
            "kind": self.kind.name,
            "lexeme": self.lexeme,
            "span": list(self.span),
            "normalized": self.normalized
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
    "ever": (TokenKind.KW_EVER, "ever"),
    
    # Scope
    "in": (TokenKind.KW_IN, "in"),
    "within": (TokenKind.KW_IN, "in"),
    "on": (TokenKind.KW_ON, "on"),
    "about": (TokenKind.KW_ABOUT, "about"),
    
    # Deictic
    "earlier": (TokenKind.KW_DEICTIC, "earlier"),
    "before": (TokenKind.KW_DEICTIC, "before"),
    "previous": (TokenKind.KW_DEICTIC, "previous"),
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
            return Token(TokenKind.QUOTED, lexeme, (start, self.pos), value)
        self.pos += 1
    
    # Unclosed quote
    lexeme = self.s_norm[start:self.pos]
    return Token(TokenKind.LEX_ERROR, lexeme, (start, self.pos), "lex_error_unclosed_quote")
```

### 4.4 ISO Date Recognition

Strict `YYYY-MM-DD` format only. Month names are NOT supported in v1.6.

```python
ISO_DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')

def _try_scan_iso_date(self) -> Optional[Token]:
    match = ISO_DATE_PATTERN.match(self.s_norm, self.pos)
    if match:
        lexeme = match.group(0)
        self.pos = match.end()
        return Token(TokenKind.ISO_DATE, lexeme, (match.start(), match.end()), lexeme)
    return None
```

### 4.5 Word Characters

```python
def _is_word_char(self, c: str) -> bool:
    """
    Word characters: alphanumeric, hyphen, underscore, apostrophe (internal).
    Allows: "don't", "last-week", "topic_name"
    """
    return c.isalnum() or c in ('-', '_', "'")
```

### 4.6 Lexer Implementation

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
    
    def tokenize(self) -> LexResult:
        while self.pos < len(self.s_norm):
            self._skip_whitespace()
            if self.pos >= len(self.s_norm):
                break
            self._scan_token()
        
        self.tokens.append(Token(TokenKind.EOF, "", (self.pos, self.pos), None))
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
        start = self.pos
        char = self.s_norm[self.pos]
        
        # Punctuation
        if char == ':':
            self._emit(TokenKind.COLON, start, start + 1)
        elif char == ',':
            self._emit(TokenKind.COMMA, start, start + 1)
        elif char == '?':
            self._emit(TokenKind.QUESTION, start, start + 1)
        elif char == '(':
            self._emit(TokenKind.LPAREN, start, start + 1)
        elif char == ')':
            self._emit(TokenKind.RPAREN, start, start + 1)
        elif char in ('"', "'"):
            tok = self._scan_quoted(char)
            self.tokens.append(tok)
            if tok.kind == TokenKind.LEX_ERROR:
                self.has_error = True
                self.error_code = tok.normalized
            return
        elif char == '-' and (self.pos + 1 >= len(self.s_norm) or self.s_norm[self.pos + 1] == ' '):
            # Standalone dash
            self._emit(TokenKind.DASH, start, start + 1)
        else:
            # Try ISO date first
            tok = self._try_scan_iso_date()
            if tok:
                self.tokens.append(tok)
                return
            
            # Try number
            if char.isdigit():
                self._scan_number()
                return
            
            # Word or keyword
            self._scan_word()
            return
        
        self.pos += 1
    
    def _emit(self, kind: TokenKind, start: int, end: int):
        lexeme = self.s_norm[start:end]
        self.tokens.append(Token(kind, lexeme, (start, end), lexeme.lower()))
    
    def _scan_number(self):
        start = self.pos
        while self.pos < len(self.s_norm) and self.s_norm[self.pos].isdigit():
            self.pos += 1
        lexeme = self.s_norm[start:self.pos]
        self.tokens.append(Token(TokenKind.NUMBER, lexeme, (start, self.pos), lexeme))
    
    def _scan_word(self):
        start = self.pos
        while self.pos < len(self.s_norm) and self._is_word_char(self.s_norm[self.pos]):
            self.pos += 1
        lexeme = self.s_norm[start:self.pos]
        lower = lexeme.lower()
        
        if lower in KEYWORD_MAP:
            kind, normalized = KEYWORD_MAP[lower]
            self.tokens.append(Token(kind, lexeme, (start, self.pos), normalized))
        else:
            self.tokens.append(Token(TokenKind.WORD, lexeme, (start, self.pos), lower))
```

---

## 5. Grammar (Recursive Descent)

The parser is **deterministic** and must not backtrack exponentially.
If parsing fails at any required point, return `FreeText(s_norm)`, along with the token stream and an error code.

### 5.1 AST Types

```python
class Mode(Enum):
    BROWSE = "browse"
    SUMMARIZE = "summarize"
    ANSWER = "answer"


@dataclass(frozen=True)
class SegmentSpec:
    explicit: bool
    query: Optional[str]  # Raw segment query text


@dataclass(frozen=True)
class SpeakerSpec:
    role: str  # "user", "assistant", or "both"


@dataclass(frozen=True)
class TemporalSpec:
    kind: str  # "yesterday", "last_week", "iso_date", etc.
    raw: str
    iso_date: Optional[str] = None  # For ISO dates
    n: Optional[int] = None         # For "N days ago"


@dataclass(frozen=True)
class DeicticSpec:
    kind: str  # "earlier", "before", "previous", "last_time"


@dataclass(frozen=True)
class TargetSpec:
    text: str
    spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class AuditInfo:
    s_norm: str
    tokens: List[dict]          # Serialized token stream
    rule_path: List[str]        # Parser rules taken
    decisions: List[str]        # Soft-keyword reinterpretations


@dataclass(frozen=True)
class MQLCommand:
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

### 5.2 Precedence and Ordering (Critical)

Parsing priority is structural, not "pattern order":

1. **Mode phrase** (if present; else default depends on subsequent forms)
2. **Explicit segment modifier** (topic/segment syntax only)
3. **Explicit speaker modifier** (did-I-say / did-you-say / did-we-say)
4. **Temporal modifier**
5. **Deictic modifier**
6. **Target extraction**

**Discussion-query forms ("when we discussed X") are treated as mode selection + target form. They MUST NOT produce `segment.explicit = True`.**

### 5.3 Mode Phrases

Accepted explicit mode prefixes:
- `browse ...`
- `summarize ...`
- `answer ...`

If no explicit mode phrase:
- Discussion-query forces `BROWSE`
- Otherwise default is `ANSWER`

### 5.4 Segment Modifiers (Explicit Only)

**Accepted explicit segment forms** (`segment.explicit = True`):

1. `in topic: <segment_query>`
2. `in topic <segment_query>`
3. `topic: <segment_query>`
4. `topic <segment_query>` — **NOT accepted** (requires colon or `in`)
5. `segment: <segment_query>`
6. `in segment <segment_query>`

`segment_query` is:
- A `QUOTED` literal, OR
- A sequence of WORD tokens up to the next recognized modifier boundary

**NOT accepted as segment modifiers** (these are discussion-query forms):
- `when we discussed X`
- `we talked about X`
- `mentioned X`

### 5.5 Discussion-Query Forms (Browse + Target; NOT Segment)

These forms force:
- `mode = BROWSE` (if not explicitly specified)
- `segment.explicit = False`
- `target` extracted as X

**Accepted forms:**
- `when we discussed <X>`
- `when we talked about <X>`
- `when we mentioned <X>`
- `when did we discuss <X>`
- `did we ever discuss <X>`
- `the last time we discussed <X>` (also sets deictic)

**This rule is NON-NEGOTIABLE: discussion-query semantics NEVER imply segment scope.**

### 5.6 Speaker Forms

**Symmetric speaker constructions:**

| Form | Speaker |
|------|---------|
| `did I say <X>` | user |
| `did I ask <X>` | user |
| `did you say <X>` | assistant |
| `did you mention <X>` | assistant |
| `did we say <X>` | both (None) |
| `my messages` / `my responses` | user |
| `your messages` / `your responses` | assistant |

### 5.7 Temporal Forms

**Supported in v1.6:**

| Form | Kind |
|------|------|
| `yesterday` | yesterday |
| `today` | today |
| `last week` | last_week |
| `this week` | this_week |
| `last month` | last_month |
| `this month` | this_month |
| `last N days` | last_n_days |
| `N days ago` | n_days_ago |
| `on YYYY-MM-DD` | iso_date |
| `YYYY-MM-DD` | iso_date |

**NOT supported in v1.6 (falls to FreeText or becomes target):**
- `on Jan 5` — Month names not parsed as temporal
- `January 5` — Month names not parsed as temporal

This is explicitly fixture-covered to lock behavior.

### 5.8 Deictic Forms

**Supported in v1.6:**
- `earlier` → `DeicticSpec(kind="earlier")`
- `before` → `DeicticSpec(kind="before")`
- `previous` → `DeicticSpec(kind="previous")`
- `the last time` → `DeicticSpec(kind="last_time")`

**NOT supported (falls to FreeText):**
- `that one`
- `the earlier answer`

### 5.9 Parser Implementation

```python
class Parser:
    def __init__(self, lex_result: LexResult):
        self.tokens = lex_result.tokens
        self.s_norm = lex_result.s_norm
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
        
        # Check for lex error
        self.lex_error = lex_result.has_error
        self.lex_error_code = lex_result.error_code
    
    def parse(self) -> Union[MQLCommand, FreeText]:
        # Lex error forces FreeText
        if self.lex_error:
            return self._make_freetext(self.lex_error_code)
        
        try:
            self._parse_mode_phrase()
            self._parse_segment_modifier()
            self._parse_speaker_modifier()
            self._parse_temporal_modifier()
            self._parse_deictic_modifier()
            self._parse_target()
            self._skip_trailing_punct()
            return self._build_command()
        except ParseError as e:
            return self._make_freetext(str(e))
    
    def _make_freetext(self, error: str) -> FreeText:
        return FreeText(
            text=self.s_norm,
            parse_error=error,
            audit=AuditInfo(
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
            target = TargetSpec(text=text, spans=spans)
        
        return MQLCommand(
            mode=self.mode or Mode.ANSWER,
            segment=self.segment,
            speaker=self.speaker,
            temporal=self.temporal,
            deictic=self.deictic,
            target=target,
            audit=AuditInfo(
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
        
        # Explicit mode prefix
        if self._at(TokenKind.KW_MODE):
            tok = self._match(TokenKind.KW_MODE)
            self.mode = Mode(tok.normalized)
            self.rule_path.append(f"explicit_mode:{self.mode.value}")
            return
        
        # Discussion-query (forces BROWSE, no segment)
        if self._try_discussion_query():
            return
        
        # Did-speaker-say (forces BROWSE)
        if self._peek().normalized == "did":
            # Will be handled in speaker modifier
            pass
    
    def _try_discussion_query(self) -> bool:
        """Match: when/where (did)? (we)? (discuss|talked|mentioned) (about)?"""
        saved = self._save()
        
        if not self._match_normalized("when", "where"):
            return False
        
        self._match_normalized("did")  # optional
        self._match_normalized("we", "i", "you")  # optional
        self._match(TokenKind.KW_DISCUSS)  # optional
        self._match_normalized("about")  # optional
        
        self.mode = Mode.BROWSE
        self.segment = SegmentSpec(explicit=False, query=None)  # Explicit: NO segment scope
        self.rule_path.append("discussion_query:browse,no_segment")
        return True
    
    # --- Segment parsing ---
    
    def _parse_segment_modifier(self):
        self.rule_path.append("segment_modifier")
        saved = self._save()
        
        # "topic:" or "segment:"
        if self._at(TokenKind.KW_SEGMENT):
            seg_tok = self._match(TokenKind.KW_SEGMENT)
            if self._match(TokenKind.COLON):
                query = self._parse_segment_query()
                if query:
                    self.segment = SegmentSpec(explicit=True, query=query)
                    self.rule_path.append(f"explicit_segment:{query}")
                    return
            self._restore(saved)
        
        # "in topic X" or "in segment X"
        if self._match_normalized("in"):
            if self._at(TokenKind.KW_SEGMENT):
                self._match(TokenKind.KW_SEGMENT)
                self._match(TokenKind.COLON)  # optional colon
                query = self._parse_segment_query()
                if query:
                    self.segment = SegmentSpec(explicit=True, query=query)
                    self.rule_path.append(f"explicit_segment_in:{query}")
                    return
            # "in" without topic/segment — NOT segment scope
            self._restore(saved)
    
    def _parse_segment_query(self) -> Optional[str]:
        """Parse segment query: QUOTED or words until modifier boundary."""
        if self._at(TokenKind.QUOTED):
            tok = self._match(TokenKind.QUOTED)
            return tok.normalized
        
        words = []
        for _ in range(5):  # Max 5 words
            tok = self._peek()
            # Stop at modifier boundaries
            if tok.kind in (TokenKind.KW_TIME_REL, TokenKind.KW_SPEAKER, TokenKind.KW_DEICTIC,
                           TokenKind.ISO_DATE, TokenKind.COLON, TokenKind.EOF):
                break
            if tok.normalized in ("yesterday", "today", "last", "this", "on"):
                break
            
            word_tok = self._accept_wordish()
            if word_tok:
                words.append(word_tok.normalized or word_tok.lexeme)
            else:
                break
        
        return " ".join(words) if words else None
    
    # --- Speaker parsing ---
    
    def _parse_speaker_modifier(self):
        self.rule_path.append("speaker_modifier")
        saved = self._save()
        
        # "did I/you/we say/ask"
        if self._match_normalized("did"):
            speaker_tok = self._match_normalized("i", "me", "you", "we")
            if speaker_tok:
                self._match_normalized("ever")  # optional
                verb_tok = self._match(TokenKind.KW_DISCUSS)
                if verb_tok:
                    self._match_normalized("about")  # optional
                    self.mode = Mode.BROWSE
                    
                    if speaker_tok.normalized in ("i", "me", "user"):
                        self.speaker = SpeakerSpec(role="user")
                    elif speaker_tok.normalized in ("you", "your", "assistant"):
                        self.speaker = SpeakerSpec(role="assistant")
                    # "we" → speaker=None (both)
                    
                    self.rule_path.append(f"speaker_did:{self.speaker.role if self.speaker else 'both'}")
                    return
            self._restore(saved)
        
        # "my messages" / "your messages"
        if self._at_normalized("my"):
            self._match_normalized("my")
            if self._match_normalized("messages", "responses"):
                self.speaker = SpeakerSpec(role="user")
                self.rule_path.append("speaker_my_messages")
                return
            self._restore(saved)
        
        if self._at_normalized("your"):
            self._match_normalized("your")
            if self._match_normalized("messages", "responses"):
                self.speaker = SpeakerSpec(role="assistant")
                self.rule_path.append("speaker_your_messages")
                return
            self._restore(saved)
    
    # --- Temporal parsing ---
    
    def _parse_temporal_modifier(self):
        self.rule_path.append("temporal_modifier")
        
        # "yesterday" / "today"
        if self._at_normalized("yesterday"):
            self._match_normalized("yesterday")
            self.temporal = TemporalSpec(kind="yesterday", raw="yesterday")
            return
        
        if self._at_normalized("today"):
            self._match_normalized("today")
            self.temporal = TemporalSpec(kind="today", raw="today")
            return
        
        # "last week/month" or "last N days"
        if self._at_normalized("last"):
            saved = self._save()
            self._match_normalized("last")
            
            if self._match_normalized("week"):
                self.temporal = TemporalSpec(kind="last_week", raw="last week")
                return
            if self._match_normalized("month"):
                self.temporal = TemporalSpec(kind="last_month", raw="last month")
                return
            if self._match_normalized("year"):
                self.temporal = TemporalSpec(kind="last_year", raw="last year")
                return
            
            num_tok = self._match(TokenKind.NUMBER)
            if num_tok:
                if self._match_normalized("days", "day"):
                    n = int(num_tok.lexeme)
                    self.temporal = TemporalSpec(kind="last_n_days", raw=f"last {n} days", n=n)
                    return
            
            self._restore(saved)
        
        # "this week/month"
        if self._at_normalized("this"):
            saved = self._save()
            self._match_normalized("this")
            
            if self._match_normalized("week"):
                self.temporal = TemporalSpec(kind="this_week", raw="this week")
                return
            if self._match_normalized("month"):
                self.temporal = TemporalSpec(kind="this_month", raw="this month")
                return
            
            self._restore(saved)
        
        # "N days ago"
        if self._at(TokenKind.NUMBER):
            saved = self._save()
            num_tok = self._match(TokenKind.NUMBER)
            if self._match_normalized("days", "day"):
                if self._match_normalized("ago"):
                    n = int(num_tok.lexeme)
                    self.temporal = TemporalSpec(kind="n_days_ago", raw=f"{n} days ago", n=n)
                    return
            self._restore(saved)
        
        # "on YYYY-MM-DD" or just "YYYY-MM-DD"
        if self._match_normalized("on"):
            if self._at(TokenKind.ISO_DATE):
                date_tok = self._match(TokenKind.ISO_DATE)
                self.temporal = TemporalSpec(kind="iso_date", raw=f"on {date_tok.lexeme}", iso_date=date_tok.lexeme)
                return
        
        if self._at(TokenKind.ISO_DATE):
            date_tok = self._match(TokenKind.ISO_DATE)
            self.temporal = TemporalSpec(kind="iso_date", raw=date_tok.lexeme, iso_date=date_tok.lexeme)
    
    # --- Deictic parsing ---
    
    def _parse_deictic_modifier(self):
        self.rule_path.append("deictic_modifier")
        
        if self._at(TokenKind.KW_DEICTIC):
            tok = self._match(TokenKind.KW_DEICTIC)
            self.deictic = DeicticSpec(kind=tok.normalized)
            return
        
        # "the last time"
        if self._at_normalized("the"):
            saved = self._save()
            self._match_normalized("the")
            if self._match_normalized("last"):
                if self._match_normalized("time"):
                    self.deictic = DeicticSpec(kind="last_time")
                    return
            self._restore(saved)
    
    # --- Target parsing ---
    
    def _parse_target(self):
        self.rule_path.append("target")
        
        # Skip leading "about"
        self._match_normalized("about")
        
        while not self._at_end():
            tok = self._peek()
            
            # Stop at trailing punctuation
            if tok.kind in (TokenKind.QUESTION, TokenKind.COMMA):
                break
            
            # Collect QUOTED, NUMBER, WORD, or soft keywords
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

- `ast: MQLCommand | FreeText`
- `now_utc: datetime` — injected for determinism
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
    temporal: Optional[Tuple[datetime, datetime]]    # UTC half-open [start, end)
    speaker: Optional[str]                           # None | "user" | "assistant"
    deictic: Optional[str]                           # Kind if present
    audit_trace: str                                 # Canonical JSON
    
    def to_dict(self) -> dict:
        return {
            "deictic": self.deictic,
            "mode": self.mode,
            "segment_explicit": self.segment_explicit,
            "segment_query": self.segment_query,
            "segment_resolved_ids": self.segment_resolved_ids,
            "speaker": self.speaker,
            "target": self.target,
            "temporal": [t.isoformat() for t in self.temporal] if self.temporal else None,
        }
```

### 6.3 Temporal Resolution (DST-Safe)

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
        start = tz.localize(datetime(d.year, d.month, d.day, 0, 0, 0))
        end = start + timedelta(days=1)
    
    else:
        return None
    
    return (start.astimezone(utc), end.astimezone(utc))
```

### 6.4 Segment Resolution Gating

Resolver MUST call segment resolver **only when `segment.explicit == True`**.

```python
def resolve_segment(
    conn: sqlite3.Connection,
    query: str
) -> Tuple[str, List[str]]:
    """
    Resolve segment query to node IDs.
    Returns (normalized_query, node_ids).
    
    If no match: node_ids = [] (NOT None).
    """
    norm_query = query.lower().replace('-', ' ').replace('_', ' ').strip()
    
    topics = get_all_topics(conn)
    
    # Exact match
    for topic in topics:
        norm_name = topic['name'].lower().replace('-', ' ').replace('_', ' ')
        if norm_name == norm_query:
            nodes, _ = get_cached_segment_nodes(conn, topic['id'])
            return (norm_query, nodes)
    
    # Contains match
    for topic in topics:
        if norm_query in topic['name'].lower():
            nodes, _ = get_cached_segment_nodes(conn, topic['id'])
            return (norm_query, nodes)
    
    return (norm_query, [])
```

### 6.5 Full Resolver

```python
class Resolver:
    def __init__(self, conn: sqlite3.Connection, now_utc: datetime, user_tz: str):
        self.conn = conn
        self.now_utc = now_utc
        self.user_tz = user_tz
    
    def resolve(self, ast: Union[MQLCommand, FreeText]) -> ResolvedQuery:
        if isinstance(ast, FreeText):
            return ResolvedQuery(
                mode="answer",
                target=ast.text,
                segment_explicit=False,
                segment_query=None,
                segment_resolved_ids=None,
                temporal=None,
                speaker=None,
                deictic=None,
                audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
            )
        
        # Target
        target = ast.target.text if ast.target else None
        
        # Temporal
        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)
        
        # Segment (explicit gate)
        segment_query = None
        segment_resolved_ids = None
        
        if ast.segment.explicit:
            segment_query = ast.segment.query
            _, segment_resolved_ids = resolve_segment(self.conn, segment_query)
        # If not explicit: segment_resolved_ids stays None (tri-state)
        
        # Speaker
        speaker = ast.speaker.role if ast.speaker else None
        
        # Deictic
        deictic = ast.deictic.kind if ast.deictic else None
        
        return ResolvedQuery(
            mode=ast.mode.value,
            target=target,
            segment_explicit=ast.segment.explicit,
            segment_query=segment_query,
            segment_resolved_ids=segment_resolved_ids,
            temporal=temporal,
            speaker=speaker,
            deictic=deictic,
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
        )
```

---

## 7. User-Level CLI Examples

| Input | mode | target | segment.explicit | speaker | temporal |
|-------|------|--------|------------------|---------|----------|
| `when we discussed coffee yesterday` | browse | coffee | false | - | yesterday |
| `browse in topic: weapon balance last week` | browse | - | true (query="weapon balance") | - | last_week |
| `did you say 'Fenway'?` | browse | Fenway | false | assistant | - |
| `summarize topic: episodic retrieval` | summarize | - | true | - | - |
| `browse on 2025-01-24 coffee` | browse | coffee | false | - | iso_date |
| `do you remember the thing from before` | FreeText | (full text) | - | - | - |

---

## 8. Integration Contract to Retrieval Pipeline

### 8.1 Field Semantics

| Field | Semantics |
|-------|-----------|
| `mode` | One of {answer, browse, summarize} |
| `target` | May be None/empty; retrieval decides handling |
| `segment_resolved_ids=None` | No segment scope requested → search globally |
| `segment_resolved_ids=[]` | Segment scope requested but no match → return empty immediately |
| `segment_resolved_ids=[ids]` | Apply segment scope to both channels |
| `temporal` | None, or half-open UTC interval [start, end) |
| `speaker` | None, or "user"/"assistant" → disables semantic retrieval |
| `deictic` | May be present; retrieval may ignore but must not reinterpret |

### 8.2 Prohibition: Implicit Politeness Stripping

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
    - {kind: "KW_WHEN", lexeme: "when", span: [0, 4], normalized: "when"}
    - ...
  parse:
    ast_kind: MQLCommand | FreeText
    rule_path: ["mode_phrase", "discussion_query:browse,no_segment", ...]
    mode: browse
    segment: {explicit: false, query: null}
    speaker: null
    temporal: {kind: "yesterday", raw: "yesterday"}
    deictic: null
    target: "coffee"
  resolve:
    now_utc: "2026-01-25T12:00:00Z"
    user_tz: "America/Chicago"
    mode: browse
    target: "coffee"
    segment_explicit: false
    segment_resolved_ids: null
    temporal_utc: ["2026-01-24T06:00:00Z", "2026-01-25T06:00:00Z"]
    speaker: null
    deictic: null
```

### 9.2 Minimal Pairs (Prevent Scope Widening)

```yaml
# Discussion-query vs explicit topic
- id: discussion_query_no_segment
  input_raw: "when we discussed coffee"
  parse:
    segment: {explicit: false}

- id: explicit_topic_has_segment
  input_raw: "in topic: coffee"
  parse:
    segment: {explicit: true, query: "coffee"}

# Speaker symmetric forms
- id: did_you_say
  input_raw: "did you say 'x'"
  parse:
    speaker: {role: "assistant"}

- id: did_i_say
  input_raw: "did I say 'x'"
  parse:
    speaker: {role: "user"}

- id: did_we_say
  input_raw: "did we say 'x'"
  parse:
    speaker: null  # both

# Segment delimiter variants
- id: topic_colon_space
  input_raw: "topic: foo"
  parse:
    segment: {explicit: true, query: "foo"}

- id: topic_colon_nospace
  input_raw: "topic:foo"
  parse:
    segment: {explicit: true, query: "foo"}
```

### 9.3 DST Boundary Fixtures (America/Chicago)

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

### 9.4 Quoting and Punctuation Fixtures

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

# Em dash normalizes to hyphen
- id: em_dash
  input_raw: "last—week"
  input_norm: "last-week"
  tokens:
    - {kind: "WORD", lexeme: "last-week"}

# Unclosed quote forces FreeText
- id: unclosed_quote
  input_raw: 'browse "unclosed'
  parse:
    ast_kind: FreeText
    parse_error: "lex_error_unclosed_quote"
```

### 9.5 Unsupported Temporal Forms

```yaml
# Month names NOT supported in v1.6 → treated as target
- id: month_name_not_temporal
  input_raw: "on Jan 5 coffee"
  parse:
    ast_kind: MQLCommand
    temporal: null  # "Jan" not recognized as temporal
    target: "jan 5 coffee"
```

---

## 10. Unit Test Plan

### 10.1 Normalizer Tests

- Smart quotes → ASCII quotes
- Em/en dash → hyphen
- NBSP → space
- Whitespace collapse
- Trim leading/trailing

### 10.2 Lexer Tests

- QUOTED with spans
- ISO_DATE strict matching (YYYY-MM-DD)
- Apostrophes stay in WORD
- Unclosed quote → LEX_ERROR

### 10.3 Parser Tests

- Mode parsing (explicit prefix, discussion-query)
- Explicit segment modifiers ONLY set segment.explicit
- Discussion-query NEVER sets segment.explicit
- Speaker symmetric forms
- Temporal forms (relative + ISO date)
- Failure paths → FreeText with parse_error

### 10.4 Resolver Tests

- Temporal → UTC half-open [start, end)
- DST fixtures pass
- Segment gating: called iff segment.explicit
- Segment tri-state: explicit-but-unmatched → [], not explicit → None
- Target propagation unchanged

---

## 11. Open Extension Points (Non-goals in v1.6)

**Explicitly excluded** (reserved for future versions):

- Month-name date parsing ("Jan 5", "January 5")
- Nested quoting and escape sequences
- Rich deictic resolution ("that one", "the earlier answer")
- Multi-command chaining ("browse X; summarize")

These are explicitly fixture-covered to lock current behavior.

---

## 12. Implementation Checklist

- [ ] Implement normalizer as pure function returning `(s_norm, NormalizationAudit)`
- [ ] Lexer emits tokens with spans into s_norm; on lex error, mark stream invalid
- [ ] Parser produces MQLCommand or FreeText with rule_path audit
- [ ] Resolver requires injected now_utc
- [ ] Resolver resolves temporal to UTC half-open
- [ ] Resolver gates segment lookup on segment.explicit
- [ ] Golden fixtures frozen and run in CI
- [ ] Minimal pairs + DST + punctuation fixtures included
