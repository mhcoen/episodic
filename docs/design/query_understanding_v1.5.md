# Query Understanding Specification v1.5
## Tokenizer + Recursive-Descent Parser (Soft Keywords) + Resolver

This document specifies the user-facing "memory query" language (MQL) for Episodic: how free-form user requests like "when we discussed coffee yesterday" are deterministically parsed into a `ResolvedQuery` used by the retrieval pipeline.

**Changes from v1.4:**
- Clarified the contract between Query Understanding and Retrieval (inputs/outputs, fail-closed semantics).
- Made "soft keywords" and backtracking rules explicit and testable (`accept_soft`, `accept_wordish`).
- Added an explicit grammar (EBNF + precedence) and error-recovery rules.
- Tightened lexer rules for quotes, unicode whitespace, and hyphen/underscore handling.
- Added a complete, criterion-aligned test plan (8 criteria with unit tests + golden fixtures).
- Added explicit scope/non-goals.

---

## 1. Scope and Non-Goals

### 1.1 Scope
- Deterministically map a single user utterance into either:
  - `MQLCommand` AST → `ResolvedQuery`, or
  - `FreeText` node → treated as "unstructured search target" by Retrieval (no implicit scope).
- Support "mode" (browse/answer/summarize), "modifiers" (temporal/segment/speaker/deictic), and "target" extraction.
- Preserve spans for audit/debug: every AST leaf can be traced back to source offsets.

### 1.2 Non-Goals
- No semantic paraphrasing or rewriting of the user's target beyond trivial normalization (whitespace, quote stripping).
- No learning-based disambiguation inside the deterministic pipeline.
- No multi-turn clarifications here (that is a higher-level UX/policy layer).

---

## 2. Data Types and Pipeline Contract

### 2.1 Token

```python
@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str                 # Verbatim slice from input
    norm: str                   # Normalized form (lowercase, escapes processed)
    span: Tuple[int, int]       # Half-open [start, end) into original string
    
    def as_word(self) -> 'Token':
        """Reinterpret this token as WORD (for soft keyword handling)."""
        return Token(TokenType.WORD, self.lexeme, self.norm, self.span)
    
    def to_tuple(self) -> Tuple[str, str, int, int]:
        """Stable serialization for audit logging."""
        return (self.type.name, self.lexeme, self.span[0], self.span[1])
```

**Span invariants:**
- Spans are half-open `[start, end)` into the original UTF-8 decoded Python string.
- Every token's `lexeme == input[start:end]`.
- Spans partition the input when including whitespace (whitespace tokens may be omitted).

### 2.2 AST

Two root node types:

```python
@dataclass(frozen=True)
class MQLCommand:
    """Successfully parsed memory query."""
    mode: str                           # "browse", "answer", "summarize"
    target: Optional[Target]            # Search terms
    temporal: Optional[Temporal]        # Time filter
    segment: Optional[Segment]          # Topic/segment scope (explicit only)
    speaker: Optional[Speaker]          # Speaker filter
    deictic: Optional[Deictic]          # "last N messages"
    audit: AuditInfo                    # Parse trace
    
    def to_dict(self) -> dict:
        """Deterministic serialization (sorted keys)."""
        ...


@dataclass(frozen=True)
class FreeText:
    """Unparseable input, routes to Retrieval as unscoped target."""
    text: str
    reason: str
    audit: AuditInfo
    
    def to_dict(self) -> dict:
        return {"type": "FreeText", "text": self.text, "reason": self.reason}


@dataclass
class AuditInfo:
    """Parse trace for debugging."""
    tokens: List[Tuple[str, str, int, int]]  # Serialized token stream
    decisions: List[str]                      # Parser decisions (soft-keyword reinterpretations, fallbacks)
    warnings: List[str]                       # Non-fatal issues (unclosed quotes, etc.)
```

### 2.3 ResolvedQuery

Returned by resolver for `MQLCommand` only.

```python
@dataclass(frozen=True)
class ResolvedQuery:
    """Final resolved query ready for retrieval pipeline."""
    mode: str                                        # "browse", "answer", "summarize"
    target: str                                      # Search terms (may be empty)
    temporal: Optional[Tuple[datetime, datetime]]    # UTC half-open [start, end)
    segment_explicit: bool                           # True if segment scope was requested
    segment_query: Optional[str]                     # Raw segment phrase for lookup
    segment_ids: Optional[List[str]]                 # None | [] | [node_ids]
    speaker: Optional[str]                           # None | "user" | "assistant"
    deictic_limit: Optional[int]                     # "last N messages" → N
    audit_trace: str                                 # Deterministic JSON serialization
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target": self.target,
            "temporal": [t.isoformat() for t in self.temporal] if self.temporal else None,
            "segment_explicit": self.segment_explicit,
            "segment_query": self.segment_query,
            "segment_ids": self.segment_ids,
            "speaker": self.speaker,
            "deictic_limit": self.deictic_limit,
        }
```

### 2.4 Fail-Closed Contract

1. **Parser failure** → Return `FreeText` rather than manufacturing scopes.
2. **Resolver failure** (e.g., bad date) → Drop invalid modifier and AUDIT-log; do not return `FreeText`.
3. **Segment lookup failure** → Return `segment_ids=[]` (requested-but-empty), not `None`.

This spec uses "drop invalid modifier and AUDIT-log" for resolver failures.

---

## 3. Lexer (Tokenizer)

### 3.1 Token Types

```python
class TokenType(Enum):
    # Literals (never reinterpreted)
    NUMBER = auto()       # 10, 5, 2025
    QUOTED = auto()       # "phrase" or 'phrase' (quotes removed in norm)
    COLON = auto()        # :
    COMMA = auto()        # ,
    QUESTION = auto()     # ?
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    EOF = auto()
    
    # Words (default)
    WORD = auto()
    
    # Soft keywords: mode starters
    KW_BROWSE = auto()      # browse, show, list, display
    KW_ANSWER = auto()      # answer
    KW_SUMMARIZE = auto()   # summarize, summary
    
    # Soft keywords: query openers
    KW_WHEN = auto()        # when
    KW_WHERE = auto()       # where
    KW_WHAT = auto()        # what
    KW_DID = auto()         # did
    
    # Soft keywords: pronouns
    KW_WE = auto()          # we, our
    KW_I = auto()           # i, me, my
    KW_YOU = auto()         # you, your
    
    # Soft keywords: verbs
    KW_DISCUSS = auto()     # discuss, discussed, talk, talked, conversation, chat, chatted
    KW_MENTION = auto()     # mention, mentioned
    KW_ASK = auto()         # ask, asked
    KW_SAY = auto()         # say, said
    KW_DECIDE = auto()      # decide, decided, conclude, concluded
    KW_EVER = auto()        # ever
    
    # Soft keywords: segment selectors (MUST be explicit to trigger scope)
    KW_TOPIC = auto()       # topic, topics
    KW_SEGMENT = auto()     # segment, segments
    KW_IN = auto()          # in, within (only scopes when paired with topic/segment)
    
    # Soft keywords: speaker selectors
    KW_ONLY = auto()        # only
    KW_USER = auto()        # user
    KW_ASSISTANT = auto()   # assistant
    
    # Soft keywords: temporal
    KW_YESTERDAY = auto()
    KW_TODAY = auto()
    KW_LAST = auto()        # last
    KW_THIS = auto()        # this
    KW_ON = auto()          # on
    KW_BETWEEN = auto()     # between
    KW_AND = auto()         # and
    KW_AGO = auto()         # ago
    
    # Soft keywords: time units
    KW_WEEK = auto()
    KW_MONTH = auto()
    KW_YEAR = auto()
    KW_DAY = auto()
    KW_DAYS = auto()
    KW_MESSAGES = auto()    # messages, exchanges, message, exchange
    
    # Month names
    MONTH_NAME = auto()     # jan, january, feb, etc.
```

**Critical property:** All `KW_*` tokens are "soft" — the parser may reinterpret them as `WORD` when a production fails.

### 3.2 Keyword Normalization Map

```python
KEYWORD_MAP: Dict[str, TokenType] = {
    # Mode starters
    "browse": TokenType.KW_BROWSE,
    "show": TokenType.KW_BROWSE,
    "list": TokenType.KW_BROWSE,
    "display": TokenType.KW_BROWSE,
    "answer": TokenType.KW_ANSWER,
    "summarize": TokenType.KW_SUMMARIZE,
    "summary": TokenType.KW_SUMMARIZE,
    
    # Query openers
    "when": TokenType.KW_WHEN,
    "where": TokenType.KW_WHERE,
    "what": TokenType.KW_WHAT,
    "did": TokenType.KW_DID,
    
    # Pronouns
    "we": TokenType.KW_WE,
    "our": TokenType.KW_WE,
    "i": TokenType.KW_I,
    "me": TokenType.KW_I,
    "my": TokenType.KW_I,
    "you": TokenType.KW_YOU,
    "your": TokenType.KW_YOU,
    
    # Verbs
    "discuss": TokenType.KW_DISCUSS,
    "discussed": TokenType.KW_DISCUSS,
    "talk": TokenType.KW_DISCUSS,
    "talked": TokenType.KW_DISCUSS,
    "conversation": TokenType.KW_DISCUSS,
    "chat": TokenType.KW_DISCUSS,
    "chatted": TokenType.KW_DISCUSS,
    "mention": TokenType.KW_MENTION,
    "mentioned": TokenType.KW_MENTION,
    "ask": TokenType.KW_ASK,
    "asked": TokenType.KW_ASK,
    "say": TokenType.KW_SAY,
    "said": TokenType.KW_SAY,
    "decide": TokenType.KW_DECIDE,
    "decided": TokenType.KW_DECIDE,
    "conclude": TokenType.KW_DECIDE,
    "concluded": TokenType.KW_DECIDE,
    "ever": TokenType.KW_EVER,
    
    # Segment selectors
    "topic": TokenType.KW_TOPIC,
    "topics": TokenType.KW_TOPIC,
    "segment": TokenType.KW_SEGMENT,
    "segments": TokenType.KW_SEGMENT,
    "in": TokenType.KW_IN,
    "within": TokenType.KW_IN,
    
    # Speaker selectors
    "only": TokenType.KW_ONLY,
    "user": TokenType.KW_USER,
    "assistant": TokenType.KW_ASSISTANT,
    
    # Temporal
    "yesterday": TokenType.KW_YESTERDAY,
    "today": TokenType.KW_TODAY,
    "last": TokenType.KW_LAST,
    "this": TokenType.KW_THIS,
    "on": TokenType.KW_ON,
    "between": TokenType.KW_BETWEEN,
    "and": TokenType.KW_AND,
    "ago": TokenType.KW_AGO,
    
    # Time units
    "week": TokenType.KW_WEEK,
    "month": TokenType.KW_MONTH,
    "year": TokenType.KW_YEAR,
    "day": TokenType.KW_DAY,
    "days": TokenType.KW_DAYS,
    "messages": TokenType.KW_MESSAGES,
    "exchanges": TokenType.KW_MESSAGES,
    "message": TokenType.KW_MESSAGES,
    "exchange": TokenType.KW_MESSAGES,
}

MONTH_NAMES = frozenset({
    "jan", "january", "feb", "february", "mar", "march",
    "apr", "april", "may", "jun", "june", "jul", "july",
    "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december"
})
```

### 3.3 Quoted Phrases

- Supports single (`'`) or double (`"`) quotes.
- `QUOTED` token is atomic; `norm` is the interior text with quotes stripped.
- **Unclosed quotes:** Lexer emits `WORD` tokens instead (fail-closed) and records an AUDIT warning.

```python
def _scan_quoted(self, quote_char: str):
    start = self.pos
    self.pos += 1  # Skip opening quote
    
    while self.pos < len(self.text):
        if self.text[self.pos] == quote_char:
            # Found closing quote
            self.pos += 1
            lexeme = self.text[start:self.pos]
            inner = lexeme[1:-1]
            self.tokens.append(Token(TokenType.QUOTED, lexeme, inner.lower(), (start, self.pos)))
            return
        self.pos += 1
    
    # Unclosed quote: emit as WORD, record warning
    lexeme = self.text[start:self.pos]
    self.tokens.append(Token(TokenType.WORD, lexeme, lexeme.lower(), (start, self.pos)))
    self.warnings.append(f"unclosed quote starting at {start}")
```

### 3.4 Hyphens and Underscores

- **Tokenization:** Treat `-` and `_` as word characters so `last-week` and `topic_name` are single tokens.
- **Segment matching (resolver):** Normalize `-` and `_` to whitespace for fuzzy matching.

```python
def _is_word_char(self, c: str) -> bool:
    return c.isalnum() or c in ('-', '_')
```

### 3.5 Whitespace and Unicode

- Normalize all Unicode whitespace to ASCII space for keyword matching and word boundaries.
- Preserve original spans; do not rewrite the source string in-place.

```python
import unicodedata

def _is_whitespace(self, c: str) -> bool:
    return c.isspace() or unicodedata.category(c) == 'Zs'
```

### 3.6 Lexer Algorithm

```python
class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []
        self.warnings: List[str] = []
    
    def tokenize(self) -> Tuple[List[Token], List[str]]:
        """
        Single left-to-right pass.
        Returns (tokens, warnings).
        """
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            self._scan_token()
        
        self.tokens.append(Token(TokenType.EOF, "", "", (self.pos, self.pos)))
        return self.tokens, self.warnings
    
    def _scan_token(self):
        start = self.pos
        char = self.text[self.pos]
        
        if char == ':':
            self._emit(TokenType.COLON, start, start + 1)
        elif char == ',':
            self._emit(TokenType.COMMA, start, start + 1)
        elif char == '?':
            self._emit(TokenType.QUESTION, start, start + 1)
        elif char == '(':
            self._emit(TokenType.LPAREN, start, start + 1)
        elif char == ')':
            self._emit(TokenType.RPAREN, start, start + 1)
        elif char in ('"', "'"):
            self._scan_quoted(char)
            return  # Already advanced pos
        elif char.isdigit():
            self._scan_number()
            return
        else:
            self._scan_word()
            return
        
        self.pos += 1
    
    def _emit(self, token_type: TokenType, start: int, end: int):
        lexeme = self.text[start:end]
        self.tokens.append(Token(token_type, lexeme, lexeme.lower(), (start, end)))
    
    def _scan_number(self):
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        lexeme = self.text[start:self.pos]
        self.tokens.append(Token(TokenType.NUMBER, lexeme, lexeme, (start, self.pos)))
    
    def _scan_word(self):
        start = self.pos
        while self.pos < len(self.text) and self._is_word_char(self.text[self.pos]):
            self.pos += 1
        lexeme = self.text[start:self.pos]
        norm = lexeme.lower()
        
        # Classify token type
        if norm in MONTH_NAMES:
            token_type = TokenType.MONTH_NAME
        elif norm in KEYWORD_MAP:
            token_type = KEYWORD_MAP[norm]
        else:
            token_type = TokenType.WORD
        
        self.tokens.append(Token(token_type, lexeme, norm, (start, self.pos)))

**Determinism requirement:** Given identical input string, produce identical token list and spans.

---

## 4. Parser

### 4.1 Stages

- **Stage A:** Tokenization only (always succeeds, may have warnings).
- **Stage B:** Parse attempt producing `MQLCommand` or `FreeText`.

Stage B must be fail-closed: if the parser cannot produce a well-formed `MQLCommand` under the grammar + soft keyword rules, return `FreeText(original_text)`.

### 4.2 Soft Keyword Rule (Critical)

Any `KW_*` token may be reinterpreted as `WORD` **only when needed to salvage parsing of a production**, and only within a bounded local scope (the current nonterminal).

**Parser helper functions:**

```python
class Parser:
    def __init__(self, tokens: List[Token], original_text: str):
        self.tokens = tokens
        self.original_text = original_text
        self.pos = 0
        self.decisions: List[str] = []  # Audit trail
    
    def accept_soft(self, *types: TokenType) -> Optional[Token]:
        """
        Accept token if it matches one of the given types.
        Does NOT reinterpret keywords as WORD.
        """
        if self._peek().type in types:
            return self._advance()
        return None
    
    def accept_wordish(self) -> Optional[Token]:
        """
        Accept WORD or any KW_* as a word.
        Records reinterpretation in audit.
        """
        tok = self._peek()
        if tok.type == TokenType.WORD:
            return self._advance()
        if tok.type.name.startswith("KW_"):
            self.decisions.append(f"reinterpreted {tok.type.name} as WORD at {tok.span}")
            self._advance()
            return tok.as_word()
        return None
    
    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]
    
    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok
    
    def _at_end(self) -> bool:
        return self._peek().type == TokenType.EOF
    
    def _save(self) -> int:
        return self.pos
    
    def _restore(self, pos: int):
        self.pos = pos
```

**Testable invariant:** For any input, if a KW_* token is reinterpreted as WORD, it MUST be recorded in `decisions` and MUST NOT create a modifier.

### 4.3 Grammar (EBNF)

```
command         ::= mode_phrase? modifiers* target_expr? trailing_punct*

mode_phrase     ::= explicit_prefix
                  | discussion_query
                  | show_phrase

explicit_prefix ::= ("browse" | "answer" | "summarize") ":"

discussion_query ::= "when" "did"? subject? verb? "about"?
                   | "where" "did"? subject? verb? "about"?
                   | "did" subject "ever"? verb "about"?
                   | "what" "did" subject verb "about"?

subject         ::= "we" | "I" | "you"

verb            ::= "discuss" | "mention" | "ask" | "say" | "decide"

show_phrase     ::= "show" "me"?
                  | "list"
                  | "display"

modifiers       ::= segment_mod | temporal_mod | speaker_mod | deictic_mod

segment_mod     ::= ("topic" | "segment") ":" phrase
                  | "in" ("topic" | "segment") phrase
                  # NOTE: bare "in <phrase>" does NOT trigger segment scope

temporal_mod    ::= "yesterday"
                  | "today"
                  | "last" "week"
                  | "last" "month"
                  | "this" "week"
                  | "this" "month"
                  | "last" NUMBER ("days" | "day")     # NOT followed by "messages"
                  | NUMBER ("days" | "weeks" | "hours") "ago"
                  | "on" date
                  | "between" date "and" date

deictic_mod     ::= "last" NUMBER ("messages" | "exchanges")

speaker_mod     ::= "only"? "what" ("I" | "you") verb
                  | ("my" | "your") ("messages" | "responses")

target_expr     ::= ("about" phrase) | phrase

phrase          ::= QUOTED | wordish+

wordish         ::= WORD | (KW_* reinterpreted as WORD)

date            ::= MONTH_NAME NUMBER (NUMBER)?

trailing_punct  ::= QUESTION | COMMA
```

**Mode precedence (strict order):**
1. Explicit prefix (`browse:`, `answer:`, `summarize:`)
2. `what did` → answer
3. `when did` / `where did` / `did I/you ever` → browse
4. `show` / `list` / `display` → browse
5. Default → answer

**Modifier parsing order:**
1. Deictic (`last N messages`) — checked BEFORE temporal to disambiguate `last`
2. Segment (`topic:`, `in topic`)
3. Temporal (`yesterday`, `last week`, etc.)
4. Speaker (`only what I said`, `my messages`)

### 4.4 Key Semantic Rules

1. **"when we discussed X …"** sets `mode=browse`, `target=X`. It does NOT imply segment scope.
2. **Segment scope** is set ONLY when `segment_mod` is present (`KW_TOPIC` or `KW_SEGMENT` explicitly).
3. **Bare "in X"** does NOT trigger segment scope — `KW_IN` alone is insufficient.
4. **Speaker modifier** sets speaker; Retrieval will disable semantic retrieval.

### 4.5 Error Recovery

Parser is permitted to:
- Ignore trailing punctuation tokens (`QUESTION`, `COMMA`) after completing a command.
- Treat unknown trailing tokens as part of `target_expr` only if doing so doesn't manufacture a modifier.

If the parser hits an unrecoverable point:
- Return `FreeText` and record the failure point in audit.

### 4.6 Parser Implementation

```python
class Parser:
    def __init__(self, tokens: List[Token], original_text: str):
        self.tokens = tokens
        self.original_text = original_text
        self.pos = 0
        self.decisions: List[str] = []
        
        # Parse state
        self.mode: Optional[str] = None
        self.target_tokens: List[Token] = []
        self.temporal: Optional[Temporal] = None
        self.segment: Optional[Segment] = None
        self.speaker: Optional[Speaker] = None
        self.deictic: Optional[Deictic] = None
    
    def parse(self) -> Union[MQLCommand, FreeText]:
        try:
            self._parse_mode_phrase()
            self._parse_modifiers()
            self._parse_target_expr()
            self._skip_trailing_punct()
            return self._build_ast()
        except ParseError as e:
            self.decisions.append(f"parse failed: {e}")
            return FreeText(
                text=self.original_text,
                reason=str(e),
                audit=AuditInfo(
                    tokens=[t.to_tuple() for t in self.tokens],
                    decisions=self.decisions,
                    warnings=[]
                )
            )
    
    def _parse_mode_phrase(self):
        """Parse mode with strict precedence."""
        
        # 1. Explicit prefix
        if self._try_explicit_prefix():
            return
        
        # 2. "what did ..." → answer
        if self._try_what_did_phrase():
            return
        
        # 3. "when/where did ..." or "did I/you ever ..." → browse
        if self._try_discussion_query():
            return
        
        # 4. "show/list/display" → browse
        if self._try_show_phrase():
            return
        
        # 5. Default
        self.mode = "answer"
    
    def _try_explicit_prefix(self) -> bool:
        saved = self._save()
        
        mode_tok = self.accept_soft(TokenType.KW_BROWSE, TokenType.KW_ANSWER, TokenType.KW_SUMMARIZE)
        if mode_tok and self.accept_soft(TokenType.COLON):
            self.mode = {
                TokenType.KW_BROWSE: "browse",
                TokenType.KW_ANSWER: "answer",
                TokenType.KW_SUMMARIZE: "summarize",
            }[mode_tok.type]
            self.decisions.append(f"explicit_prefix: {self.mode}")
            return True
        
        self._restore(saved)
        return False
    
    def _try_what_did_phrase(self) -> bool:
        """Match: what did (we|you|I) (verb) about?"""
        saved = self._save()
        
        if not self.accept_soft(TokenType.KW_WHAT):
            return False
        if not self.accept_soft(TokenType.KW_DID):
            self._restore(saved)
            return False
        
        subject = self.accept_soft(TokenType.KW_WE, TokenType.KW_YOU, TokenType.KW_I)
        if not subject:
            self._restore(saved)
            return False
        
        # Optional verb
        self.accept_soft(TokenType.KW_SAY, TokenType.KW_DISCUSS, TokenType.KW_DECIDE,
                        TokenType.KW_MENTION, TokenType.KW_ASK)
        
        # Optional "about"
        self.accept_soft(TokenType.KW_ABOUT) if hasattr(TokenType, 'KW_ABOUT') else None
        
        self.mode = "answer"
        if subject.type == TokenType.KW_YOU:
            self.speaker = Speaker(role="assistant")
        elif subject.type == TokenType.KW_I:
            self.speaker = Speaker(role="user")
        
        self.decisions.append(f"what_did_phrase: mode=answer, speaker={self.speaker}")
        return True
    
    def _try_discussion_query(self) -> bool:
        """Match: when/where did ... OR did I/you ever ..."""
        saved = self._save()
        
        # "when/where did we discuss"
        if self.accept_soft(TokenType.KW_WHEN, TokenType.KW_WHERE):
            self.accept_soft(TokenType.KW_DID)
            self.accept_soft(TokenType.KW_WE, TokenType.KW_I, TokenType.KW_YOU)
            self.accept_soft(TokenType.KW_DISCUSS, TokenType.KW_SAY, TokenType.KW_MENTION)
            # Skip "about" if present
            if self._peek().norm == "about":
                self._advance()
            self.mode = "browse"
            self.decisions.append("discussion_query: when/where")
            return True
        
        # "did I/you ever say"
        if self.accept_soft(TokenType.KW_DID):
            subject = self.accept_soft(TokenType.KW_I, TokenType.KW_YOU)
            if subject:
                self.accept_soft(TokenType.KW_EVER)
                self.accept_soft(TokenType.KW_SAY, TokenType.KW_DISCUSS, TokenType.KW_MENTION, TokenType.KW_ASK)
                # Skip "about"
                if self._peek().norm == "about":
                    self._advance()
                self.mode = "browse"
                self.speaker = Speaker(role="user" if subject.type == TokenType.KW_I else "assistant")
                self.decisions.append(f"discussion_query: did_subject, speaker={self.speaker}")
                return True
            self._restore(saved)
            return False
        
        return False
    
    def _try_show_phrase(self) -> bool:
        """Match: show me? | list | display"""
        if self.accept_soft(TokenType.KW_BROWSE):
            if self._peek().norm == "me":
                self._advance()
            self.mode = "browse"
            self.decisions.append("show_phrase")
            return True
        return False
    
    def _parse_modifiers(self):
        """Parse modifiers in precedence order until none match."""
        while not self._at_end():
            # Deictic FIRST (disambiguates "last N messages" from "last N days")
            if self._try_deictic_mod():
                continue
            if self._try_segment_mod():
                continue
            if self._try_temporal_mod():
                continue
            if self._try_speaker_mod():
                continue
            break
    
    def _try_segment_mod(self) -> bool:
        """
        Match: (topic|segment) ":" phrase
             | "in" (topic|segment) phrase
        
        CRITICAL: Bare "in <phrase>" does NOT trigger segment scope.
        """
        saved = self._save()
        
        # "topic:" or "segment:"
        if self.accept_soft(TokenType.KW_TOPIC, TokenType.KW_SEGMENT):
            if self.accept_soft(TokenType.COLON):
                phrase = self._parse_phrase(max_words=5)
                if phrase:
                    self.segment = Segment(phrase=phrase)
                    self.decisions.append(f"segment_mod: phrase={phrase}")
                    return True
            self._restore(saved)
            return False
        
        # "in topic X" or "in segment X" — REQUIRES topic/segment keyword
        if self.accept_soft(TokenType.KW_IN):
            if self.accept_soft(TokenType.KW_TOPIC, TokenType.KW_SEGMENT):
                phrase = self._parse_phrase(max_words=5)
                if phrase:
                    self.segment = Segment(phrase=phrase)
                    self.decisions.append(f"segment_mod: in_topic phrase={phrase}")
                    return True
            # "in" without topic/segment → NOT segment scope, restore
            self._restore(saved)
            return False
        
        return False
    
    def _try_temporal_mod(self) -> bool:
        """Parse temporal modifier (excludes deictic)."""
        saved = self._save()
        
        # "yesterday"
        if self.accept_soft(TokenType.KW_YESTERDAY):
            self.temporal = Temporal(kind="yesterday", raw="yesterday")
            self.decisions.append("temporal_mod: yesterday")
            return True
        
        # "today"
        if self.accept_soft(TokenType.KW_TODAY):
            self.temporal = Temporal(kind="today", raw="today")
            self.decisions.append("temporal_mod: today")
            return True
        
        # "last week/month" or "last N days"
        if self.accept_soft(TokenType.KW_LAST):
            if self.accept_soft(TokenType.KW_WEEK):
                self.temporal = Temporal(kind="last_week", raw="last week")
                self.decisions.append("temporal_mod: last_week")
                return True
            
            if self.accept_soft(TokenType.KW_MONTH):
                self.temporal = Temporal(kind="last_month", raw="last month")
                self.decisions.append("temporal_mod: last_month")
                return True
            
            num_tok = self.accept_soft(TokenType.NUMBER)
            if num_tok:
                # Check it's NOT "last N messages" (deictic)
                if self._peek().type == TokenType.KW_MESSAGES:
                    self._restore(saved)
                    return False
                
                if self.accept_soft(TokenType.KW_DAYS, TokenType.KW_DAY):
                    n = int(num_tok.norm)
                    self.temporal = Temporal(kind="last_n_days", raw=f"last {n} days", n=n)
                    self.decisions.append(f"temporal_mod: last_{n}_days")
                    return True
            
            self._restore(saved)
            return False
        
        # "this week/month"
        if self.accept_soft(TokenType.KW_THIS):
            if self.accept_soft(TokenType.KW_WEEK):
                self.temporal = Temporal(kind="this_week", raw="this week")
                self.decisions.append("temporal_mod: this_week")
                return True
            if self.accept_soft(TokenType.KW_MONTH):
                self.temporal = Temporal(kind="this_month", raw="this month")
                self.decisions.append("temporal_mod: this_month")
                return True
            self._restore(saved)
            return False
        
        # "N days/weeks ago"
        num_tok = self.accept_soft(TokenType.NUMBER)
        if num_tok:
            unit = self.accept_soft(TokenType.KW_DAYS, TokenType.KW_WEEK)
            if unit and self.accept_soft(TokenType.KW_AGO):
                n = int(num_tok.norm)
                unit_str = "days" if unit.type == TokenType.KW_DAYS else "weeks"
                self.temporal = Temporal(kind=f"{n}_{unit_str}_ago", raw=f"{n} {unit_str} ago", n=n)
                self.decisions.append(f"temporal_mod: {n}_{unit_str}_ago")
                return True
            self._restore(saved)
            return False
        
        # "on DATE"
        if self.accept_soft(TokenType.KW_ON):
            date_str = self._try_parse_date()
            if date_str:
                self.temporal = Temporal(kind="explicit_date", raw=f"on {date_str}", date1=date_str)
                self.decisions.append(f"temporal_mod: on_{date_str}")
                return True
            self._restore(saved)
            return False
        
        # "between DATE and DATE"
        if self.accept_soft(TokenType.KW_BETWEEN):
            date1 = self._try_parse_date()
            if date1 and self.accept_soft(TokenType.KW_AND):
                date2 = self._try_parse_date()
                if date2:
                    self.temporal = Temporal(
                        kind="date_range",
                        raw=f"between {date1} and {date2}",
                        date1=date1, date2=date2
                    )
                    self.decisions.append(f"temporal_mod: between_{date1}_and_{date2}")
                    return True
            self._restore(saved)
            return False
        
        return False
    
    def _try_deictic_mod(self) -> bool:
        """Match: last NUMBER (messages|exchanges)"""
        saved = self._save()
        
        if not self.accept_soft(TokenType.KW_LAST):
            return False
        
        num_tok = self.accept_soft(TokenType.NUMBER)
        if not num_tok:
            self._restore(saved)
            return False
        
        if not self.accept_soft(TokenType.KW_MESSAGES):
            self._restore(saved)
            return False
        
        self.deictic = Deictic(count=int(num_tok.norm))
        self.decisions.append(f"deictic_mod: last_{self.deictic.count}_messages")
        return True
    
    def _try_speaker_mod(self) -> bool:
        """
        Match: "only"? "what" (I|you) verb
             | (my|your) (messages|responses)
        """
        saved = self._save()
        
        # "only what I/you said"
        self.accept_soft(TokenType.KW_ONLY)  # Optional
        
        if self._peek().norm == "what":
            self._advance()
            subject = self.accept_soft(TokenType.KW_I, TokenType.KW_YOU)
            if subject and self.accept_soft(TokenType.KW_SAY, TokenType.KW_MENTION, TokenType.KW_ASK):
                role = "user" if subject.type == TokenType.KW_I else "assistant"
                self.speaker = Speaker(role=role)
                self.decisions.append(f"speaker_mod: only_what_{role}_said")
                return True
            self._restore(saved)
            return False
        
        # "my messages" / "your messages"
        if self._peek().norm == "my":
            self._advance()
            if self.accept_soft(TokenType.KW_MESSAGES) or self._peek().norm == "responses":
                if self._peek().norm == "responses":
                    self._advance()
                self.speaker = Speaker(role="user")
                self.decisions.append("speaker_mod: my_messages")
                return True
            self._restore(saved)
            return False
        
        if self._peek().norm == "your":
            self._advance()
            if self.accept_soft(TokenType.KW_MESSAGES) or self._peek().norm == "responses":
                if self._peek().norm == "responses":
                    self._advance()
                self.speaker = Speaker(role="assistant")
                self.decisions.append("speaker_mod: your_messages")
                return True
            self._restore(saved)
            return False
        
        self._restore(saved)
        return False
    
    def _parse_target_expr(self):
        """Collect remaining tokens as target."""
        # Skip leading "about"
        if self._peek().norm == "about":
            self._advance()
        
        while not self._at_end():
            tok = self._peek()
            
            # Stop at trailing punctuation
            if tok.type in (TokenType.QUESTION, TokenType.COMMA):
                break
            
            # Collect token (use accept_wordish for soft keywords)
            word_tok = self.accept_wordish()
            if word_tok:
                self.target_tokens.append(word_tok)
            elif tok.type == TokenType.QUOTED:
                self.target_tokens.append(self._advance())
            elif tok.type == TokenType.NUMBER:
                self.target_tokens.append(self._advance())
            else:
                break
    
    def _skip_trailing_punct(self):
        """Ignore trailing QUESTION, COMMA."""
        while self._peek().type in (TokenType.QUESTION, TokenType.COMMA):
            self._advance()
    
    def _parse_phrase(self, max_words: int = 5) -> Optional[str]:
        """Parse phrase: QUOTED or bare words."""
        if self._peek().type == TokenType.QUOTED:
            tok = self._advance()
            return tok.norm
        
        words = []
        for _ in range(max_words):
            tok = self.accept_wordish()
            if tok:
                words.append(tok.norm)
            else:
                break
        
        return " ".join(words) if words else None
    
    def _try_parse_date(self) -> Optional[str]:
        """Parse: MONTH_NAME NUMBER (NUMBER)?"""
        if self._peek().type != TokenType.MONTH_NAME:
            return None
        
        month = self._advance()
        day = self.accept_soft(TokenType.NUMBER)
        if not day:
            return None
        
        year = self.accept_soft(TokenType.NUMBER)
        
        if year:
            return f"{month.lexeme} {day.lexeme} {year.lexeme}"
        return f"{month.lexeme} {day.lexeme}"
    
    def _build_ast(self) -> MQLCommand:
        target = None
        if self.target_tokens:
            terms = tuple(t.norm for t in self.target_tokens)
            spans = tuple(t.span for t in self.target_tokens)
            target = Target(terms=terms, spans=spans)
        
        return MQLCommand(
            mode=self.mode or "answer",
            target=target,
            temporal=self.temporal,
            segment=self.segment,
            speaker=self.speaker,
            deictic=self.deictic,
            audit=AuditInfo(
                tokens=[t.to_tuple() for t in self.tokens],
                decisions=self.decisions,
                warnings=[]
            )
        )
```

---

## 5. Resolver

### 5.1 Inputs

- `ast: MQLCommand or FreeText`
- `now_utc: datetime` — injected by caller for determinism in tests
- `user_tz: str` — from config (default `America/Chicago`)
- `conn: sqlite3.Connection` — for segment lookup

### 5.2 Temporal Resolution

Produces UTC half-open `[start, end)`:
- Interpret relative phrases in user timezone.
- Snap boundaries to local midnight or appropriate unit boundary.
- Convert to UTC using zoneinfo.

```python
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta


def resolve_temporal(
    t: Temporal,
    now_utc: datetime,
    user_tz: str
) -> Optional[Tuple[datetime, datetime]]:
    """
    Convert Temporal to UTC half-open interval [start, end).
    Returns None if resolution fails (logged to audit).
    """
    try:
        tz = ZoneInfo(user_tz)
        utc = ZoneInfo("UTC")
        local_now = now_utc.astimezone(tz)
        
        def midnight(dt: datetime) -> datetime:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        today = midnight(local_now)
        
        if t.kind == "yesterday":
            start = today - timedelta(days=1)
            end = today
        
        elif t.kind == "today":
            start = today
            end = today + timedelta(days=1)
        
        elif t.kind == "last_week":
            # ISO week: Monday to Sunday
            days_to_monday = local_now.weekday()
            this_monday = today - timedelta(days=days_to_monday)
            start = this_monday - timedelta(days=7)
            end = this_monday
        
        elif t.kind == "this_week":
            days_to_monday = local_now.weekday()
            start = today - timedelta(days=days_to_monday)
            end = start + timedelta(days=7)
        
        elif t.kind == "last_month":
            first_of_month = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if local_now.month == 1:
                start = first_of_month.replace(year=local_now.year - 1, month=12)
            else:
                start = first_of_month.replace(month=local_now.month - 1)
            end = first_of_month
        
        elif t.kind == "this_month":
            start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if local_now.month == 12:
                end = start.replace(year=local_now.year + 1, month=1)
            else:
                end = start.replace(month=local_now.month + 1)
        
        elif t.kind == "last_n_days":
            start = today - timedelta(days=t.n)
            end = today + timedelta(days=1)
        
        elif t.kind == "explicit_date":
            from dateutil import parser as dateparser
            parsed = dateparser.parse(t.date1, default=local_now.replace(tzinfo=None))
            start = tz.localize(midnight(parsed))
            end = start + timedelta(days=1)
        
        elif t.kind == "date_range":
            from dateutil import parser as dateparser
            d1 = dateparser.parse(t.date1, default=local_now.replace(tzinfo=None))
            d2 = dateparser.parse(t.date2, default=local_now.replace(tzinfo=None))
            start = tz.localize(midnight(d1))
            end = tz.localize(midnight(d2)) + timedelta(days=1)
        
        else:
            return None
        
        return (start.astimezone(utc), end.astimezone(utc))
    
    except Exception:
        return None  # Drop modifier, log in audit
```

### 5.3 Segment Resolution Gate (Explicit Only)

The resolver only attempts segment resolution when the AST has an **explicit** `segment_mod`.

- **No explicit segment_mod:** `segment_explicit=False`, `segment_ids=None` (not requested).
- **Explicit but no match:** `segment_explicit=True`, `segment_ids=[]` (requested-but-empty).

```python
def resolve_segment(
    conn: sqlite3.Connection,
    phrase: str
) -> Tuple[str, List[str]]:
    """
    Resolve segment phrase to node IDs.
    Returns (normalized_query, node_ids).
    """
    # Normalize phrase: lowercase, replace - and _ with space
    norm_phrase = phrase.lower().replace('-', ' ').replace('_', ' ').strip()
    
    topics = get_all_topics(conn)
    
    # Exact match
    for topic in topics:
        norm_name = topic['name'].lower().replace('-', ' ').replace('_', ' ')
        if norm_name == norm_phrase:
            nodes, _ = get_cached_segment_nodes(conn, topic['id'])
            return (norm_phrase, nodes)
    
    # Contains match
    for topic in topics:
        if norm_phrase in topic['name'].lower():
            nodes, _ = get_cached_segment_nodes(conn, topic['id'])
            return (norm_phrase, nodes)
    
    # No match
    return (norm_phrase, [])
```

### 5.4 Full Resolver

```python
class Resolver:
    def __init__(
        self,
        conn: sqlite3.Connection,
        now_utc: datetime,
        user_tz: str = "America/Chicago"
    ):
        self.conn = conn
        self.now_utc = now_utc
        self.user_tz = user_tz
        self.audit_entries: List[str] = []
    
    def resolve(self, ast: Union[MQLCommand, FreeText]) -> ResolvedQuery:
        if isinstance(ast, FreeText):
            return ResolvedQuery(
                mode="answer",
                target=ast.text,
                temporal=None,
                segment_explicit=False,
                segment_query=None,
                segment_ids=None,
                speaker=None,
                deictic_limit=None,
                audit_trace=json.dumps(ast.to_dict(), sort_keys=True)
            )
        
        # Target
        target = ast.target.text if ast.target else ""
        
        # Temporal
        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)
            if temporal is None:
                self.audit_entries.append(f"temporal resolution failed for {ast.temporal.raw}")
        
        # Segment (explicit gate)
        segment_explicit = ast.segment is not None
        segment_query = None
        segment_ids = None
        
        if segment_explicit:
            segment_query, segment_ids = resolve_segment(self.conn, ast.segment.phrase)
            self.audit_entries.append(f"segment lookup: query={segment_query}, found={len(segment_ids)} nodes")
        
        # Speaker
        speaker = ast.speaker.role if ast.speaker else None
        
        # Deictic
        deictic = ast.deictic.count if ast.deictic else None
        
        # Build audit trace
        audit_dict = {
            "ast": ast.to_dict(),
            "resolver_entries": self.audit_entries,
        }
        
        return ResolvedQuery(
            mode=ast.mode,
            target=target,
            temporal=temporal,
            segment_explicit=segment_explicit,
            segment_query=segment_query,
            segment_ids=segment_ids,
            speaker=speaker,
            deictic_limit=deictic,
            audit_trace=json.dumps(audit_dict, sort_keys=True, default=str)
        )
```

---

## 6. Retrieval Routing Contract

| Field | Value | Retrieval Behavior |
|-------|-------|-------------------|
| `segment_explicit=False` | `segment_ids=None` | Search all nodes |
| `segment_explicit=True` | `segment_ids=[]` | Return empty (requested, not found) |
| `segment_explicit=True` | `segment_ids=[...]` | Search only these nodes |
| `speaker=None` | | Lexical + semantic retrieval |
| `speaker="user"/"assistant"` | | Lexical only (semantic disabled) |
| `mode="browse"`, empty target | | Return recent exchanges |
| `mode="answer"/"summarize"`, empty target, no deictic | | Return empty (strict) |

**Invariant:** Speaker scope disables semantic retrieval because embeddings are exchange-level.

---

## 7. Unit Test Plan (Criterion-Aligned)

### Criterion 1: Deterministic token stream and spans
**Unit tests:**
- Identical input yields identical tokens (including spans)
- Span invariant: `lexeme == input[start:end]`

**Golden fixtures:**
- Suite with punctuation, quotes, unicode whitespace

### Criterion 2: No accidental segment scope
**Unit tests:**
- `"in coffee"` does NOT set `segment_explicit=True`
- `"when we discussed coffee"` does NOT set `segment_explicit=True`
- `"topic: coffee"` DOES set `segment_explicit=True`

**Golden fixtures:**
- All three cases above

### Criterion 3: Soft keywords are safely reinterpretable
**Unit tests:**
- Input where a keyword would break parsing is treated as WORD
- Reinterpretation is recorded in audit
- Reinterpretation does NOT create a modifier

**Golden fixtures:**
- `"browse topic: topic"` where second "topic" is target text

### Criterion 4: Fail-closed on parse failure
**Unit tests:**
- Syntactically invalid constructs return FreeText
- FreeText preserves original input

**Golden fixtures:**
- Unbalanced quotes, random punctuation, partial modifier sequences

### Criterion 5: Temporal resolver is DST-safe and half-open
**Unit tests:**
- `"yesterday"` resolves to `[local midnight, next local midnight)` in UTC
- Boundary test: timestamp exactly at end boundary is excluded

**Golden fixtures:**
- Winter date
- DST transition date (spring forward, fall back)

### Criterion 6: Speaker modifier mapping correctness
**Unit tests:**
- `"did I mention X"` → `speaker=user`
- `"did you say X"` → `speaker=assistant`

**Golden fixtures:**
- Both cases

### Criterion 7: Segment explicit gate behavior
**Unit tests:**
- Segment resolution invoked only when `segment_explicit=True`
- Explicit-but-empty returns `segment_ids=[]`, NOT `None`

**Golden fixtures:**
- `"topic: does-not-exist"` → `segment_explicit=True`, `segment_ids=[]`

### Criterion 8: Audit trace stability
**Unit tests:**
- Audit serialization is deterministic (stable key ordering)
- Soft keyword reinterpretations are recorded

**Golden fixtures:**
- Verify presence of stable substrings

---

## 8. Golden Fixtures

### 8.1 Fixture Format

```yaml
- name: <test_name>
  input: "<query string>"
  now_utc: "2026-01-25T12:00:00Z"
  user_tz: "America/Chicago"
  expect_tokens: [["TYPE", "lexeme", start, end], ...]
  expect_ast:
    kind: MQLCommand | FreeText
    mode: browse | answer | summarize
    target: "<string>"
    temporal: { kind: ..., raw: ... }
    segment: { phrase: ... }
    speaker: { role: ... }
    deictic: { count: ... }
  expect_resolved:
    mode: ...
    target: ...
    temporal_utc: ["<start>", "<end>"]
    segment_explicit: true | false
    segment_query: ...
    segment_ids: null | [] | [...]
    speaker: null | "user" | "assistant"
    deictic_limit: null | <int>
  expect_audit_contains: ["substring1", "substring2"]
```

### 8.2 Core Fixtures

```yaml
# Mode starters
- name: explicit_browse_prefix
  input: "Browse: coffee"
  now_utc: "2026-01-25T12:00:00Z"
  expect_ast:
    kind: MQLCommand
    mode: browse
    target: "coffee"
  expect_resolved:
    mode: browse
    target: "coffee"
    segment_explicit: false

- name: explicit_summarize_prefix
  input: "Summarize: retrieval bugs"
  expect_ast:
    kind: MQLCommand
    mode: summarize
    target: "retrieval bugs"

# Discussion queries (NO segment scope)
- name: when_we_discussed_no_segment
  input: "when we discussed coffee yesterday?"
  now_utc: "2026-01-25T12:00:00Z"
  expect_ast:
    kind: MQLCommand
    mode: browse
    target: "coffee"
    temporal: { kind: yesterday }
    segment: null
  expect_resolved:
    mode: browse
    target: "coffee"
    temporal_utc: ["2026-01-24T06:00:00Z", "2026-01-25T06:00:00Z"]
    segment_explicit: false
    segment_ids: null

# Explicit segment scope
- name: explicit_topic_colon
  input: "topic: hiring pipeline"
  expect_ast:
    kind: MQLCommand
    segment: { phrase: "hiring pipeline" }
  expect_resolved:
    segment_explicit: true
    segment_query: "hiring pipeline"

- name: explicit_in_topic
  input: "in topic coffee-brewing what was the ratio"
  expect_ast:
    segment: { phrase: "coffee-brewing" }
  expect_resolved:
    segment_explicit: true

# Bare "in" does NOT trigger segment
- name: in_the_morning_no_segment
  input: "in the morning we discussed coffee"
  expect_ast:
    kind: MQLCommand
    segment: null
    target: "morning discussed coffee"
  expect_resolved:
    segment_explicit: false
    segment_ids: null

# Deictic vs temporal disambiguation
- name: deictic_last_10_messages
  input: "last 10 messages"
  expect_ast:
    kind: MQLCommand
    mode: browse
    deictic: { count: 10 }
    temporal: null
  expect_resolved:
    deictic_limit: 10
    temporal_utc: null

- name: temporal_last_5_days
  input: "last 5 days coffee"
  expect_ast:
    temporal: { kind: last_n_days, n: 5 }
    deictic: null
  expect_resolved:
    deictic_limit: null

# Speaker scope
- name: did_i_ever_ask
  input: "did I ever ask about BM25"
  expect_ast:
    mode: browse
    speaker: { role: user }
    target: "bm25"
  expect_resolved:
    speaker: "user"

- name: did_you_say
  input: "did you say 'FTS5 rebuild'?"
  expect_ast:
    mode: browse
    speaker: { role: assistant }
    target: "fts5 rebuild"
  expect_resolved:
    speaker: "assistant"

# Soft keyword reinterpretation
- name: topic_as_target
  input: "browse topic: topic"
  expect_ast:
    mode: browse
    segment: { phrase: "topic" }
  expect_audit_contains: ["reinterpreted KW_TOPIC as WORD"]

# Edit stability
- name: case_insensitive
  input: "BROWSE: Coffee"
  expect_resolved:
    mode: browse
    target: "coffee"

- name: extra_whitespace
  input: "browse:   coffee   yesterday"
  expect_resolved:
    mode: browse
    target: "coffee"

- name: trailing_punctuation
  input: "Browse: coffee?"
  expect_resolved:
    target: "coffee"

# Quoted phrases
- name: quoted_target
  input: 'Browse: "cold exposure" yesterday'
  expect_ast:
    target: "cold exposure"

# Fail-closed
- name: freetext_garbage
  input: "ok show me the thing from before"
  expect_ast:
    kind: MQLCommand
    mode: browse
    target: "thing from before"

- name: unclosed_quote
  input: "browse: \"unclosed"
  expect_ast:
    kind: MQLCommand
  expect_audit_contains: ["unclosed quote"]

# DST boundaries
- name: dst_spring_forward
  input: "yesterday"
  now_utc: "2024-03-10T15:00:00Z"
  user_tz: "America/Chicago"
  expect_resolved:
    temporal_utc: ["2024-03-09T06:00:00Z", "2024-03-10T05:00:00Z"]

- name: dst_fall_back
  input: "yesterday"
  now_utc: "2024-11-03T15:00:00Z"
  user_tz: "America/Chicago"
  expect_resolved:
    temporal_utc: ["2024-11-02T05:00:00Z", "2024-11-03T06:00:00Z"]
```

---

## 9. Examples (User-Level Commands)

| Input | mode | target | temporal | segment_explicit | speaker |
|-------|------|--------|----------|------------------|---------|
| `when we discussed coffee yesterday` | browse | coffee | yesterday | false | - |
| `browse topic: hiring pipeline` | browse | - | - | true | - |
| `summarize last week: episodic retrieval bugs` | summarize | episodic retrieval bugs | last_week | false | - |
| `did you say 'FTS5 rebuild'?` | browse | FTS5 rebuild | - | false | assistant |
| `last 10 messages` | browse | - | - | false | - |
| `in topic coffee-brewing what was the ratio` | answer | what was the ratio | - | true | - |

---

## 10. Integration Notes

1. **This spec defines only query understanding output fields** and the semantics that prevent unintended widening of scope.
2. **Retrieval pipeline must enforce its own policies:**
   - Empty-result policies (no hallucination)
   - Treat `FreeText` as "target only, no implicit scopes"
   - Disable semantic retrieval when `speaker` is set
3. **FreeText contract:** Retrieval treats as unscoped lexical search on the original text.
