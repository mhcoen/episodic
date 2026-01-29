# Query Understanding Specification v1.4
## Tokenizer + Parser with Soft Keywords

---

## 1. Design Goals

1. **Deterministic** — Same input + same `now_utc` → identical token stream, AST, ResolvedQuery
2. **No accidental scopes** — "in" doesn't trigger segment scope without explicit "topic/segment"
3. **Soft keywords** — Parser reinterprets keywords as WORD when production fails
4. **Fail-closed** — Invalid Stage B → FreeText, not broadened search
5. **Audit-stable** — Logging uses deterministic serialization

---

## 2. Token Specification

### 2.1 Token Types

```python
class TokenType(Enum):
    # Literals (never reinterpreted)
    NUMBER = auto()       # 10, 5, etc.
    QUOTED = auto()       # "phrase" or 'phrase'
    COLON = auto()        # :
    QUESTION = auto()     # ?
    EOF = auto()
    
    # Soft keywords (parser may reinterpret as WORD)
    # Mode starters
    KW_BROWSE = auto()    # browse, show, list, display
    KW_ANSWER = auto()    # answer
    KW_SUMMARIZE = auto() # summarize, summary
    KW_WHAT = auto()      # what
    KW_WHEN = auto()      # when
    KW_WHERE = auto()     # where
    KW_DID = auto()       # did
    
    # Scope
    KW_TOPIC = auto()     # topic, segment
    KW_IN = auto()        # in, within
    KW_ABOUT = auto()     # about
    
    # Speaker
    KW_I = auto()         # i, me, my
    KW_YOU = auto()       # you, your
    KW_WE = auto()        # we, our
    KW_ONLY = auto()      # only
    
    # Temporal
    KW_YESTERDAY = auto()
    KW_TODAY = auto()
    KW_LAST = auto()      # last
    KW_THIS = auto()      # this
    KW_ON = auto()        # on
    KW_BETWEEN = auto()   # between
    KW_AND = auto()       # and
    
    # Units
    KW_WEEK = auto()
    KW_MONTH = auto()
    KW_DAY = auto()
    KW_DAYS = auto()
    KW_MESSAGES = auto()  # messages, exchanges, message, exchange
    
    # Verbs
    KW_SAID = auto()      # said, say, asked, mentioned, recommended
    KW_DISCUSSED = auto() # discussed, talked
    KW_EVER = auto()      # ever
    KW_DECIDE = auto()    # decide, decided, conclude, concluded
    
    # Month names (for date parsing)
    MONTH_NAME = auto()
    
    # Default: any unrecognized word
    WORD = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str           # Original text
    start: int            # Start position
    end: int              # End position
    normalized: str       # Lowercase
    
    def as_word(self) -> 'Token':
        """Reinterpret this token as WORD (for soft keyword handling)."""
        return Token(TokenType.WORD, self.lexeme, self.start, self.end, self.normalized)
    
    def to_tuple(self) -> Tuple[str, str, int, int]:
        """Stable serialization for audit logging."""
        return (self.type.name, self.lexeme, self.start, self.end)
```

### 2.2 Keyword Map

```python
KEYWORD_MAP: Dict[str, TokenType] = {
    # Mode
    "browse": TokenType.KW_BROWSE,
    "show": TokenType.KW_BROWSE,
    "list": TokenType.KW_BROWSE,
    "display": TokenType.KW_BROWSE,
    "answer": TokenType.KW_ANSWER,
    "summarize": TokenType.KW_SUMMARIZE,
    "summary": TokenType.KW_SUMMARIZE,
    "what": TokenType.KW_WHAT,
    "when": TokenType.KW_WHEN,
    "where": TokenType.KW_WHERE,
    "did": TokenType.KW_DID,
    
    # Scope
    "topic": TokenType.KW_TOPIC,
    "segment": TokenType.KW_TOPIC,
    "in": TokenType.KW_IN,
    "within": TokenType.KW_IN,
    "about": TokenType.KW_ABOUT,
    
    # Speaker
    "i": TokenType.KW_I,
    "me": TokenType.KW_I,
    "my": TokenType.KW_I,
    "you": TokenType.KW_YOU,
    "your": TokenType.KW_YOU,
    "we": TokenType.KW_WE,
    "our": TokenType.KW_WE,
    "only": TokenType.KW_ONLY,
    
    # Temporal
    "yesterday": TokenType.KW_YESTERDAY,
    "today": TokenType.KW_TODAY,
    "last": TokenType.KW_LAST,
    "this": TokenType.KW_THIS,
    "on": TokenType.KW_ON,
    "between": TokenType.KW_BETWEEN,
    "and": TokenType.KW_AND,
    
    # Units
    "week": TokenType.KW_WEEK,
    "month": TokenType.KW_MONTH,
    "day": TokenType.KW_DAY,
    "days": TokenType.KW_DAYS,
    "messages": TokenType.KW_MESSAGES,
    "exchanges": TokenType.KW_MESSAGES,
    "message": TokenType.KW_MESSAGES,
    "exchange": TokenType.KW_MESSAGES,
    
    # Verbs
    "said": TokenType.KW_SAID,
    "say": TokenType.KW_SAID,
    "asked": TokenType.KW_SAID,
    "ask": TokenType.KW_SAID,
    "mentioned": TokenType.KW_SAID,
    "mention": TokenType.KW_SAID,
    "recommended": TokenType.KW_SAID,
    "recommend": TokenType.KW_SAID,
    "discussed": TokenType.KW_DISCUSSED,
    "discuss": TokenType.KW_DISCUSSED,
    "talked": TokenType.KW_DISCUSSED,
    "talk": TokenType.KW_DISCUSSED,
    "ever": TokenType.KW_EVER,
    "decide": TokenType.KW_DECIDE,
    "decided": TokenType.KW_DECIDE,
    "conclude": TokenType.KW_DECIDE,
    "concluded": TokenType.KW_DECIDE,
}

MONTH_NAMES = frozenset({
    "jan", "january", "feb", "february", "mar", "march",
    "apr", "april", "may", "jun", "june", "jul", "july",
    "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december"
})
```

### 2.3 Lexer

```python
class Lexer:
    """
    Tokenizes input into a stream of tokens.
    Keywords are emitted as KW_* types but parser may reinterpret as WORD.
    """
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            self._scan_token()
        
        self.tokens.append(Token(TokenType.EOF, "", len(self.text), len(self.text), ""))
        return self.tokens
    
    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
    
    def _scan_token(self):
        start = self.pos
        char = self.text[self.pos]
        
        if char == ':':
            self._emit(TokenType.COLON, start, start + 1)
            self.pos += 1
        elif char == '?':
            self._emit(TokenType.QUESTION, start, start + 1)
            self.pos += 1
        elif char in ('"', "'"):
            self._scan_quoted(char)
        elif char.isdigit():
            self._scan_number()
        else:
            self._scan_word()
    
    def _emit(self, token_type: TokenType, start: int, end: int):
        lexeme = self.text[start:end]
        self.tokens.append(Token(token_type, lexeme, start, end, lexeme.lower()))
    
    def _scan_quoted(self, quote_char: str):
        start = self.pos
        self.pos += 1
        while self.pos < len(self.text) and self.text[self.pos] != quote_char:
            self.pos += 1
        if self.pos < len(self.text):
            self.pos += 1
        lexeme = self.text[start:self.pos]
        inner = lexeme[1:-1] if len(lexeme) >= 2 else ""
        self.tokens.append(Token(TokenType.QUOTED, lexeme, start, self.pos, inner.lower()))
    
    def _scan_number(self):
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        lexeme = self.text[start:self.pos]
        self.tokens.append(Token(TokenType.NUMBER, lexeme, start, self.pos, lexeme))
    
    def _scan_word(self):
        start = self.pos
        while self.pos < len(self.text) and self._is_word_char(self.text[self.pos]):
            self.pos += 1
        lexeme = self.text[start:self.pos]
        normalized = lexeme.lower()
        
        # Classify token type
        if normalized in MONTH_NAMES:
            token_type = TokenType.MONTH_NAME
        elif normalized in KEYWORD_MAP:
            token_type = KEYWORD_MAP[normalized]
        else:
            token_type = TokenType.WORD
        
        self.tokens.append(Token(token_type, lexeme, start, self.pos, normalized))
    
    def _is_word_char(self, c: str) -> bool:
        return c.isalnum() or c in ('-', '_')
```

---

## 3. AST Definition

```python
@dataclass(frozen=True)
class Query:
    """Search terms extracted from input."""
    terms: Tuple[str, ...]
    spans: Tuple[Tuple[int, int], ...]
    
    @property
    def text(self) -> str:
        return " ".join(self.terms)
    
    def to_dict(self) -> dict:
        return {"terms": list(self.terms), "spans": list(self.spans)}


@dataclass(frozen=True)
class Temporal:
    """Temporal scope specification."""
    kind: str  # yesterday, today, last_week, this_week, last_month, this_month, last_n_days, explicit_date, date_range
    raw: str
    n: Optional[int] = None
    date1: Optional[str] = None
    date2: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "kind": self.kind, "raw": self.raw, "n": self.n,
            "date1": self.date1, "date2": self.date2
        }.items() if v is not None}


@dataclass(frozen=True)
class Segment:
    """Segment/topic scope. Only created for explicit scope markers."""
    phrase: str
    
    def to_dict(self) -> dict:
        return {"phrase": self.phrase}


@dataclass(frozen=True)
class Speaker:
    """Speaker scope."""
    role: str  # "user" or "assistant"
    
    def to_dict(self) -> dict:
        return {"role": self.role}


@dataclass(frozen=True)
class Deictic:
    """Deictic limit (last N messages/exchanges)."""
    count: int
    
    def to_dict(self) -> dict:
        return {"count": self.count}


@dataclass(frozen=True)
class MQLCommand:
    """Successfully parsed memory query."""
    mode: str  # "browse", "answer", "summarize"
    query: Optional[Query]
    temporal: Optional[Temporal]
    segment: Optional[Segment]
    speaker: Optional[Speaker]
    deictic: Optional[Deictic]
    
    def to_dict(self) -> dict:
        d = {"type": "MQLCommand", "mode": self.mode}
        if self.query:
            d["query"] = self.query.to_dict()
        if self.temporal:
            d["temporal"] = self.temporal.to_dict()
        if self.segment:
            d["segment"] = self.segment.to_dict()
        if self.speaker:
            d["speaker"] = self.speaker.to_dict()
        if self.deictic:
            d["deictic"] = self.deictic.to_dict()
        return d


@dataclass(frozen=True)
class FreeText:
    """Unparseable input, routes to Stage B."""
    text: str
    reason: str
    
    def to_dict(self) -> dict:
        return {"type": "FreeText", "text": self.text, "reason": self.reason}
```

---

## 4. Parser

### 4.1 Grammar (Precedence Rules)

```
command ::= mode_phrase? modifiers* query_remainder

# MODE PRECEDENCE (strict, checked in order):
# 1. Explicit prefix: "browse:" | "answer:" | "summarize:"
# 2. "what did" → answer
# 3. "when did" | "where did" | "did I/you ever" → browse  
# 4. "show" | "list" | "display" → browse
# 5. Default → answer

mode_phrase ::= explicit_prefix
              | what_did_phrase
              | when_where_did_phrase
              | did_i_you_phrase
              | show_list_phrase

explicit_prefix ::= ("browse" | "answer" | "summarize") ":"

what_did_phrase ::= "what" "did" ("we" | "you" | "I") (say_verb | decide_verb) "about"?
                  # Sets mode=answer, optionally sets speaker

when_where_did_phrase ::= ("when" | "where") "did"? ("we" | "I" | "you")? discuss_verb? "about"?
                        # Sets mode=browse

did_i_you_phrase ::= "did" ("I" | "you") "ever"? say_verb "about"?
                   # Sets mode=browse, sets speaker

show_list_phrase ::= ("show" "me"? | "list" | "display")
                   # Sets mode=browse

# MODIFIERS (can appear in any order, after mode phrase)
modifiers ::= segment_mod | temporal_mod | speaker_mod | deictic_mod

# SEGMENT: REQUIRES EXPLICIT "topic" OR "segment" KEYWORD
segment_mod ::= ("topic" | "segment") ":" phrase
              | "in" ("topic" | "segment") phrase
              # NOTE: bare "in <phrase>" does NOT trigger segment scope

# TEMPORAL: LAST disambiguation
temporal_mod ::= "yesterday"
               | "today"
               | "last" "week"
               | "last" "month"
               | "this" "week"
               | "this" "month"
               | "last" NUMBER "days"?    # Only if NOT followed by "messages/exchanges"
               | "on" date
               | "between" date "and" date

# DEICTIC: Takes precedence over temporal for "last N messages/exchanges"
deictic_mod ::= "last" NUMBER ("messages" | "exchanges")

# SPEAKER: Requires speech verb
speaker_mod ::= "only"? "what" ("I" | "you") said_verb
              | ("my" | "your") ("messages" | "responses")

query_remainder ::= (QUOTED | WORD | keyword_as_word)*
                  # Keywords not consumed by mode/modifiers become query terms
```

### 4.2 Implementation

```python
class Parser:
    """
    Recursive descent parser with soft keyword handling.
    Keywords can be reinterpreted as WORD when they don't fit a production.
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.original_text = self._reconstruct_text()
        
        # Parse state
        self.mode: Optional[str] = None
        self.query_tokens: List[Token] = []
        self.temporal: Optional[Temporal] = None
        self.segment: Optional[Segment] = None
        self.speaker: Optional[Speaker] = None
        self.deictic: Optional[Deictic] = None
    
    def parse(self) -> Union[MQLCommand, FreeText]:
        try:
            self._parse_mode_phrase()
            self._parse_modifiers()
            self._collect_query_remainder()
            return self._build_ast()
        except ParseError as e:
            return FreeText(text=self.original_text, reason=str(e))
    
    # ─────────────────────────────────────────────────────────────
    # Token access with soft keyword support
    # ─────────────────────────────────────────────────────────────
    
    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]
    
    def _at(self, *types: TokenType) -> bool:
        return self._peek().type in types
    
    def _at_normalized(self, *values: str) -> bool:
        return self._peek().normalized in values
    
    def _match(self, *types: TokenType) -> Optional[Token]:
        if self._at(*types):
            tok = self._peek()
            self.pos += 1
            return tok
        return None
    
    def _at_end(self) -> bool:
        return self._at(TokenType.EOF)
    
    def _save(self) -> int:
        return self.pos
    
    def _restore(self, pos: int):
        self.pos = pos
    
    def _reconstruct_text(self) -> str:
        if not self.tokens:
            return ""
        # Find bounds from first to last non-EOF token
        start = self.tokens[0].start
        end = self.tokens[-2].end if len(self.tokens) > 1 else start
        # We don't have original text, approximate from lexemes
        return " ".join(t.lexeme for t in self.tokens if t.type != TokenType.EOF)
    
    # ─────────────────────────────────────────────────────────────
    # Mode phrase parsing (strict precedence)
    # ─────────────────────────────────────────────────────────────
    
    def _parse_mode_phrase(self):
        """Parse mode phrase with strict precedence."""
        
        # 1. Explicit prefix: "browse:" | "answer:" | "summarize:"
        if self._try_explicit_prefix():
            return
        
        # 2. "what did we/you/I say/decide about" → answer
        if self._try_what_did_phrase():
            return
        
        # 3. "when/where did we discuss" → browse
        if self._try_when_where_phrase():
            return
        
        # 4. "did I/you ever say" → browse
        if self._try_did_i_you_phrase():
            return
        
        # 5. "show/list/display" → browse
        if self._try_show_list_phrase():
            return
        
        # 6. Default: answer
        self.mode = "answer"
    
    def _try_explicit_prefix(self) -> bool:
        """Match: (browse|answer|summarize) ":"."""
        saved = self._save()
        
        mode_tok = self._match(TokenType.KW_BROWSE, TokenType.KW_ANSWER, TokenType.KW_SUMMARIZE)
        if mode_tok and self._match(TokenType.COLON):
            if mode_tok.type == TokenType.KW_BROWSE:
                self.mode = "browse"
            elif mode_tok.type == TokenType.KW_ANSWER:
                self.mode = "answer"
            else:
                self.mode = "summarize"
            return True
        
        self._restore(saved)
        return False
    
    def _try_what_did_phrase(self) -> bool:
        """Match: what did (we|you|I) (say_verb|decide_verb) about?"""
        saved = self._save()
        
        if not self._match(TokenType.KW_WHAT):
            return False
        
        if not self._match(TokenType.KW_DID):
            self._restore(saved)
            return False
        
        subject = self._match(TokenType.KW_WE, TokenType.KW_YOU, TokenType.KW_I)
        if not subject:
            self._restore(saved)
            return False
        
        # Optional verb
        self._match(TokenType.KW_SAID, TokenType.KW_DISCUSSED, TokenType.KW_DECIDE)
        
        # Optional "about"
        self._match(TokenType.KW_ABOUT)
        
        self.mode = "answer"
        if subject.type == TokenType.KW_YOU:
            self.speaker = Speaker(role="assistant")
        elif subject.type == TokenType.KW_I:
            self.speaker = Speaker(role="user")
        
        return True
    
    def _try_when_where_phrase(self) -> bool:
        """Match: (when|where) did? (we|I|you)? discuss_verb? about?"""
        saved = self._save()
        
        if not self._match(TokenType.KW_WHEN, TokenType.KW_WHERE):
            return False
        
        self._match(TokenType.KW_DID)  # Optional
        self._match(TokenType.KW_WE, TokenType.KW_I, TokenType.KW_YOU)  # Optional
        self._match(TokenType.KW_DISCUSSED, TokenType.KW_SAID)  # Optional
        self._match(TokenType.KW_ABOUT)  # Optional
        
        self.mode = "browse"
        return True
    
    def _try_did_i_you_phrase(self) -> bool:
        """Match: did (I|you) ever? say_verb about?"""
        saved = self._save()
        
        if not self._match(TokenType.KW_DID):
            return False
        
        subject = self._match(TokenType.KW_I, TokenType.KW_YOU)
        if not subject:
            self._restore(saved)
            return False
        
        self._match(TokenType.KW_EVER)  # Optional
        self._match(TokenType.KW_SAID, TokenType.KW_DISCUSSED)  # Optional
        self._match(TokenType.KW_ABOUT)  # Optional
        
        self.mode = "browse"
        self.speaker = Speaker(role="user" if subject.type == TokenType.KW_I else "assistant")
        return True
    
    def _try_show_list_phrase(self) -> bool:
        """Match: (show me?|list|display)."""
        saved = self._save()
        
        if self._match(TokenType.KW_BROWSE):
            # "show me"
            if self._peek().normalized == "me":
                self.pos += 1
            self.mode = "browse"
            return True
        
        self._restore(saved)
        return False
    
    # ─────────────────────────────────────────────────────────────
    # Modifier parsing
    # ─────────────────────────────────────────────────────────────
    
    def _parse_modifiers(self):
        """Parse modifiers until none match."""
        while not self._at_end():
            if self._try_deictic_mod():  # Check deictic BEFORE temporal (LAST disambiguation)
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
        if self._match(TokenType.KW_TOPIC):
            if self._match(TokenType.COLON):
                phrase = self._parse_phrase()
                if phrase:
                    self.segment = Segment(phrase=phrase)
                    return True
            self._restore(saved)
            return False
        
        # "in topic X" or "in segment X"
        if self._match(TokenType.KW_IN):
            if self._match(TokenType.KW_TOPIC):
                phrase = self._parse_phrase()
                if phrase:
                    self.segment = Segment(phrase=phrase)
                    return True
            # "in" WITHOUT topic/segment → NOT segment scope, restore
            self._restore(saved)
            return False
        
        return False
    
    def _try_temporal_mod(self) -> bool:
        """Parse temporal modifier (excludes deictic "last N messages")."""
        saved = self._save()
        
        # "yesterday"
        if self._match(TokenType.KW_YESTERDAY):
            self.temporal = Temporal(kind="yesterday", raw="yesterday")
            return True
        
        # "today"
        if self._match(TokenType.KW_TODAY):
            self.temporal = Temporal(kind="today", raw="today")
            return True
        
        # "last week/month" or "last N days" (but NOT "last N messages")
        if self._match(TokenType.KW_LAST):
            if self._match(TokenType.KW_WEEK):
                self.temporal = Temporal(kind="last_week", raw="last week")
                return True
            
            if self._match(TokenType.KW_MONTH):
                self.temporal = Temporal(kind="last_month", raw="last month")
                return True
            
            # "last N days" - check it's NOT "last N messages"
            num_tok = self._match(TokenType.NUMBER)
            if num_tok:
                # Lookahead: if followed by MESSAGES, this is deictic, not temporal
                if self._at(TokenType.KW_MESSAGES):
                    # Put back NUMBER and LAST
                    self._restore(saved)
                    return False
                
                # "last N days" or "last N day"
                if self._match(TokenType.KW_DAYS, TokenType.KW_DAY):
                    n = int(num_tok.normalized)
                    self.temporal = Temporal(kind="last_n_days", raw=f"last {n} days", n=n)
                    return True
                
                # "last N" without unit - treat LAST as query word
                self._restore(saved)
                return False
            
            # Just "last" with no unit - restore
            self._restore(saved)
            return False
        
        # "this week/month"
        if self._match(TokenType.KW_THIS):
            if self._match(TokenType.KW_WEEK):
                self.temporal = Temporal(kind="this_week", raw="this week")
                return True
            if self._match(TokenType.KW_MONTH):
                self.temporal = Temporal(kind="this_month", raw="this month")
                return True
            self._restore(saved)
            return False
        
        # "on DATE"
        if self._match(TokenType.KW_ON):
            date_str = self._try_parse_date()
            if date_str:
                self.temporal = Temporal(kind="explicit_date", raw=f"on {date_str}", date1=date_str)
                return True
            self._restore(saved)
            return False
        
        # "between DATE and DATE"
        if self._match(TokenType.KW_BETWEEN):
            date1 = self._try_parse_date()
            if date1 and self._match(TokenType.KW_AND):
                date2 = self._try_parse_date()
                if date2:
                    self.temporal = Temporal(
                        kind="date_range",
                        raw=f"between {date1} and {date2}",
                        date1=date1, date2=date2
                    )
                    return True
            self._restore(saved)
            return False
        
        return False
    
    def _try_deictic_mod(self) -> bool:
        """Match: last NUMBER (messages|exchanges)."""
        saved = self._save()
        
        if not self._match(TokenType.KW_LAST):
            return False
        
        num_tok = self._match(TokenType.NUMBER)
        if not num_tok:
            self._restore(saved)
            return False
        
        if not self._match(TokenType.KW_MESSAGES):
            self._restore(saved)
            return False
        
        self.deictic = Deictic(count=int(num_tok.normalized))
        return True
    
    def _try_speaker_mod(self) -> bool:
        """
        Match: "only"? "what" (I|you) said_verb
             | (my|your) (messages|responses)
        """
        saved = self._save()
        
        # "only what I/you said"
        self._match(TokenType.KW_ONLY)  # Optional
        
        if self._at_normalized("what"):
            self.pos += 1
            subject = self._match(TokenType.KW_I, TokenType.KW_YOU)
            if subject and self._match(TokenType.KW_SAID):
                self.speaker = Speaker(role="user" if subject.type == TokenType.KW_I else "assistant")
                return True
            self._restore(saved)
            return False
        
        # "my messages" / "your messages"
        if self._at(TokenType.KW_I) and self._peek().normalized == "my":
            self.pos += 1
            if self._at(TokenType.KW_MESSAGES) or self._at_normalized("responses"):
                self.pos += 1
                self.speaker = Speaker(role="user")
                return True
            self._restore(saved)
            return False
        
        if self._at(TokenType.KW_YOU) and self._peek().normalized == "your":
            self.pos += 1
            if self._at(TokenType.KW_MESSAGES) or self._at_normalized("responses"):
                self.pos += 1
                self.speaker = Speaker(role="assistant")
                return True
            self._restore(saved)
            return False
        
        self._restore(saved)
        return False
    
    def _try_parse_date(self) -> Optional[str]:
        """Parse: MONTH_NAME NUMBER (NUMBER)?"""
        if not self._at(TokenType.MONTH_NAME):
            return None
        
        month = self._match(TokenType.MONTH_NAME)
        day = self._match(TokenType.NUMBER)
        if not day:
            self.pos -= 1
            return None
        
        year = self._match(TokenType.NUMBER)
        
        if year:
            return f"{month.lexeme} {day.lexeme} {year.lexeme}"
        return f"{month.lexeme} {day.lexeme}"
    
    def _parse_phrase(self) -> Optional[str]:
        """Parse phrase: QUOTED or bare words until next modifier/end."""
        if self._at(TokenType.QUOTED):
            tok = self._match(TokenType.QUOTED)
            return tok.normalized
        
        words = []
        while self._at(TokenType.WORD, TokenType.NUMBER) or self._peek().type.name.startswith("KW_"):
            tok = self._peek()
            # Stop at modifier-starting keywords
            if tok.type in (TokenType.KW_YESTERDAY, TokenType.KW_TODAY, TokenType.KW_LAST,
                           TokenType.KW_THIS, TokenType.KW_ON, TokenType.KW_BETWEEN,
                           TokenType.KW_ONLY, TokenType.KW_IN, TokenType.KW_TOPIC):
                break
            words.append(tok.normalized)
            self.pos += 1
            
            # Limit phrase length
            if len(words) >= 5:
                break
        
        return " ".join(words) if words else None
    
    # ─────────────────────────────────────────────────────────────
    # Query remainder collection
    # ─────────────────────────────────────────────────────────────
    
    def _collect_query_remainder(self):
        """Collect all remaining tokens as query terms."""
        # Skip leading "about"
        self._match(TokenType.KW_ABOUT)
        
        while not self._at_end():
            tok = self._peek()
            
            # Skip punctuation
            if tok.type in (TokenType.QUESTION, TokenType.COLON):
                self.pos += 1
                continue
            
            # Skip filler words
            if tok.normalized in ("the", "a", "an", "and", "or", "of", "for", "to", "me"):
                self.pos += 1
                continue
            
            # Collect token (keyword or not)
            if tok.type in (TokenType.WORD, TokenType.QUOTED, TokenType.NUMBER):
                self.query_tokens.append(tok)
            elif tok.type.name.startswith("KW_"):
                # Soft keyword: treat as word in query
                self.query_tokens.append(tok.as_word())
            
            self.pos += 1
    
    # ─────────────────────────────────────────────────────────────
    # AST construction
    # ─────────────────────────────────────────────────────────────
    
    def _build_ast(self) -> MQLCommand:
        query = None
        if self.query_tokens:
            terms = tuple(t.normalized for t in self.query_tokens)
            spans = tuple((t.start, t.end) for t in self.query_tokens)
            query = Query(terms=terms, spans=spans)
        
        return MQLCommand(
            mode=self.mode or "answer",
            query=query,
            temporal=self.temporal,
            segment=self.segment,
            speaker=self.speaker,
            deictic=self.deictic
        )


class ParseError(Exception):
    pass
```

---

## 5. Stage B: LLM Parser

### 5.1 Schema and Validation

```python
STAGE_B_SCHEMA = {
    "type": "object",
    "required": ["mode", "target"],
    "additionalProperties": False,
    "properties": {
        "mode": {"enum": ["browse", "answer", "summarize"]},
        "target": {"type": "string"},
        "temporal_kind": {
            "enum": [None, "yesterday", "today", "last_week", "this_week",
                    "last_month", "this_month", "last_n_days", "explicit_date", "date_range"]
        },
        "temporal_raw": {"type": ["string", "null"]},
        "temporal_n": {"type": ["integer", "null"]},
        "segment_scope_requested": {"type": "boolean"},
        "segment_phrase": {"type": ["string", "null"]},
        "speaker": {"enum": [None, "user", "assistant"]},
        "deictic_limit": {"type": ["integer", "null"]}
    }
}


def validate_stage_b_output(raw: dict) -> Tuple[Optional[MQLCommand], str]:
    """
    Validate LLM output against strict schema.
    Returns (ast, "") on success or (None, error_message) on failure.
    """
    errors = []
    
    # Check for unknown keys (fail-closed)
    allowed_keys = {"mode", "target", "temporal_kind", "temporal_raw", "temporal_n",
                   "segment_scope_requested", "segment_phrase", "speaker", "deictic_limit"}
    unknown = set(raw.keys()) - allowed_keys
    if unknown:
        errors.append(f"unknown keys: {unknown}")
    
    # Mode
    mode = raw.get("mode")
    if mode not in ("browse", "answer", "summarize"):
        errors.append(f"invalid mode: {mode!r}")
    
    # Target
    target = raw.get("target")
    if not isinstance(target, str):
        errors.append(f"target must be string")
    
    # Empty target validation
    if mode in ("answer", "summarize") and not target:
        deictic = raw.get("deictic_limit")
        if not deictic:
            errors.append(f"empty target requires deictic_limit for mode={mode}")
    
    # Temporal consistency
    temporal_kind = raw.get("temporal_kind")
    temporal_raw = raw.get("temporal_raw")
    temporal_n = raw.get("temporal_n")
    
    if temporal_kind:
        allowed_kinds = {None, "yesterday", "today", "last_week", "this_week",
                        "last_month", "this_month", "last_n_days", "explicit_date", "date_range"}
        if temporal_kind not in allowed_kinds:
            errors.append(f"invalid temporal_kind: {temporal_kind!r}")
        if not temporal_raw:
            errors.append("temporal_raw required when temporal_kind set")
        if temporal_kind == "last_n_days" and not isinstance(temporal_n, int):
            errors.append("temporal_n required for last_n_days")
    
    # Segment consistency (fail-closed on inference attempt)
    seg_requested = raw.get("segment_scope_requested", False)
    seg_phrase = raw.get("segment_phrase")
    
    if not isinstance(seg_requested, bool):
        errors.append("segment_scope_requested must be boolean")
    if seg_requested and not seg_phrase:
        errors.append("segment_phrase required when segment_scope_requested=true")
    if not seg_requested and seg_phrase:
        errors.append("segment_phrase set but segment_scope_requested=false (inference rejected)")
    
    # Speaker
    speaker = raw.get("speaker")
    if speaker not in (None, "user", "assistant"):
        errors.append(f"invalid speaker: {speaker!r}")
    
    # Deictic
    deictic = raw.get("deictic_limit")
    if deictic is not None and (not isinstance(deictic, int) or deictic < 1):
        errors.append(f"deictic_limit must be positive int")
    
    if errors:
        return None, "; ".join(errors)
    
    # Build AST
    query = Query(terms=(target,), spans=((0, len(target)),)) if target else None
    temporal = None
    if temporal_kind:
        temporal = Temporal(kind=temporal_kind, raw=temporal_raw, n=temporal_n)
    segment = Segment(phrase=seg_phrase) if seg_requested else None
    speaker_node = Speaker(role=speaker) if speaker else None
    deictic_node = Deictic(count=deictic) if deictic else None
    
    return MQLCommand(
        mode=mode,
        query=query,
        temporal=temporal,
        segment=segment,
        speaker=speaker_node,
        deictic=deictic_node
    ), ""
```

### 5.2 LLM Invocation

```python
STAGE_B_PROMPT = '''Parse this memory recall query into structured fields.

Output a JSON object with EXACTLY these fields (no others):
{
  "mode": "browse" | "answer" | "summarize",
  "target": "<search terms>",
  "temporal_kind": null | "yesterday" | "today" | "last_week" | "this_week" | "last_month" | "this_month" | "last_n_days" | "explicit_date" | "date_range",
  "temporal_raw": "<exact phrase>" | null,
  "temporal_n": <int for last_n_days> | null,
  "segment_scope_requested": true | false,
  "segment_phrase": "<topic name>" | null,
  "speaker": "user" | "assistant" | null,
  "deictic_limit": <int> | null
}

RULES:
- segment_scope_requested=true ONLY for explicit "in topic X", "topic:X", "in segment X"
- "when we discussed X" means target=X, NOT segment scope
- speaker requires speech verb: "you said" → assistant, "I asked" → user
- empty target requires deictic_limit for answer/summarize modes

Query: {query}
'''


def stage_b_parse(query: str, config: dict) -> Union[MQLCommand, FreeText]:
    """
    Parse via LLM with strict validation.
    On any failure, returns FreeText (fail-closed).
    """
    import json
    
    prompt = STAGE_B_PROMPT.format(query=query)
    
    try:
        response = llm_call(
            model=config.get("llm_model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        raw = json.loads(response)
        ast, error = validate_stage_b_output(raw)
        
        if ast:
            return ast
        else:
            return FreeText(text=query, reason=f"validation failed: {error}")
    
    except json.JSONDecodeError as e:
        return FreeText(text=query, reason=f"JSON error: {e}")
    
    except Exception as e:
        return FreeText(text=query, reason=f"LLM error: {e}")
```

---

## 6. Resolver

### 6.1 Contract

```python
@dataclass(frozen=True)
class ResolvedQuery:
    """Final resolved query ready for retrieval pipeline."""
    mode: str                                        # "browse", "answer", "summarize"
    target: str                                      # Search terms (may be empty for browse)
    temporal: Optional[Tuple[datetime, datetime]]    # UTC half-open [start, end)
    segment_scope: Optional[List[str]]               # None | [] | [node_ids]
    speaker: Optional[str]                           # None | "user" | "assistant"
    deictic_limit: Optional[int]
    parse_stage: str                                 # "A" or "B"
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target": self.target,
            "temporal": [t.isoformat() for t in self.temporal] if self.temporal else None,
            "segment_scope": self.segment_scope,
            "speaker": self.speaker,
            "deictic_limit": self.deictic_limit,
            "parse_stage": self.parse_stage,
        }
```

### 6.2 Temporal Resolution (DST-Safe)

```python
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta


def resolve_temporal(
    t: Temporal,
    now_utc: datetime,
    user_tz: str
) -> Tuple[datetime, datetime]:
    """
    Convert Temporal to UTC half-open interval [start, end).
    
    Boundaries are local midnight converted to UTC.
    DST transitions handled by zoneinfo.
    """
    tz = ZoneInfo(user_tz)
    utc = ZoneInfo("UTC")
    local_now = now_utc.astimezone(tz)
    
    def local_midnight(dt: datetime) -> datetime:
        """Return start of day in local timezone."""
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    today = local_midnight(local_now)
    
    if t.kind == "yesterday":
        start = today - timedelta(days=1)
        end = today
    
    elif t.kind == "today":
        start = today
        end = today + timedelta(days=1)
    
    elif t.kind == "last_week":
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
        start = tz.localize(local_midnight(parsed))
        end = start + timedelta(days=1)
    
    elif t.kind == "date_range":
        from dateutil import parser as dateparser
        d1 = dateparser.parse(t.date1, default=local_now.replace(tzinfo=None))
        d2 = dateparser.parse(t.date2, default=local_now.replace(tzinfo=None))
        start = tz.localize(local_midnight(d1))
        end = tz.localize(local_midnight(d2)) + timedelta(days=1)
    
    else:
        raise ValueError(f"Unknown temporal kind: {t.kind}")
    
    return (start.astimezone(utc), end.astimezone(utc))
```

### 6.3 Segment Resolution

```python
def resolve_segment(conn: sqlite3.Connection, phrase: str) -> List[str]:
    """
    Resolve segment phrase to node IDs.
    Returns [] if not found (tri-state: explicit request, no match).
    """
    topics = get_all_topics(conn)
    phrase_lower = phrase.lower().strip()
    
    # Exact match (highest priority)
    for topic in topics:
        name = topic['name'].lower()
        if name == phrase_lower or name.replace('-', ' ') == phrase_lower:
            nodes, _ = get_cached_segment_nodes(conn, topic['id'])
            return nodes
    
    # Contains match
    for topic in topics:
        if phrase_lower in topic['name'].lower():
            nodes, _ = get_cached_segment_nodes(conn, topic['id'])
            return nodes
    
    # No match
    return []
```

### 6.4 Full Resolver

```python
class Resolver:
    def __init__(self, conn: sqlite3.Connection, now_utc: datetime, user_tz: str):
        self.conn = conn
        self.now_utc = now_utc
        self.user_tz = user_tz
    
    def resolve(self, ast: Union[MQLCommand, FreeText], parse_stage: str) -> ResolvedQuery:
        if isinstance(ast, FreeText):
            # Treat FreeText as plain query with no scopes
            return ResolvedQuery(
                mode="answer",
                target=ast.text,
                temporal=None,
                segment_scope=None,
                speaker=None,
                deictic_limit=None,
                parse_stage=parse_stage
            )
        
        # Resolve MQLCommand
        target = ast.query.text if ast.query else ""
        
        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)
        
        segment_scope = None
        if ast.segment:
            segment_scope = resolve_segment(self.conn, ast.segment.phrase)
        
        speaker = ast.speaker.role if ast.speaker else None
        deictic = ast.deictic.count if ast.deictic else None
        
        return ResolvedQuery(
            mode=ast.mode,
            target=target,
            temporal=temporal,
            segment_scope=segment_scope,
            speaker=speaker,
            deictic_limit=deictic,
            parse_stage=parse_stage
        )
```

---

## 7. Retrieval Routing Contract

The resolver outputs `ResolvedQuery` which maps to retrieval behavior:

| Field | Tri-State | Retrieval Behavior |
|-------|-----------|-------------------|
| `segment_scope=None` | Not requested | Search all nodes |
| `segment_scope=[]` | Requested, not found | Return empty (no fallback) |
| `segment_scope=[ids]` | Requested, found | Search only these nodes |
| `speaker=None` | Both | Lexical + semantic |
| `speaker="user"/"assistant"` | Filtered | Lexical only (semantic disabled) |
| `mode="browse"`, empty target | N/A | Return recent exchanges |
| `mode="answer"/"summarize"`, empty target, no deictic | N/A | Return empty (strict) |

**Invariant:** Speaker scope disables semantic retrieval because embeddings are exchange-level.

---

## 8. AUDIT Logging

```python
def serialize_tokens(tokens: List[Token]) -> List[Tuple[str, str, int, int]]:
    """Stable serialization for logging."""
    return [t.to_tuple() for t in tokens if t.type != TokenType.EOF]


def understand_query(
    query: str,
    conn: sqlite3.Connection,
    now_utc: datetime,
    config: dict
) -> ResolvedQuery:
    """
    Full parse pipeline with audit logging.
    """
    user_tz = config.get("timezone", "America/Chicago")
    
    # Tokenize
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    logger.debug("AUDIT: [lex] input=%r tokens=%s", query, serialize_tokens(tokens))
    
    # Parse (Stage A)
    parser = Parser(tokens)
    ast = parser.parse()
    parse_stage = "A"
    logger.debug("AUDIT: [parse_a] ast=%s", json.dumps(ast.to_dict(), sort_keys=True))
    
    # Stage B if FreeText
    if isinstance(ast, FreeText):
        logger.debug("AUDIT: [parse_a] free_text reason=%r, invoking stage_b", ast.reason)
        ast = stage_b_parse(query, config)
        parse_stage = "B"
        logger.debug("AUDIT: [parse_b] ast=%s", json.dumps(ast.to_dict(), sort_keys=True))
    
    # Resolve
    resolver = Resolver(conn, now_utc, user_tz)
    resolved = resolver.resolve(ast, parse_stage)
    logger.debug("AUDIT: [resolve] result=%s", json.dumps(resolved.to_dict(), sort_keys=True))
    
    return resolved
```

---

## 9. Golden Fixtures

### 9.1 Format

```python
@dataclass
class GoldenFixture:
    name: str
    input: str
    now_utc: str  # ISO format
    expected_tokens: Optional[List[Tuple[str, str]]]  # [(type, lexeme), ...]
    expected_ast: dict
    expected_resolved: dict
```

### 9.2 Explicit Command Forms

```python
FIXTURES_EXPLICIT = [
    GoldenFixture(
        name="explicit_browse_prefix",
        input="Browse: coffee",
        now_utc="2026-01-25T18:00:00Z",
        expected_tokens=[("KW_BROWSE", "Browse"), ("COLON", ":"), ("WORD", "coffee")],
        expected_ast={"type": "MQLCommand", "mode": "browse", "query": {"terms": ["coffee"]}},
        expected_resolved={"mode": "browse", "target": "coffee", "segment_scope": None},
    ),
    
    GoldenFixture(
        name="explicit_answer_prefix",
        input="Answer: what is BM25",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={"type": "MQLCommand", "mode": "answer", "query": {"terms": ["what", "is", "bm25"]}},
        expected_resolved={"mode": "answer", "target": "what is bm25"},
    ),
    
    GoldenFixture(
        name="explicit_summarize_prefix",
        input="Summarize: our coffee discussion",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={"type": "MQLCommand", "mode": "summarize"},
        expected_resolved={"mode": "summarize"},
    ),
]
```

### 9.3 Temporal Modifiers

```python
FIXTURES_TEMPORAL = [
    GoldenFixture(
        name="temporal_yesterday",
        input="what did you say about coffee yesterday",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "mode": "answer",
            "temporal": {"kind": "yesterday"},
            "speaker": {"role": "assistant"},
            "query": {"terms": ["coffee"]},
        },
        expected_resolved={
            "mode": "answer",
            "target": "coffee",
            "temporal": ["2026-01-24T06:00:00+00:00", "2026-01-25T06:00:00+00:00"],
            "speaker": "assistant",
        },
    ),
    
    GoldenFixture(
        name="temporal_last_week",
        input="topic:coffee last week",
        now_utc="2026-01-25T18:00:00Z",  # Saturday
        expected_ast={
            "type": "MQLCommand",
            "temporal": {"kind": "last_week"},
            "segment": {"phrase": "coffee"},
        },
        expected_resolved={
            # Mon Jan 13 to Mon Jan 20 CST
            "temporal": ["2026-01-13T06:00:00+00:00", "2026-01-20T06:00:00+00:00"],
        },
    ),
    
    GoldenFixture(
        name="temporal_last_5_days",
        input="show me coffee last 5 days",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "mode": "browse",
            "temporal": {"kind": "last_n_days", "n": 5},
        },
        expected_resolved={
            "temporal": ["2026-01-20T06:00:00+00:00", "2026-01-26T06:00:00+00:00"],
        },
    ),
    
    GoldenFixture(
        name="temporal_explicit_date",
        input="on Jan 20 what did we discuss",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "temporal": {"kind": "explicit_date", "date1": "Jan 20"},
        },
        expected_resolved={
            "temporal": ["2026-01-20T06:00:00+00:00", "2026-01-21T06:00:00+00:00"],
        },
    ),
]
```

### 9.4 Deictic vs Temporal Disambiguation

```python
FIXTURES_DEICTIC = [
    GoldenFixture(
        name="deictic_last_10_messages",
        input="last 10 messages",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "mode": "browse",
            "deictic": {"count": 10},
            "temporal": None,  # NOT temporal
        },
        expected_resolved={
            "mode": "browse",
            "target": "",
            "deictic_limit": 10,
            "temporal": None,
        },
    ),
    
    GoldenFixture(
        name="deictic_last_5_messages_about_coffee",
        input="last 5 messages about coffee",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "mode": "browse",
            "deictic": {"count": 5},
            "query": {"terms": ["coffee"]},
        },
        expected_resolved={
            "deictic_limit": 5,
            "target": "coffee",
        },
    ),
    
    GoldenFixture(
        name="temporal_last_5_days_not_messages",
        input="last 5 days coffee",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "temporal": {"kind": "last_n_days", "n": 5},
            "deictic": None,  # NOT deictic
        },
        expected_resolved={
            "deictic_limit": None,
            "temporal": ["2026-01-20T06:00:00+00:00", "2026-01-26T06:00:00+00:00"],
        },
    ),
]
```

### 9.5 Segment Scope (Explicit Only)

```python
FIXTURES_SEGMENT = [
    GoldenFixture(
        name="segment_explicit_topic_colon",
        input="topic:FTS-migration what did we decide",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "segment": {"phrase": "fts-migration"},
        },
        expected_resolved={
            "segment_scope": "RESOLVE",  # DB-dependent
        },
    ),
    
    GoldenFixture(
        name="segment_explicit_in_topic",
        input="in topic coffee-brewing what was the ratio",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "segment": {"phrase": "coffee-brewing"},
        },
        expected_resolved={
            "segment_scope": "RESOLVE",
        },
    ),
    
    # CRITICAL: "in the morning" must NOT trigger segment scope
    GoldenFixture(
        name="no_segment_in_the_morning",
        input="in the morning we discussed coffee",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "segment": None,  # NO segment scope
            "query": {"terms": ["morning", "discussed", "coffee"]},
        },
        expected_resolved={
            "segment_scope": None,  # NOT []
        },
    ),
    
    # CRITICAL: "when we discussed X" must NOT trigger segment scope
    GoldenFixture(
        name="no_segment_discussed",
        input="when we discussed coffee yesterday",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "mode": "browse",
            "segment": None,
            "query": {"terms": ["coffee"]},
            "temporal": {"kind": "yesterday"},
        },
        expected_resolved={
            "segment_scope": None,
            "target": "coffee",
        },
    ),
    
    # Edge case: "topic:" with missing phrase
    GoldenFixture(
        name="segment_empty_phrase",
        input="topic: yesterday",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            # "yesterday" consumed as phrase, no temporal
            "segment": {"phrase": "yesterday"},
        },
        expected_resolved={
            "segment_scope": [],  # Requested, not found
        },
    ),
]
```

### 9.6 Speaker Scope

```python
FIXTURES_SPEAKER = [
    GoldenFixture(
        name="speaker_did_i_ever",
        input="did I ever ask about BM25",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "mode": "browse",
            "speaker": {"role": "user"},
            "query": {"terms": ["bm25"]},
        },
        expected_resolved={
            "speaker": "user",
        },
    ),
    
    GoldenFixture(
        name="speaker_only_what_you_said",
        input="only what you said about segment caching",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "speaker": {"role": "assistant"},
        },
        expected_resolved={
            "speaker": "assistant",
        },
    ),
]
```

### 9.7 Edit Stability

```python
FIXTURES_EDIT_STABILITY = [
    GoldenFixture(
        name="case_insensitive",
        input="BROWSE: Coffee",
        now_utc="2026-01-25T18:00:00Z",
        expected_resolved={"mode": "browse", "target": "coffee"},
    ),
    
    GoldenFixture(
        name="extra_whitespace",
        input="browse:   coffee   yesterday",
        now_utc="2026-01-25T18:00:00Z",
        expected_resolved={"mode": "browse", "target": "coffee", "temporal": ["2026-01-24T06:00:00+00:00", "2026-01-25T06:00:00+00:00"]},
    ),
    
    GoldenFixture(
        name="trailing_punctuation",
        input="Browse: coffee?",
        now_utc="2026-01-25T18:00:00Z",
        expected_resolved={"mode": "browse", "target": "coffee"},
    ),
    
    GoldenFixture(
        name="quoted_phrase_preserved",
        input='Browse: "cold exposure" yesterday',
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "query": {"terms": ["cold exposure"]},  # Atomic
        },
        expected_resolved={
            "target": "cold exposure",
        },
    ),
]
```

### 9.8 DST Boundary Tests

```python
FIXTURES_DST = [
    # Spring forward: March 10, 2024 at 2am becomes 3am
    GoldenFixture(
        name="dst_spring_forward_yesterday",
        input="yesterday",
        now_utc="2024-03-10T15:00:00Z",  # 11am EDT
        expected_resolved={
            # March 9 midnight EST (UTC-5) = 05:00 UTC
            # March 10 midnight EDT (UTC-4) = 04:00 UTC
            "temporal": ["2024-03-09T05:00:00+00:00", "2024-03-10T04:00:00+00:00"],
        },
    ),
    
    # Fall back: November 3, 2024 at 2am repeats
    GoldenFixture(
        name="dst_fall_back_yesterday",
        input="yesterday",
        now_utc="2024-11-03T15:00:00Z",  # 10am EST
        expected_resolved={
            # Nov 2 midnight EDT (UTC-4) = 04:00 UTC
            # Nov 3 midnight EST (UTC-5) = 05:00 UTC
            "temporal": ["2024-11-02T04:00:00+00:00", "2024-11-03T05:00:00+00:00"],
        },
    ),
]
```

### 9.9 Non-Memory Text / Stage B Routing

```python
FIXTURES_FREETEXT = [
    GoldenFixture(
        name="freetext_prose",
        input="the quick brown fox jumped over the lazy dog",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",  # Parser produces MQLCommand, not FreeText
            "mode": "answer",
            "query": {"terms": ["quick", "brown", "fox", "jumped", "over", "lazy", "dog"]},
            "segment": None,
            "temporal": None,
        },
        expected_resolved={
            "mode": "answer",
            "segment_scope": None,
        },
    ),
]
```

---

## 10. Test Runner

```python
def run_golden_fixtures():
    all_fixtures = (
        FIXTURES_EXPLICIT + FIXTURES_TEMPORAL + FIXTURES_DEICTIC +
        FIXTURES_SEGMENT + FIXTURES_SPEAKER + FIXTURES_EDIT_STABILITY +
        FIXTURES_DST + FIXTURES_FREETEXT
    )
    
    failures = []
    
    for f in all_fixtures:
        now_utc = datetime.fromisoformat(f.now_utc.replace("Z", "+00:00"))
        
        # Tokenize
        tokens = Lexer(f.input).tokenize()
        
        if f.expected_tokens:
            actual = [(t.type.name, t.lexeme) for t in tokens if t.type != TokenType.EOF]
            if actual != f.expected_tokens:
                failures.append(f"{f.name}: tokens mismatch\n  expected: {f.expected_tokens}\n  actual: {actual}")
        
        # Parse
        ast = Parser(tokens).parse()
        
        if f.expected_ast:
            actual_ast = ast.to_dict()
            for key, expected in f.expected_ast.items():
                if expected is not None and actual_ast.get(key) != expected:
                    failures.append(f"{f.name}: ast.{key} mismatch\n  expected: {expected}\n  actual: {actual_ast.get(key)}")
        
        # Resolve
        resolver = Resolver(mock_conn(), now_utc, "America/Chicago")
        resolved = resolver.resolve(ast, "A")
        
        if f.expected_resolved:
            actual_resolved = resolved.to_dict()
            for key, expected in f.expected_resolved.items():
                if expected == "RESOLVE":
                    continue
                if actual_resolved.get(key) != expected:
                    failures.append(f"{f.name}: resolved.{key} mismatch\n  expected: {expected}\n  actual: {actual_resolved.get(key)}")
    
    if failures:
        for fail in failures:
            print(f"FAIL: {fail}")
        raise AssertionError(f"{len(failures)} fixture(s) failed")
    
    print(f"OK: {len(all_fixtures)} fixtures passed")
```

---

## 11. Success Criteria

1. **No accidental segment scope** — "in the morning" and "when we discussed X" do NOT trigger segment_scope
2. **LAST disambiguation** — "last 5 messages" → deictic; "last 5 days" → temporal
3. **Mode precedence** — Explicit prefix > what did > when/where did > show/list > default
4. **Soft keywords** — Unused keywords become query terms
5. **DST-safe** — All temporal tests pass across spring forward / fall back
6. **Fail-closed Stage B** — Invalid LLM output → FreeText, unknown keys rejected
7. **Edit stability** — Case, whitespace, punctuation don't change parse
8. **Audit stability** — Logging uses deterministic serialization (to_tuple, to_dict, sort_keys)
9. **All golden fixtures pass**
