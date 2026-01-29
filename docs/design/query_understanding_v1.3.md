# Query Understanding Specification v1.3
## Tokenizer + Parser Architecture

---

## 1. Design Goals

1. **Deterministic** — Same input → same AST → same ResolvedQuery
2. **Auditable** — Token stream and AST logged for every parse
3. **Compositional** — Modifiers combine without pattern explosion
4. **Extensible** — Add new keywords/constructs without rewriting
5. **Testable** — Golden fixtures with frozen `now_utc`

---

## 2. Token Specification

### 2.1 Token Types

```python
class TokenType(Enum):
    # Mode keywords
    BROWSE = auto()      # browse, show, list, display, when, where
    ANSWER = auto()      # answer, what
    SUMMARIZE = auto()   # summarize, summary
    
    # Scope keywords
    TOPIC = auto()       # topic, segment
    IN = auto()          # in, within
    ABOUT = auto()       # about
    
    # Speaker keywords
    I = auto()           # i, me, my
    YOU = auto()         # you, your
    WE = auto()          # we, our
    
    # Temporal keywords
    YESTERDAY = auto()
    TODAY = auto()
    LAST = auto()        # last (week/month/N days)
    THIS = auto()        # this (week/month)
    ON = auto()          # on (date)
    BETWEEN = auto()     # between X and Y
    AND = auto()
    
    # Units
    WEEK = auto()
    MONTH = auto()
    DAY = auto()
    DAYS = auto()
    MESSAGES = auto()    # messages, exchanges
    
    # Verbs (for pattern detection)
    DID = auto()
    SAID = auto()        # said, say, asked, mentioned, recommended
    DISCUSSED = auto()   # discussed, talked
    EVER = auto()
    
    # Structure
    COLON = auto()       # :
    COMMA = auto()       # ,
    QUESTION = auto()    # ?
    
    # Literals
    NUMBER = auto()      # 10, 5, etc.
    QUOTED = auto()      # "phrase" or 'phrase'
    MONTH_NAME = auto()  # jan, january, feb, etc.
    WORD = auto()        # anything else
    
    # Special
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str           # Original text
    span: Tuple[int, int]  # (start, end) in original string
    normalized: str      # Lowercase, stripped
```

### 2.2 Keyword Mapping

```python
KEYWORDS = {
    # Mode
    "browse": TokenType.BROWSE,
    "show": TokenType.BROWSE,
    "list": TokenType.BROWSE,
    "display": TokenType.BROWSE,
    "when": TokenType.BROWSE,
    "where": TokenType.BROWSE,
    "answer": TokenType.ANSWER,
    "what": TokenType.ANSWER,
    "summarize": TokenType.SUMMARIZE,
    "summary": TokenType.SUMMARIZE,
    
    # Scope
    "topic": TokenType.TOPIC,
    "segment": TokenType.TOPIC,
    "in": TokenType.IN,
    "within": TokenType.IN,
    "about": TokenType.ABOUT,
    
    # Speaker
    "i": TokenType.I,
    "me": TokenType.I,
    "my": TokenType.I,
    "you": TokenType.YOU,
    "your": TokenType.YOU,
    "we": TokenType.WE,
    "our": TokenType.WE,
    
    # Temporal
    "yesterday": TokenType.YESTERDAY,
    "today": TokenType.TODAY,
    "last": TokenType.LAST,
    "this": TokenType.THIS,
    "on": TokenType.ON,
    "between": TokenType.BETWEEN,
    "and": TokenType.AND,
    
    # Units
    "week": TokenType.WEEK,
    "month": TokenType.MONTH,
    "day": TokenType.DAY,
    "days": TokenType.DAYS,
    "messages": TokenType.MESSAGES,
    "exchanges": TokenType.MESSAGES,
    "message": TokenType.MESSAGES,
    "exchange": TokenType.MESSAGES,
    
    # Verbs
    "did": TokenType.DID,
    "said": TokenType.SAID,
    "say": TokenType.SAID,
    "asked": TokenType.SAID,
    "ask": TokenType.SAID,
    "mentioned": TokenType.SAID,
    "mention": TokenType.SAID,
    "recommended": TokenType.SAID,
    "recommend": TokenType.SAID,
    "discussed": TokenType.DISCUSSED,
    "discuss": TokenType.DISCUSSED,
    "talked": TokenType.DISCUSSED,
    "talk": TokenType.DISCUSSED,
    
    # Other
    "ever": TokenType.EVER,
}

MONTH_NAMES = {
    "jan", "january", "feb", "february", "mar", "march",
    "apr", "april", "may", "jun", "june", "jul", "july",
    "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december"
}
```

### 2.3 Lexer

```python
class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
            
            start = self.pos
            char = self.text[self.pos]
            
            if char == ':':
                self._emit(TokenType.COLON, start, start + 1)
                self.pos += 1
            elif char == ',':
                self._emit(TokenType.COMMA, start, start + 1)
                self.pos += 1
            elif char == '?':
                self._emit(TokenType.QUESTION, start, start + 1)
                self.pos += 1
            elif char in ('"', "'"):
                self._read_quoted(char)
            elif char.isdigit():
                self._read_number()
            else:
                self._read_word()
        
        self.tokens.append(Token(TokenType.EOF, "", (self.pos, self.pos), ""))
        return self.tokens
    
    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
    
    def _emit(self, type: TokenType, start: int, end: int):
        value = self.text[start:end]
        self.tokens.append(Token(type, value, (start, end), value.lower()))
    
    def _read_quoted(self, quote_char: str):
        start = self.pos
        self.pos += 1  # Skip opening quote
        while self.pos < len(self.text) and self.text[self.pos] != quote_char:
            self.pos += 1
        if self.pos < len(self.text):
            self.pos += 1  # Skip closing quote
        value = self.text[start:self.pos]
        # Strip quotes for normalized
        inner = value[1:-1] if len(value) >= 2 else value
        self.tokens.append(Token(TokenType.QUOTED, value, (start, self.pos), inner.lower()))
    
    def _read_number(self):
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        value = self.text[start:self.pos]
        self.tokens.append(Token(TokenType.NUMBER, value, (start, self.pos), value))
    
    def _read_word(self):
        start = self.pos
        while self.pos < len(self.text) and self._is_word_char(self.text[self.pos]):
            self.pos += 1
        value = self.text[start:self.pos]
        normalized = value.lower()
        
        # Classify
        if normalized in KEYWORDS:
            token_type = KEYWORDS[normalized]
        elif normalized in MONTH_NAMES:
            token_type = TokenType.MONTH_NAME
        else:
            token_type = TokenType.WORD
        
        self.tokens.append(Token(token_type, value, (start, self.pos), normalized))
    
    def _is_word_char(self, c: str) -> bool:
        return c.isalnum() or c in ('-', '_')
```

---

## 3. AST Definition

### 3.1 Node Types

```python
class ASTNode:
    """Base class for all AST nodes."""
    pass


@dataclass
class Query(ASTNode):
    """The target search terms."""
    terms: List[str]
    spans: List[Tuple[int, int]]  # Source locations
    
    @property
    def text(self) -> str:
        return " ".join(self.terms)


@dataclass
class Temporal(ASTNode):
    """Temporal scope."""
    kind: Literal[
        "yesterday", "today",
        "last_week", "this_week", "last_month", "this_month",
        "last_n_days", "explicit_date", "date_range"
    ]
    raw: str
    n: Optional[int] = None  # For last_n_days
    date1: Optional[str] = None  # For explicit_date or date_range start
    date2: Optional[str] = None  # For date_range end


@dataclass
class Segment(ASTNode):
    """Segment/topic scope."""
    phrase: str
    explicit: bool = True  # Always True in parser (only explicit triggers scope)


@dataclass
class Speaker(ASTNode):
    """Speaker scope."""
    role: Literal["user", "assistant"]


@dataclass
class Deictic(ASTNode):
    """Deictic limit (last N messages)."""
    count: int
    unit: Literal["messages", "exchanges"]


@dataclass
class MQLCommand(ASTNode):
    """Root AST node for a parsed memory query."""
    mode: Literal["browse", "answer", "summarize"]
    query: Optional[Query]
    temporal: Optional[Temporal]
    segment: Optional[Segment]
    speaker: Optional[Speaker]
    deictic: Optional[Deictic]


@dataclass
class FreeText(ASTNode):
    """Fallback: unparseable input, route to Stage B."""
    text: str
    reason: str
```

---

## 4. Parser (Recursive Descent)

### 4.1 Grammar (Informal EBNF)

```
command     ::= mode_phrase modifiers* query?

mode_phrase ::= browse_phrase | answer_phrase | summarize_phrase | implicit

browse_phrase    ::= "browse" ":" 
                   | "show" "me"?
                   | "when" "did"? ("we" | "I" | "you") (discuss_verb | say_verb)
                   | "where" "did"? "we" discuss_verb
                   | "did" ("I" | "you") "ever"? say_verb
                   | "list"
                   | "display"

answer_phrase    ::= "answer" ":"
                   | "what" "did" ("we" | "you") (say_verb | "decide" | "conclude")

summarize_phrase ::= "summarize" ":"?
                   | "summary" "of"?

modifiers   ::= segment_mod | temporal_mod | speaker_mod | deictic_mod

segment_mod ::= ("topic" | "segment") ":" phrase
              | "in" ("topic" | "segment")? phrase

temporal_mod ::= "yesterday"
               | "today"  
               | "last" ("week" | "month" | NUMBER "days"?)
               | "this" ("week" | "month")
               | "on" date
               | "between" date "and" date

speaker_mod ::= "only"? "what" ("I" | "you") say_verb
              | ("my" | "your") "messages"?

deictic_mod ::= "last" NUMBER ("messages" | "exchanges")

query       ::= ("about" | phrase_start) phrase
              | phrase  -- remaining words

phrase      ::= QUOTED | WORD+
date        ::= MONTH_NAME NUMBER (NUMBER)?  -- Jan 24 or Jan 24 2026

discuss_verb ::= "discussed" | "talked" "about"?
say_verb     ::= "said" | "asked" | "mentioned" | "recommended"
```

### 4.2 Parser Implementation

```python
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        
        # Accumulated parse state
        self.mode: Optional[str] = None
        self.query_terms: List[str] = []
        self.query_spans: List[Tuple[int, int]] = []
        self.temporal: Optional[Temporal] = None
        self.segment: Optional[Segment] = None
        self.speaker: Optional[Speaker] = None
        self.deictic: Optional[Deictic] = None
    
    def parse(self) -> Union[MQLCommand, FreeText]:
        """Main entry point."""
        try:
            self._parse_command()
            return self._build_result()
        except ParseError as e:
            return FreeText(
                text=self._original_text(),
                reason=str(e)
            )
    
    # ─────────────────────────────────────────────────────────────
    # Token helpers
    # ─────────────────────────────────────────────────────────────
    
    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF
    
    def _at(self, *types: TokenType) -> bool:
        return self._peek().type in types
    
    def _match(self, *types: TokenType) -> Optional[Token]:
        if self._at(*types):
            tok = self._peek()
            self.pos += 1
            return tok
        return None
    
    def _expect(self, *types: TokenType) -> Token:
        tok = self._match(*types)
        if not tok:
            raise ParseError(f"Expected {types}, got {self._peek().type}")
        return tok
    
    def _at_end(self) -> bool:
        return self._at(TokenType.EOF)
    
    # ─────────────────────────────────────────────────────────────
    # Mode phrases
    # ─────────────────────────────────────────────────────────────
    
    def _parse_command(self):
        """Parse mode phrase, then modifiers, then remaining query."""
        self._try_mode_phrase()
        self._parse_modifiers()
        self._parse_remaining_query()
        
        # Default mode if not set
        if self.mode is None:
            self.mode = "answer"
    
    def _try_mode_phrase(self):
        """Try to parse an explicit mode phrase."""
        
        # "browse:" / "answer:" / "summarize:"
        if self._at(TokenType.BROWSE):
            self._match(TokenType.BROWSE)
            self._match(TokenType.COLON)
            self.mode = "browse"
            return
        
        if self._at(TokenType.ANSWER):
            tok = self._match(TokenType.ANSWER)
            if self._match(TokenType.COLON):
                self.mode = "answer"
                return
            # "what did you/we say/decide about X"
            if tok.normalized == "what":
                if self._match(TokenType.DID):
                    speaker_tok = self._match(TokenType.YOU, TokenType.WE, TokenType.I)
                    if speaker_tok:
                        if speaker_tok.type == TokenType.YOU:
                            self.speaker = Speaker(role="assistant")
                        verb = self._match(TokenType.SAID, TokenType.DISCUSSED)
                        self.mode = "answer"
                        return
            # Didn't match pattern, put back
            self.pos -= 1
        
        if self._at(TokenType.SUMMARIZE):
            self._match(TokenType.SUMMARIZE)
            self._match(TokenType.COLON)  # Optional
            self.mode = "summarize"
            return
        
        # "show me" / "list" / "display"
        if self._at(TokenType.BROWSE):
            self._match(TokenType.BROWSE)
            self.mode = "browse"
            # "show me" - consume optional "me"
            if self._peek().normalized == "me":
                self.pos += 1
            return
        
        # "when did we discuss X" / "where did we talk about X"
        if self._peek().normalized in ("when", "where"):
            self._match(TokenType.BROWSE)
            self._match(TokenType.DID)  # optional
            self._match(TokenType.WE, TokenType.I, TokenType.YOU)  # optional
            self._match(TokenType.DISCUSSED, TokenType.SAID)  # optional
            self.mode = "browse"
            return
        
        # "did I ever ask about X" / "did you ever say X"
        if self._at(TokenType.DID):
            self._match(TokenType.DID)
            speaker_tok = self._match(TokenType.I, TokenType.YOU)
            if speaker_tok:
                if speaker_tok.type == TokenType.I:
                    self.speaker = Speaker(role="user")
                else:
                    self.speaker = Speaker(role="assistant")
            self._match(TokenType.EVER)  # optional
            self._match(TokenType.SAID, TokenType.DISCUSSED)  # optional
            self.mode = "browse"
            return
    
    # ─────────────────────────────────────────────────────────────
    # Modifiers
    # ─────────────────────────────────────────────────────────────
    
    def _parse_modifiers(self):
        """Parse any number of modifiers until we hit query content."""
        while not self._at_end():
            if self._try_segment_modifier():
                continue
            if self._try_temporal_modifier():
                continue
            if self._try_speaker_modifier():
                continue
            if self._try_deictic_modifier():
                continue
            break  # No more modifiers
    
    def _try_segment_modifier(self) -> bool:
        """
        topic:X | segment:X | in topic X | in segment X | in X
        Only explicit forms trigger segment scope.
        """
        # "topic:" or "segment:"
        if self._at(TokenType.TOPIC):
            self._match(TokenType.TOPIC)
            if self._match(TokenType.COLON):
                phrase = self._parse_phrase()
                self.segment = Segment(phrase=phrase, explicit=True)
                return True
            else:
                # "topic" without colon might be query word, put back
                self.pos -= 1
                return False
        
        # "in topic X" / "in segment X" / "in X" (less explicit)
        if self._at(TokenType.IN):
            self._match(TokenType.IN)
            if self._match(TokenType.TOPIC):
                phrase = self._parse_phrase()
                self.segment = Segment(phrase=phrase, explicit=True)
                return True
            else:
                # "in X" without "topic/segment" - ambiguous, don't treat as segment
                self.pos -= 1
                return False
        
        return False
    
    def _try_temporal_modifier(self) -> bool:
        """yesterday | today | last week | last N days | on DATE | between DATE and DATE"""
        
        if self._match(TokenType.YESTERDAY):
            self.temporal = Temporal(kind="yesterday", raw="yesterday")
            return True
        
        if self._match(TokenType.TODAY):
            self.temporal = Temporal(kind="today", raw="today")
            return True
        
        if self._at(TokenType.LAST):
            self._match(TokenType.LAST)
            
            if self._match(TokenType.WEEK):
                self.temporal = Temporal(kind="last_week", raw="last week")
                return True
            
            if self._match(TokenType.MONTH):
                self.temporal = Temporal(kind="last_month", raw="last month")
                return True
            
            num_tok = self._match(TokenType.NUMBER)
            if num_tok:
                n = int(num_tok.value)
                if self._match(TokenType.DAYS, TokenType.DAY):
                    self.temporal = Temporal(kind="last_n_days", raw=f"last {n} days", n=n)
                    return True
                # Could be "last 10 messages" - deictic, not temporal
                if self._at(TokenType.MESSAGES):
                    self.pos -= 1  # Put back NUMBER
                    self.pos -= 1  # Put back LAST
                    return False
            
            # Just "last" without qualifier - put back
            self.pos -= 1
            return False
        
        if self._at(TokenType.THIS):
            self._match(TokenType.THIS)
            
            if self._match(TokenType.WEEK):
                self.temporal = Temporal(kind="this_week", raw="this week")
                return True
            
            if self._match(TokenType.MONTH):
                self.temporal = Temporal(kind="this_month", raw="this month")
                return True
            
            self.pos -= 1
            return False
        
        if self._at(TokenType.ON):
            self._match(TokenType.ON)
            date = self._try_parse_date()
            if date:
                self.temporal = Temporal(kind="explicit_date", raw=f"on {date}", date1=date)
                return True
            self.pos -= 1
            return False
        
        if self._at(TokenType.BETWEEN):
            self._match(TokenType.BETWEEN)
            date1 = self._try_parse_date()
            if date1 and self._match(TokenType.AND):
                date2 = self._try_parse_date()
                if date2:
                    self.temporal = Temporal(
                        kind="date_range",
                        raw=f"between {date1} and {date2}",
                        date1=date1,
                        date2=date2
                    )
                    return True
            # Failed, restore
            self.pos -= 1
            return False
        
        return False
    
    def _try_parse_date(self) -> Optional[str]:
        """Parse: MONTH_NAME NUMBER (NUMBER)?"""
        if not self._at(TokenType.MONTH_NAME):
            return None
        
        month_tok = self._match(TokenType.MONTH_NAME)
        day_tok = self._match(TokenType.NUMBER)
        if not day_tok:
            self.pos -= 1
            return None
        
        year_tok = self._match(TokenType.NUMBER)
        
        if year_tok:
            return f"{month_tok.value} {day_tok.value} {year_tok.value}"
        return f"{month_tok.value} {day_tok.value}"
    
    def _try_speaker_modifier(self) -> bool:
        """
        "only what I said" | "only what you said" | "my messages" | "your messages"
        Requires speech verb for "what I/you said".
        """
        # "only what I/you said"
        if self._peek().normalized == "only":
            self.pos += 1
            if self._peek().normalized == "what":
                self.pos += 1
                speaker_tok = self._match(TokenType.I, TokenType.YOU)
                if speaker_tok:
                    if self._match(TokenType.SAID):
                        role = "user" if speaker_tok.type == TokenType.I else "assistant"
                        self.speaker = Speaker(role=role)
                        return True
                # Failed pattern
                self.pos -= 3 if speaker_tok else -2
                return False
            self.pos -= 1
            return False
        
        # "my messages" / "your messages"
        if self._at(TokenType.I) and self._peek().normalized == "my":
            self._match(TokenType.I)
            if self._match(TokenType.MESSAGES):
                self.speaker = Speaker(role="user")
                return True
            self.pos -= 1
            return False
        
        if self._at(TokenType.YOU) and self._peek().normalized == "your":
            self._match(TokenType.YOU)
            if self._match(TokenType.MESSAGES):
                self.speaker = Speaker(role="assistant")
                return True
            self.pos -= 1
            return False
        
        return False
    
    def _try_deictic_modifier(self) -> bool:
        """last N messages | last N exchanges"""
        if not self._at(TokenType.LAST):
            return False
        
        self._match(TokenType.LAST)
        num_tok = self._match(TokenType.NUMBER)
        if num_tok:
            if self._match(TokenType.MESSAGES):
                self.deictic = Deictic(count=int(num_tok.value), unit="messages")
                return True
        
        # Not deictic, restore
        if num_tok:
            self.pos -= 1
        self.pos -= 1
        return False
    
    # ─────────────────────────────────────────────────────────────
    # Query extraction
    # ─────────────────────────────────────────────────────────────
    
    def _parse_remaining_query(self):
        """Collect remaining tokens as query."""
        # Skip "about" if present
        self._match(TokenType.ABOUT)
        
        while not self._at_end():
            tok = self._peek()
            
            # Skip trailing punctuation
            if tok.type in (TokenType.QUESTION, TokenType.COMMA):
                self.pos += 1
                continue
            
            # Skip filler words
            if tok.normalized in ("the", "a", "an", "and", "or", "of", "for", "to"):
                self.pos += 1
                continue
            
            # Accumulate query terms
            if tok.type in (TokenType.WORD, TokenType.QUOTED, TokenType.NUMBER):
                self.query_terms.append(tok.normalized if tok.type != TokenType.QUOTED else tok.normalized)
                self.query_spans.append(tok.span)
            
            self.pos += 1
    
    def _parse_phrase(self) -> str:
        """Parse a phrase (quoted or bare words until modifier/end)."""
        if self._at(TokenType.QUOTED):
            tok = self._match(TokenType.QUOTED)
            return tok.normalized
        
        words = []
        while self._at(TokenType.WORD, TokenType.NUMBER):
            tok = self._match(TokenType.WORD, TokenType.NUMBER)
            words.append(tok.normalized)
        
        return " ".join(words)
    
    # ─────────────────────────────────────────────────────────────
    # Result building
    # ─────────────────────────────────────────────────────────────
    
    def _build_result(self) -> MQLCommand:
        query = None
        if self.query_terms:
            query = Query(terms=self.query_terms, spans=self.query_spans)
        
        return MQLCommand(
            mode=self.mode or "answer",
            query=query,
            temporal=self.temporal,
            segment=self.segment,
            speaker=self.speaker,
            deictic=self.deictic
        )
    
    def _original_text(self) -> str:
        if not self.tokens:
            return ""
        start = self.tokens[0].span[0]
        end = self.tokens[-2].span[1] if len(self.tokens) > 1 else start
        return self.tokens[0].value  # Simplified


class ParseError(Exception):
    pass
```

---

## 5. Resolver (AST → ResolvedQuery)

### 5.1 ResolvedQuery Contract

```python
@dataclass
class ResolvedQuery:
    mode: Literal["answer", "browse", "summarize"]
    target: str
    temporal: Optional[Tuple[datetime, datetime]]  # UTC half-open [start, end)
    segment_scope: Optional[List[str]]              # None | [] | [node_ids]
    speaker: Optional[Literal["user", "assistant"]]
    deictic_limit: Optional[int]
    
    # Audit
    parse_stage: Literal["A", "B"]
    ast: Optional[Union[MQLCommand, FreeText]]
```

### 5.2 Resolver Implementation

```python
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta


class Resolver:
    def __init__(
        self,
        conn: sqlite3.Connection,
        now_utc: datetime,  # Injected, never call datetime.now()
        user_tz: str = "America/Chicago"
    ):
        self.conn = conn
        self.now_utc = now_utc
        self.user_tz = user_tz
        self.tz = ZoneInfo(user_tz)
    
    def resolve(self, ast: Union[MQLCommand, FreeText]) -> ResolvedQuery:
        if isinstance(ast, FreeText):
            # Route to Stage B
            return self._resolve_free_text(ast)
        
        return self._resolve_mql(ast)
    
    def _resolve_mql(self, cmd: MQLCommand) -> ResolvedQuery:
        # Target
        target = cmd.query.text if cmd.query else ""
        
        # Validate: answer/summarize require non-empty target (unless deictic)
        if cmd.mode in ("answer", "summarize") and not target and not cmd.deictic:
            # Could route to Stage B or return empty
            pass  # Allow for now, pipeline will handle
        
        # Temporal
        temporal = None
        if cmd.temporal:
            temporal = self._resolve_temporal(cmd.temporal)
        
        # Segment (only if explicit)
        segment_scope = None
        if cmd.segment and cmd.segment.explicit:
            segment_scope = self._resolve_segment(cmd.segment.phrase)
        
        # Speaker
        speaker = cmd.speaker.role if cmd.speaker else None
        
        # Deictic
        deictic = cmd.deictic.count if cmd.deictic else None
        
        return ResolvedQuery(
            mode=cmd.mode,
            target=target,
            temporal=temporal,
            segment_scope=segment_scope,
            speaker=speaker,
            deictic_limit=deictic,
            parse_stage="A",
            ast=cmd
        )
    
    def _resolve_temporal(self, t: Temporal) -> Tuple[datetime, datetime]:
        """Convert Temporal AST node to UTC half-open interval."""
        local_now = self.now_utc.astimezone(self.tz)
        
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
            days_since_monday = local_now.weekday()
            this_monday = today - timedelta(days=days_since_monday)
            start = this_monday - timedelta(days=7)
            end = this_monday
        
        elif t.kind == "this_week":
            days_since_monday = local_now.weekday()
            start = today - timedelta(days=days_since_monday)
            end = start + timedelta(days=7)
        
        elif t.kind == "last_month":
            first_of_month = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if local_now.month == 1:
                first_of_last = first_of_month.replace(year=local_now.year - 1, month=12)
            else:
                first_of_last = first_of_month.replace(month=local_now.month - 1)
            start = first_of_last
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
            parsed = dateparser.parse(t.date1, default=local_now)
            start = self.tz.localize(midnight(parsed))
            end = start + timedelta(days=1)
        
        elif t.kind == "date_range":
            from dateutil import parser as dateparser
            start_parsed = dateparser.parse(t.date1, default=local_now)
            end_parsed = dateparser.parse(t.date2, default=local_now)
            start = self.tz.localize(midnight(start_parsed))
            end = self.tz.localize(midnight(end_parsed)) + timedelta(days=1)
        
        else:
            raise ValueError(f"Unknown temporal kind: {t.kind}")
        
        utc = ZoneInfo("UTC")
        return (start.astimezone(utc), end.astimezone(utc))
    
    def _resolve_segment(self, phrase: str) -> List[str]:
        """Resolve segment phrase to node IDs. Returns [] if not found."""
        topics = get_all_topics(self.conn)
        phrase_lower = phrase.lower().strip()
        
        # Exact match first
        for topic in topics:
            if topic['name'].lower() == phrase_lower:
                nodes, _ = get_cached_segment_nodes(self.conn, topic['id'])
                return nodes
        
        # Contains match
        for topic in topics:
            if phrase_lower in topic['name'].lower():
                nodes, _ = get_cached_segment_nodes(self.conn, topic['id'])
                return nodes
        
        return []  # Not found → empty scope → empty results
    
    def _resolve_free_text(self, ft: FreeText) -> ResolvedQuery:
        """Placeholder: route to Stage B LLM."""
        return ResolvedQuery(
            mode="answer",
            target=ft.text,
            temporal=None,
            segment_scope=None,
            speaker=None,
            deictic_limit=None,
            parse_stage="B_pending",
            ast=ft
        )
```

---

## 6. Stage B Integration

```python
def understand_query(
    query: str,
    conn: sqlite3.Connection,
    now_utc: datetime,
    config: dict
) -> ResolvedQuery:
    """
    Full parse pipeline:
    1. Tokenize
    2. Parse to AST
    3. If FreeText → Stage B LLM
    4. Resolve AST → ResolvedQuery
    """
    # Tokenize
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    logger.debug("AUDIT: [tokenize] input=%r tokens=%r", query, tokens)
    
    # Parse
    parser = Parser(tokens)
    ast = parser.parse()
    logger.debug("AUDIT: [parse] ast=%r", ast)
    
    # Stage B for FreeText
    if isinstance(ast, FreeText):
        logger.debug("AUDIT: [parse] free_text, routing to stage_b")
        ast = stage_b_parse(query, config)
        if isinstance(ast, FreeText):
            # Stage B also failed, use as-is
            pass
    
    # Resolve
    resolver = Resolver(conn, now_utc, config.get("timezone", "America/Chicago"))
    result = resolver.resolve(ast)
    logger.debug("AUDIT: [resolve] result=%r", result)
    
    return result
```

---

## 7. Golden Fixtures

### 7.1 Fixture Format

```python
@dataclass
class GoldenFixture:
    input: str
    now_utc: str  # ISO format, frozen
    expected_tokens: List[Tuple[str, str]]  # (type, value)
    expected_ast: dict
    expected_resolved: dict


GOLDEN_FIXTURES = [
    # ─────────────────────────────────────────────────────────────
    # Explicit command forms
    # ─────────────────────────────────────────────────────────────
    GoldenFixture(
        input="Browse: coffee",
        now_utc="2026-01-25T18:00:00Z",
        expected_tokens=[
            ("BROWSE", "Browse"),
            ("COLON", ":"),
            ("WORD", "coffee"),
            ("EOF", ""),
        ],
        expected_ast={
            "type": "MQLCommand",
            "mode": "browse",
            "query": {"terms": ["coffee"]},
            "temporal": None,
            "segment": None,
            "speaker": None,
            "deictic": None,
        },
        expected_resolved={
            "mode": "browse",
            "target": "coffee",
            "temporal": None,
            "segment_scope": None,
            "speaker": None,
            "deictic_limit": None,
        },
    ),
    
    GoldenFixture(
        input="last 10 messages",
        now_utc="2026-01-25T18:00:00Z",
        expected_tokens=[
            ("LAST", "last"),
            ("NUMBER", "10"),
            ("MESSAGES", "messages"),
            ("EOF", ""),
        ],
        expected_ast={
            "type": "MQLCommand",
            "mode": "browse",
            "query": None,
            "deictic": {"count": 10, "unit": "messages"},
        },
        expected_resolved={
            "mode": "browse",
            "target": "",
            "deictic_limit": 10,
        },
    ),
    
    # ─────────────────────────────────────────────────────────────
    # Temporal modifiers
    # ─────────────────────────────────────────────────────────────
    GoldenFixture(
        input="what did you say about coffee yesterday",
        now_utc="2026-01-25T18:00:00Z",  # Saturday
        expected_ast={
            "type": "MQLCommand",
            "mode": "answer",
            "query": {"terms": ["coffee"]},
            "temporal": {"kind": "yesterday", "raw": "yesterday"},
            "speaker": {"role": "assistant"},
        },
        expected_resolved={
            "mode": "answer",
            "target": "coffee",
            # Jan 24 00:00 CST = Jan 24 06:00 UTC
            # Jan 25 00:00 CST = Jan 25 06:00 UTC
            "temporal": ("2026-01-24T06:00:00Z", "2026-01-25T06:00:00Z"),
            "speaker": "assistant",
        },
    ),
    
    GoldenFixture(
        input="topic:coffee last week",
        now_utc="2026-01-25T18:00:00Z",  # Saturday
        expected_ast={
            "type": "MQLCommand",
            "mode": "answer",
            "query": None,
            "temporal": {"kind": "last_week"},
            "segment": {"phrase": "coffee", "explicit": True},
        },
        expected_resolved={
            "mode": "answer",
            "target": "",
            # Last week = Mon Jan 13 to Mon Jan 20 (CST midnight → UTC)
            "temporal": ("2026-01-13T06:00:00Z", "2026-01-20T06:00:00Z"),
            "segment_scope": "RESOLVE",  # Placeholder, depends on DB
        },
    ),
    
    # ─────────────────────────────────────────────────────────────
    # Speaker scope
    # ─────────────────────────────────────────────────────────────
    GoldenFixture(
        input="did I ever ask about BM25",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "mode": "browse",
            "query": {"terms": ["bm25"]},
            "speaker": {"role": "user"},
        },
        expected_resolved={
            "mode": "browse",
            "target": "bm25",
            "speaker": "user",
        },
    ),
    
    GoldenFixture(
        input="only what you said about segment caching",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "mode": "answer",
            "query": {"terms": ["segment", "caching"]},
            "speaker": {"role": "assistant"},
        },
        expected_resolved={
            "mode": "answer",
            "target": "segment caching",
            "speaker": "assistant",
        },
    ),
    
    # ─────────────────────────────────────────────────────────────
    # Segment scope (explicit only)
    # ─────────────────────────────────────────────────────────────
    GoldenFixture(
        input="in topic FTS-migration what did we decide",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",
            "mode": "answer",
            "query": {"terms": ["decide"]},
            "segment": {"phrase": "fts-migration", "explicit": True},
        },
        expected_resolved={
            "segment_scope": "RESOLVE",  # Will be [] or [ids]
        },
    ),
    
    # ─────────────────────────────────────────────────────────────
    # Non-memory text (→ FreeText → Stage B)
    # ─────────────────────────────────────────────────────────────
    GoldenFixture(
        input="when we discussed coffee yesterday",
        now_utc="2026-01-25T18:00:00Z",
        expected_ast={
            "type": "MQLCommand",  # Should parse, not FreeText
            "mode": "browse",
            "query": {"terms": ["coffee"]},
            "temporal": {"kind": "yesterday"},
            # NO segment scope - "discussed coffee" is target, not scope
            "segment": None,
        },
        expected_resolved={
            "mode": "browse",
            "target": "coffee",
            "segment_scope": None,  # NOT inferred
        },
    ),
    
    # ─────────────────────────────────────────────────────────────
    # Minimal edit stability
    # ─────────────────────────────────────────────────────────────
    GoldenFixture(
        input="BROWSE: Coffee",  # Case change
        now_utc="2026-01-25T18:00:00Z",
        expected_resolved={
            "mode": "browse",
            "target": "coffee",
        },
    ),
    
    GoldenFixture(
        input="browse:  coffee",  # Extra space
        now_utc="2026-01-25T18:00:00Z",
        expected_resolved={
            "mode": "browse",
            "target": "coffee",
        },
    ),
    
    GoldenFixture(
        input="Browse: coffee?",  # Trailing punctuation
        now_utc="2026-01-25T18:00:00Z",
        expected_resolved={
            "mode": "browse",
            "target": "coffee",
        },
    ),
]
```

### 7.2 Test Runner

```python
def test_golden_fixtures():
    for fixture in GOLDEN_FIXTURES:
        now_utc = datetime.fromisoformat(fixture.now_utc.replace("Z", "+00:00"))
        
        # Tokenize
        lexer = Lexer(fixture.input)
        tokens = lexer.tokenize()
        
        if fixture.expected_tokens:
            actual_tokens = [(t.type.name, t.value) for t in tokens]
            assert actual_tokens == fixture.expected_tokens, f"Tokens mismatch for: {fixture.input}"
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        if fixture.expected_ast:
            actual_ast = ast_to_dict(ast)
            for key, expected_val in fixture.expected_ast.items():
                if expected_val is not None:
                    assert actual_ast.get(key) == expected_val, \
                        f"AST.{key} mismatch for: {fixture.input}"
        
        # Resolve
        resolver = Resolver(mock_conn(), now_utc, "America/Chicago")
        resolved = resolver.resolve(ast)
        
        if fixture.expected_resolved:
            for key, expected_val in fixture.expected_resolved.items():
                if expected_val == "RESOLVE":
                    continue  # Skip DB-dependent
                actual_val = getattr(resolved, key)
                if key == "temporal" and expected_val:
                    # Compare as ISO strings
                    actual_val = (
                        actual_val[0].isoformat() if actual_val else None,
                        actual_val[1].isoformat() if actual_val else None,
                    )
                assert actual_val == expected_val, \
                    f"Resolved.{key} mismatch for: {fixture.input}\n  expected: {expected_val}\n  actual: {actual_val}"
```

---

## 8. Success Criteria

1. **Deterministic** — Same input + same `now_utc` → identical token stream, AST, and ResolvedQuery
2. **Compositional** — Modifiers (temporal, segment, speaker, deictic) combine freely
3. **Explicit scope only** — "when we discussed X" → target, not segment_scope
4. **Fail-closed Stage B** — Invalid LLM → FreeText fallback or empty
5. **DST-safe** — Uses `zoneinfo`, tested across transitions
6. **Audit trail** — Tokens, AST, resolved logged at DEBUG
7. **Edit stability** — Case, whitespace, punctuation don't change parse
8. **Golden fixtures pass** — All fixtures in test suite
