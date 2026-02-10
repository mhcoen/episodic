"""
MQL Query Understanding Types

All dataclasses for the lexer, parser, and resolver pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional, Tuple, Union


class TokenKind(Enum):
    """Token types for the MQL lexer."""

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
    KW_WHAT = auto()      # what
    KW_DID = auto()       # did
    KW_HAVE = auto()      # have, has
    KW_EVER = auto()      # ever
    KW_IN = auto()        # in, within
    KW_ON = auto()        # on
    KW_ABOUT = auto()     # about
    KW_DEICTIC = auto()   # earlier, previous
    KW_TIME = auto()      # time (for "last time" disambiguation)
    KW_DISCOURSE = auto() # before, previously, already (broadness cues, NOT temporal)

    # Default
    WORD = auto()         # Any other word

    # Special
    EOF = auto()
    LEX_ERROR = auto()    # Unclosed quote, invalid character


@dataclass(frozen=True)
class Token:
    """A token from the lexer with span and provenance information."""

    kind: TokenKind
    lexeme: str                 # Original substring from s_norm
    span: Tuple[int, int]       # (start, end) codepoint offsets into s_norm
    normalized: Optional[str]   # Canonical form (for keywords)
    index: int                  # Position in token stream (for provenance)

    def as_word(self) -> Token:
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


@dataclass(frozen=True)
class NormalizationAudit:
    """Audit record for normalization transformation."""

    raw: str
    normalized: str
    changes: Tuple[str, ...]  # Immutable for frozen dataclass


@dataclass
class LexResult:
    """Result from lexer tokenization."""

    tokens: List[Token]
    s_norm: str
    has_error: bool
    error_code: Optional[str]


class Mode(Enum):
    """Query mode."""

    BROWSE = "browse"
    SUMMARIZE = "summarize"
    ANSWER = "answer"


@dataclass(frozen=True)
class SpanInfo:
    """Provenance information for any AST node derived from user text."""

    span: Tuple[int, int]              # [start, end) into s_norm
    source_tokens: Tuple[int, ...]     # Indices into token stream


@dataclass(frozen=True)
class SegmentSpec:
    """Segment specification from parsing."""

    explicit: bool
    query: Optional[str]  # Raw segment query text
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class SpeakerSpec:
    """Speaker specification from parsing."""

    role: str  # "user", "assistant", or "both"
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class TemporalSpec:
    """Temporal specification from parsing."""

    kind: str  # "yesterday", "last_week", "iso_date", etc.
    raw: str
    iso_date: Optional[str] = None  # For ISO dates
    n: Optional[int] = None         # For "N days ago"
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class DeicticSpec:
    """Deictic specification from parsing."""

    kind: str  # "earlier", "previous", "last_time"
    provenance: Optional[SpanInfo] = None


@dataclass(frozen=True)
class TargetSpec:
    """Target specification from parsing."""

    text: str
    spans: Tuple[Tuple[int, int], ...]
    source_tokens: Tuple[int, ...]  # Token indices for full provenance


@dataclass(frozen=True)
class AuditInfo:
    """Audit information for the parsing process."""

    s_raw: str                   # Original input (for reference)
    s_norm: str                  # Normalized input (canonical for spans)
    tokens: Tuple[dict, ...]     # Serialized token stream (immutable)
    rule_path: Tuple[str, ...]   # Parser rules taken
    decisions: Tuple[str, ...]   # Soft-keyword reinterpretations


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

    This makes the "discussion-query -> browse mode, no segment scope" rule
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
        """Canonical serialization."""
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
    """Fallback for unparseable input."""

    text: str
    parse_error: str
    audit: AuditInfo

    def to_dict(self) -> dict:
        """Canonical serialization."""
        return {
            "ast_kind": "FreeText",
            "parse_error": self.parse_error,
            "text": self.text,
        }


# Type alias for AST union
AST = Union[MQLCommand, DiscussionQuery, FreeText]


@dataclass(frozen=True)
class ResolvedQuery:
    """
    Fully resolved query ready for the retrieval pipeline.

    All temporal values are timezone-aware UTC datetimes.
    """

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
    ast_kind: str = "FreeText"                       # "MQLCommand", "DiscussionQuery", or "FreeText"

    def to_dict(self) -> dict:
        """Canonical serialization."""
        return {
            "ast_kind": self.ast_kind,
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


@dataclass
class SegmentResolutionResult:
    """Result of segment resolution with disambiguation information."""

    normalized_query: str
    node_ids: List[str]
    is_ambiguous: bool
    candidates: Optional[List[dict]]  # All matching topics if ambiguous
    audit_notes: List[str]
