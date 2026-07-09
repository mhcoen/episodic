"""
MQL Parser

Recursive descent parser producing MQLCommand, DiscussionQuery, or FreeText.

INVARIANT: Any LEX_ERROR forces FreeText - no partial parsing allowed.
"""

from __future__ import annotations

from typing import List, Optional

from .types import (
    AST,
    AuditInfo,
    DeicticSpec,
    DiscussionQuery,
    FreeText,
    LexResult,
    Mode,
    MQLCommand,
    SegmentSpec,
    SpanInfo,
    SpeakerSpec,
    TargetSpec,
    TemporalSpec,
    Token,
    TokenKind,
)


from episodic.query.discussion_parser import _DiscussionQueryParserMixin


class ParseError(Exception):
    """Parsing error that triggers FreeText fallback."""
    pass


class Parser(_DiscussionQueryParserMixin):
    """
    MQL recursive descent parser.

    Produces one of:
    - MQLCommand: Standard structured command
    - DiscussionQuery: Discussion-query forms ("when we discussed X", etc.)
    - FreeText: Fallback for unparseable input

    INVARIANT: Any LEX_ERROR forces FreeText.
    """

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

    def parse(self) -> AST:
        """
        Parse the token stream into an AST.

        Returns MQLCommand, DiscussionQuery, or FreeText.
        """
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

    # --- Discussion query forms ---


    # --- Standard MQL parsing ---

    def _parse_mode_phrase(self):
        """Parse optional mode prefix (browse, summarize, answer)."""
        self.rule_path.append("mode_phrase")

        if self._at(TokenKind.KW_MODE):
            tok = self._match(TokenKind.KW_MODE)
            self.mode = Mode(tok.normalized)
            self.rule_path.append(f"explicit_mode:{self.mode.value}")

    def _parse_segment_modifier(self):
        """
        Parse segment modifiers.

        Accepted forms:
        - topic: <query>
        - segment: <query>
        - in topic <query>
        - in topic: <query>
        - in segment <query>
        """
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
        if self._at_normalized("in"):
            self._match_normalized("in")
            if self._at(TokenKind.KW_SEGMENT):
                self._match(TokenKind.KW_SEGMENT)
                self._match(TokenKind.COLON)  # optional colon after "in topic"
                query, provenance = self._parse_segment_query()
                if query:
                    self.segment = SegmentSpec(explicit=True, query=query, provenance=provenance)
                    self.rule_path.append(f"explicit_segment_in:{query}")
                    return
            self._restore(saved)

    def _parse_segment_query(self) -> tuple[Optional[str], Optional[SpanInfo]]:
        """
        Parse segment query: QUOTED or words until modifier boundary.

        NOTE: KW_SPEAKER should NOT truncate segment names - words like "we", "our"
        are reinterpreted as WORD in segment names (e.g., "our-research", "we research").
        """
        if self._at(TokenKind.QUOTED):
            tok = self._match(TokenKind.QUOTED)
            return tok.normalized, SpanInfo(tok.span, (tok.index,))

        words = []
        tokens_used = []
        start_span = None
        end_span = None

        for _ in range(10):  # Max 10 words for segment names
            tok = self._peek()
            # Stop at modifier boundaries - but NOT KW_SPEAKER, which should be
            # reinterpreted as WORD in segment names (e.g., "we research", "our-project")
            if tok.kind in (TokenKind.KW_TIME_REL, TokenKind.KW_DEICTIC,
                           TokenKind.KW_TIME, TokenKind.KW_DISCOURSE, TokenKind.ISO_DATE,
                           TokenKind.COLON, TokenKind.EOF, TokenKind.QUESTION, TokenKind.COMMA):
                break
            # Stop at temporal-like normalized values
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

    def _parse_speaker_modifier(self):
        """Parse speaker modifiers like "my messages" / "your messages"."""
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
        """Parse temporal modifiers."""
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
        """Parse deictic modifiers (earlier, previous)."""
        if self._at(TokenKind.KW_DEICTIC):
            tok = self._match(TokenKind.KW_DEICTIC)
            self.deictic = DeicticSpec(
                kind=tok.normalized,
                provenance=SpanInfo(tok.span, (tok.index,))
            )
            self.rule_path.append(f"deictic:{tok.normalized}")

    def _parse_target(self):
        """Parse target text."""
        self.rule_path.append("target")

        # Skip optional "about"
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
        """Skip trailing question marks and commas."""
        while self._at(TokenKind.QUESTION, TokenKind.COMMA):
            self.pos += 1

    def _build_command(self) -> MQLCommand:
        """Build MQLCommand AST node."""
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
                tokens=tuple(t.to_dict() for t in self.tokens),
                rule_path=tuple(self.rule_path),
                decisions=tuple(self.decisions)
            )
        )

    def _make_freetext(self, error: str) -> FreeText:
        """Build FreeText AST node for fallback."""
        return FreeText(
            text=self.s_norm,
            parse_error=error,
            audit=AuditInfo(
                s_raw=self.s_raw,
                s_norm=self.s_norm,
                tokens=tuple(t.to_dict() for t in self.tokens),
                rule_path=tuple(self.rule_path),
                decisions=tuple(self.decisions)
            )
        )

    # --- Token helpers ---

    def _peek(self, offset: int = 0) -> Token:
        """Look ahead at a token without consuming it."""
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def _at(self, *kinds: TokenKind) -> bool:
        """Check if current token matches any of the given kinds."""
        return self._peek().kind in kinds

    def _at_normalized(self, *values: str) -> bool:
        """Check if current token's normalized value matches any of the given values."""
        tok = self._peek()
        return tok.normalized in values

    def _match(self, *kinds: TokenKind) -> Optional[Token]:
        """Consume token if it matches any of the given kinds."""
        if self._at(*kinds):
            tok = self._peek()
            self.pos += 1
            return tok
        return None

    def _match_normalized(self, *values: str) -> Optional[Token]:
        """Consume token if its normalized value matches."""
        if self._at_normalized(*values):
            tok = self._peek()
            self.pos += 1
            return tok
        return None

    def _at_end(self) -> bool:
        """Check if at end of token stream."""
        return self._at(TokenKind.EOF)

    def _save(self) -> int:
        """Save current position for backtracking."""
        return self.pos

    def _restore(self, pos: int):
        """Restore position for backtracking."""
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


def parse(lex_result: LexResult, s_raw: str) -> AST:
    """Convenience function to parse a lex result."""
    return Parser(lex_result, s_raw).parse()
