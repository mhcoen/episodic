"""Discussion-query grammar rules for the MQL Parser.

Mixin split out of parser.py; Parser inherits it, so these methods run on the
Parser instance and call the low-level primitives (self._peek/_match/_save/...)
that remain in Parser.
"""

from __future__ import annotations

from typing import List, Optional

from .types import (
    AST, AuditInfo, DeicticSpec, DiscussionQuery, FreeText, LexResult, Mode,
    MQLCommand, SegmentSpec, SpanInfo, SpeakerSpec, TargetSpec, TemporalSpec,
    Token, TokenKind,
)


class _DiscussionQueryParserMixin:
    """Parsing of 'what/when/have/did we discuss X' query forms."""

    def _try_discussion_query(self) -> Optional[DiscussionQuery]:
        """
        Try to match discussion-query forms. Returns DiscussionQuery or None.

        Forms:
        - when/where (did)? (we|I|you)? discuss/talked/mentioned (about)? <X> (before|previously)?
        - what did (we|I|you) discuss/talked/mentioned (about)? <X> <temporal>?
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

        # Try "what" forms ("what did we discuss yesterday")
        if self._at(TokenKind.KW_WHAT):
            result = self._parse_what_discussion_query()
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
        """
        Parse: when/where discussion query forms.

        Patterns:
        - when (did)? (we|I|you)? discuss/talked/mentioned (about)? <X>
        - when was the last time (we|I|you) discuss/talked/mentioned <X>
        - when did (we|I|you) last discuss/talked/mentioned <X>
        """
        when_tok = self._match(TokenKind.KW_WHEN)  # consume when/where

        # Check for "when was the last time" pattern
        if self._at_normalized("was"):
            saved_after_when = self._save()
            self._match_normalized("was")
            self._match_normalized("the")  # optional "the"
            if self._at_normalized("last") and self._peek(1).kind == TokenKind.KW_TIME:
                last_tok = self._match_normalized("last")
                time_tok = self._match(TokenKind.KW_TIME)
                self.deictic = DeicticSpec(
                    kind="last_time",
                    provenance=SpanInfo((last_tok.span[0], time_tok.span[1]), (last_tok.index, time_tok.index))
                )

                # Now expect speaker + discuss
                speaker_tok = self._match(TokenKind.KW_SPEAKER)
                if speaker_tok:
                    self.speaker = SpeakerSpec(
                        role=speaker_tok.normalized,
                        provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
                    )

                if self._at(TokenKind.KW_DISCUSS):
                    self._match(TokenKind.KW_DISCUSS)
                    self._match(TokenKind.KW_ABOUT)  # optional about
                    self._parse_discussion_target_and_modifiers()
                    return self._build_discussion_query("when_last_time")

            # "was" but not "was the last time" - restore and try standard path
            self._restore(saved_after_when)

        # Standard path: when (did)? (speaker)? discuss
        self._match(TokenKind.KW_DID)   # optional did

        speaker_tok = self._match(TokenKind.KW_SPEAKER)  # optional speaker
        if speaker_tok:
            self.speaker = SpeakerSpec(
                role=speaker_tok.normalized,
                provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
            )

        # Check for "last" before discuss (e.g., "when did we last discuss")
        if self._at_normalized("last"):
            last_tok = self._match_normalized("last")
            # Check if next is "time" for deictic, or discuss verb
            if self._peek().kind == TokenKind.KW_TIME:
                time_tok = self._match(TokenKind.KW_TIME)
                self.deictic = DeicticSpec(
                    kind="last_time",
                    provenance=SpanInfo((last_tok.span[0], time_tok.span[1]), (last_tok.index, time_tok.index))
                )
            elif self._at(TokenKind.KW_DISCUSS):
                # "when did we last discuss" - deictic implied
                self.deictic = DeicticSpec(
                    kind="last_time",
                    provenance=SpanInfo(last_tok.span, (last_tok.index,))
                )

        # Must have discussion verb to be a discussion query
        if not self._at(TokenKind.KW_DISCUSS):
            return None

        self._match(TokenKind.KW_DISCUSS)
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

        # BUG FIX: Must assign speaker to self.speaker
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

        # BUG FIX: Must assign speaker to self.speaker
        speaker = SpeakerSpec(
            role=speaker_tok.normalized,
            provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
        )
        self.speaker = speaker

        if self._match(TokenKind.KW_EVER):
            self.has_broadness_cue = True

        if not self._match(TokenKind.KW_DISCUSS):
            return None

        self._match(TokenKind.KW_ABOUT)

        self._parse_discussion_target_and_modifiers()

        return self._build_discussion_query("did_speaker")

    def _parse_what_discussion_query(self) -> Optional[DiscussionQuery]:
        """
        Parse: what did/do (we|I|you) discuss/talk about <X>? <temporal>?

        Patterns:
        - what did we discuss yesterday
        - what did I say about databases
        - what do you know about X
        - what have we talked about
        """
        self._match(TokenKind.KW_WHAT)  # consume what

        # "what did" or "what do" or "what have"
        has_did = self._match(TokenKind.KW_DID)
        has_have = self._match(TokenKind.KW_HAVE) if not has_did else None
        has_do = self._match_normalized("do") if not has_did and not has_have else None

        # If none of these, this isn't a discussion query form
        if not has_did and not has_have and not has_do:
            return None

        # Optional speaker
        speaker_tok = self._match(TokenKind.KW_SPEAKER)
        if speaker_tok:
            self.speaker = SpeakerSpec(
                role=speaker_tok.normalized,
                provenance=SpanInfo(speaker_tok.span, (speaker_tok.index,))
            )

        # "ever" is a broadness cue
        if self._match(TokenKind.KW_EVER):
            self.has_broadness_cue = True

        # Must have a discussion verb to be a discussion query
        if not self._match(TokenKind.KW_DISCUSS):
            return None

        self._match(TokenKind.KW_ABOUT)  # optional about

        # Parse target and trailing modifiers (including temporal)
        self._parse_discussion_target_and_modifiers()

        return self._build_discussion_query("what_we")

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
                next_tok = self._peek(1)
                if next_tok.normalized in ("week", "month", "year", "days", "day"):
                    break
                if next_tok.kind == TokenKind.NUMBER:
                    break
                # Also check for "last time" (deictic, not target)
                if next_tok.kind == TokenKind.KW_TIME:
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

        # Check for trailing deictic "last time"
        if self._at_normalized("last") and self._peek(1).kind == TokenKind.KW_TIME:
            last_tok = self._match_normalized("last")
            time_tok = self._match(TokenKind.KW_TIME)
            self.deictic = DeicticSpec(
                kind="last_time",
                provenance=SpanInfo((last_tok.span[0], time_tok.span[1]), (last_tok.index, time_tok.index))
            )

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
                tokens=tuple(t.to_dict() for t in self.tokens),
                rule_path=tuple(self.rule_path + [f"discussion_query:{form}"]),
                decisions=tuple(self.decisions)
            )
        )
