"""
MQL Lexer

Tokenizer that emits tokens with spans into s_norm and token indices.

CRITICAL: Unknown characters emit LEX_ERROR, never silently skip.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .types import LexResult, Token, TokenKind


# Keyword normalization map: lowercase -> (TokenKind, normalized_value)
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
    "spoke": (TokenKind.KW_DISCUSS, "spoke"),
    "speak": (TokenKind.KW_DISCUSS, "spoke"),
    "spoken": (TokenKind.KW_DISCUSS, "spoke"),
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
    "what": (TokenKind.KW_WHAT, "what"),
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


# ISO date pattern: strict YYYY-MM-DD
ISO_DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')

# WORD pattern: allows internal hyphens, underscores, apostrophes
# Single alphanumeric: [A-Za-z0-9]
# Multi-char with internal punctuation: [A-Za-z0-9]([A-Za-z0-9_'-]*[A-Za-z0-9])?
WORD_PATTERN = re.compile(r"[A-Za-z0-9]([A-Za-z0-9_'\-]*[A-Za-z0-9])?|[A-Za-z0-9]")

# Punctuation characters that become distinct tokens
PUNCT_TOKENS = {
    ':': TokenKind.COLON,
    ',': TokenKind.COMMA,
    '?': TokenKind.QUESTION,
    '(': TokenKind.LPAREN,
    ')': TokenKind.RPAREN,
}


class Lexer:
    """
    MQL tokenizer that produces a stream of tokens with spans and indices.

    CRITICAL: Unknown characters emit LEX_ERROR, never silently skip.
    """

    def __init__(self, s_norm: str):
        self.s_norm = s_norm
        self.pos = 0
        self.tokens: List[Token] = []
        self.has_error = False
        self.error_code: Optional[str] = None
        self._token_index = 0

    def _next_index(self) -> int:
        """Get next token index and increment counter."""
        idx = self._token_index
        self._token_index += 1
        return idx

    def tokenize(self) -> LexResult:
        """
        Tokenize the input string.

        Returns LexResult with token stream, error flag, and error code.
        """
        while self.pos < len(self.s_norm):
            self._skip_whitespace()
            if self.pos >= len(self.s_norm):
                break
            self._scan_token()

        # Append EOF token
        self.tokens.append(Token(
            TokenKind.EOF, "", (self.pos, self.pos), None, self._next_index()
        ))

        return LexResult(
            tokens=self.tokens,
            s_norm=self.s_norm,
            has_error=self.has_error,
            error_code=self.error_code
        )

    def _skip_whitespace(self):
        """Skip over space characters."""
        while self.pos < len(self.s_norm) and self.s_norm[self.pos] == ' ':
            self.pos += 1

    def _scan_token(self):
        """Scan and emit the next token."""
        char = self.s_norm[self.pos]

        # Quoted string
        if char in ('"', "'"):
            tok = self._scan_quoted(char)
            self.tokens.append(tok)
            if tok.kind == TokenKind.LEX_ERROR:
                self.has_error = True
                self.error_code = tok.normalized
            return

        # Known punctuation (distinct tokens)
        if char in PUNCT_TOKENS:
            start = self.pos
            self.pos += 1
            self.tokens.append(Token(
                PUNCT_TOKENS[char], char, (start, self.pos), char, self._next_index()
            ))
            return

        # Standalone dash (not internal to word)
        if char == '-':
            # Check if this is standalone (not followed by alphanumeric that would start a word)
            # A dash at position 0 or after whitespace is standalone if not followed by alnum
            if self.pos + 1 >= len(self.s_norm) or not self.s_norm[self.pos + 1].isalnum():
                start = self.pos
                self.pos += 1
                self.tokens.append(Token(
                    TokenKind.DASH, char, (start, self.pos), char, self._next_index()
                ))
                return
            else:
                # Dash followed by alphanumeric - this is LEX_ERROR since words can't start with dash
                start = self.pos
                self.pos += 1
                self.tokens.append(Token(
                    TokenKind.LEX_ERROR, char, (start, self.pos), "leading_dash", self._next_index()
                ))
                self.has_error = True
                if not self.error_code:
                    self.error_code = "leading_dash"
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

    def _scan_quoted(self, quote_char: str) -> Token:
        """Scan a quoted string."""
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

    def _try_scan_iso_date(self) -> Optional[Token]:
        """Try to scan an ISO date (YYYY-MM-DD). Returns None if not a date."""
        match = ISO_DATE_PATTERN.match(self.s_norm, self.pos)
        if match:
            lexeme = match.group(0)
            start = match.start()
            end = match.end()
            self.pos = end
            return Token(TokenKind.ISO_DATE, lexeme, (start, end), lexeme, self._next_index())
        return None

    def _is_word_start(self, c: str) -> bool:
        """Check if character can start a word (alphanumeric only)."""
        return c.isalnum()

    def _scan_word(self) -> Token:
        """Scan a word token, checking for keywords."""
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

    def _scan_number(self):
        """Scan a number token (sequence of digits)."""
        start = self.pos
        while self.pos < len(self.s_norm) and self.s_norm[self.pos].isdigit():
            self.pos += 1
        lexeme = self.s_norm[start:self.pos]
        self.tokens.append(Token(TokenKind.NUMBER, lexeme, (start, self.pos), lexeme, self._next_index()))


def tokenize(s_norm: str) -> LexResult:
    """Convenience function to tokenize a normalized string."""
    return Lexer(s_norm).tokenize()
