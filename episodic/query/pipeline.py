"""
MQL Pipeline

Main entry point for the query understanding system.

parse_query(raw_input, conn, now_utc, user_tz) -> ResolvedQuery
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from .lexer import tokenize
from .normalizer import normalize
from .parser import parse
from .resolver import resolve
from .types import AST, ResolvedQuery


def parse_query(
    raw_input: str,
    conn: Optional[sqlite3.Connection] = None,
    now_utc: Optional[datetime] = None,
    user_tz: str = "America/Chicago"
) -> ResolvedQuery:
    """
    Parse raw user input into a ResolvedQuery.

    This is the main entry point for the MQL query understanding system.

    Args:
        raw_input: Raw user input string
        conn: SQLite connection for segment lookup (optional for unit tests)
        now_utc: Current UTC time (timezone-aware). If None, uses current time.
        user_tz: User's timezone (default: America/Chicago)

    Returns:
        ResolvedQuery ready for the retrieval pipeline

    Pipeline stages:
        1. Normalize: Unicode/whitespace normalization
        2. Tokenize: Emit tokens with spans and indices
        3. Parse: Build AST (MQLCommand, DiscussionQuery, or FreeText)
        4. Resolve: Convert AST to ResolvedQuery with temporal/segment resolution
    """
    # Default now_utc to current time if not provided
    if now_utc is None:
        from zoneinfo import ZoneInfo
        now_utc = datetime.now(ZoneInfo("UTC"))

    # Stage 1: Normalize
    s_norm, norm_audit = normalize(raw_input)

    # Stage 2: Tokenize
    lex_result = tokenize(s_norm)

    # Stage 3: Parse
    ast = parse(lex_result, raw_input)

    # Stage 4: Resolve
    resolved = resolve(ast, conn, now_utc, user_tz)

    return resolved


def parse_to_ast(raw_input: str) -> AST:
    """
    Parse raw input to AST only (no resolution).

    Useful for testing and debugging the parser.
    """
    s_norm, _ = normalize(raw_input)
    lex_result = tokenize(s_norm)
    return parse(lex_result, raw_input)


def tokenize_input(raw_input: str):
    """
    Tokenize raw input (normalize + lex).

    Returns (LexResult, NormalizationAudit) tuple.
    Useful for testing and debugging the lexer.
    """
    s_norm, norm_audit = normalize(raw_input)
    lex_result = tokenize(s_norm)
    return lex_result, norm_audit
