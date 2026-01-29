"""
MQL Query Understanding Module

Provides the lexer -> parser -> resolver pipeline for transforming
user input into structured ResolvedQuery objects.

Main entry point: parse_query(raw_input, conn, now_utc, user_tz)
"""

from .lexer import Lexer, tokenize, KEYWORD_MAP
from .normalizer import normalize, PUNCT_MAP
from .parser import Parser, parse, ParseError
from .pipeline import parse_query, parse_to_ast, tokenize_input
from .resolver import Resolver, resolve, resolve_temporal, resolve_segment
from .classifier import classify_and_extract_intent, classify_freetext, extract_memory_intent, ClassificationResult
from .retrieval import (
    QueryExecutor,
    RetrievedNode,
    RetrievalResult,
    execute_query,
    format_retrieval_for_context,
)
from .types import (
    # Enums
    TokenKind,
    Mode,
    # Token types
    Token,
    LexResult,
    NormalizationAudit,
    # AST types
    SpanInfo,
    SegmentSpec,
    SpeakerSpec,
    TemporalSpec,
    DeicticSpec,
    TargetSpec,
    AuditInfo,
    MQLCommand,
    DiscussionQuery,
    FreeText,
    AST,
    # Resolution types
    ResolvedQuery,
    SegmentResolutionResult,
)

__all__ = [
    # Main entry point
    "parse_query",
    "parse_to_ast",
    "tokenize_input",
    # Normalizer
    "normalize",
    "PUNCT_MAP",
    # Lexer
    "Lexer",
    "tokenize",
    "KEYWORD_MAP",
    # Parser
    "Parser",
    "parse",
    "ParseError",
    # Resolver
    "Resolver",
    "resolve",
    "resolve_temporal",
    "resolve_segment",
    # Types - Enums
    "TokenKind",
    "Mode",
    # Types - Token
    "Token",
    "LexResult",
    "NormalizationAudit",
    # Types - AST
    "SpanInfo",
    "SegmentSpec",
    "SpeakerSpec",
    "TemporalSpec",
    "DeicticSpec",
    "TargetSpec",
    "AuditInfo",
    "MQLCommand",
    "DiscussionQuery",
    "FreeText",
    "AST",
    # Types - Resolution
    "ResolvedQuery",
    "SegmentResolutionResult",
    # Classifier
    "classify_and_extract_intent",
    "classify_freetext",  # Deprecated
    "extract_memory_intent",  # Deprecated
    "ClassificationResult",
    # Retrieval
    "QueryExecutor",
    "RetrievedNode",
    "RetrievalResult",
    "execute_query",
    "format_retrieval_for_context",
]
