"""Tests for episodic.kg.extractor (input assembly and JSON cleaning)."""

import json
import sqlite3
import pytest

from episodic.kg.extractor import clean_llm_json
from episodic.kg.prompt_template import (
    format_extraction_input,
    build_extraction_context,
    EXTRACTION_SYSTEM_PROMPT,
)
from episodic.kg.schema import ensure_kg_schema


@pytest.fixture
def ctx_db():
    """In-memory DB with nodes + KG schema for context tests."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            id TEXT,
            content TEXT,
            role TEXT DEFAULT 'user',
            is_meta_query INTEGER DEFAULT 0
        )
    """)
    # Insert some test nodes
    conn.execute(
        "INSERT INTO nodes VALUES (1, 'uuid1', 'Hello, how are you?', 'user', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (2, 'uuid2', 'I am fine, thanks.', 'assistant', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (3, 'uuid3', 'I use Python for data analysis.', 'user', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (4, 'uuid4', 'Python is great for that.', 'assistant', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (5, 'uuid5', 'I prefer Vim over VS Code.', 'user', 0)"
    )
    ensure_kg_schema(conn)
    conn.commit()
    yield conn
    conn.close()


def test_format_extraction_input():
    """Output is valid JSON with all required keys."""
    result = format_extraction_input(
        node_id=1,
        source_text="Hello world",
        recent_context=["user: Hi", "assistant: Hello"],
        entity_dictionary=[{
            'entity_id': 1, 'entity_type': 'person',
            'canonical_name': '<user>', 'canonical_key': 'user:self',
            'aliases': [],
        }],
        kg_neighborhood=[],
    )
    parsed = json.loads(result)
    assert parsed['node_id'] == 1
    assert parsed['source_text'] == "Hello world"
    assert len(parsed['recent_context']) == 2
    assert len(parsed['entity_dictionary']) == 1
    assert isinstance(parsed['kg_neighborhood'], list)


def test_clean_llm_json_markdown_fences():
    """Markdown fences stripped correctly."""
    raw = '```json\n{"key": "value"}\n```'
    cleaned = clean_llm_json(raw)
    parsed = json.loads(cleaned)
    assert parsed == {"key": "value"}


def test_clean_llm_json_plain():
    """Plain JSON passed through unchanged."""
    raw = '{"key": "value"}'
    cleaned = clean_llm_json(raw)
    assert cleaned == raw


def test_clean_llm_json_with_whitespace():
    """Leading/trailing whitespace stripped."""
    raw = '  \n  {"key": "value"}  \n  '
    cleaned = clean_llm_json(raw)
    parsed = json.loads(cleaned)
    assert parsed == {"key": "value"}


def test_clean_llm_json_bom():
    """BOM character stripped."""
    raw = '\ufeff{"key": "value"}'
    cleaned = clean_llm_json(raw)
    parsed = json.loads(cleaned)
    assert parsed == {"key": "value"}


def test_build_extraction_context_user_node(ctx_db):
    """User node returns full context dict."""
    result = build_extraction_context(3, lookback=2, conn=ctx_db)
    assert result is not None
    assert result['node_id'] == 3
    assert 'Python' in result['source_text']
    assert isinstance(result['recent_context'], list)
    assert isinstance(result['entity_dictionary'], list)


def test_build_extraction_context_assistant_node(ctx_db):
    """Assistant node returns None (Phase 0: user only)."""
    result = build_extraction_context(2, lookback=2, conn=ctx_db)
    assert result is None


def test_build_extraction_context_nonexistent_node(ctx_db):
    """Nonexistent node returns None."""
    result = build_extraction_context(999, lookback=2, conn=ctx_db)
    assert result is None


def test_build_extraction_context_lookback(ctx_db):
    """Recent context includes correct number of preceding turns."""
    result = build_extraction_context(5, lookback=3, conn=ctx_db)
    assert result is not None
    # Should have up to 3 preceding turns
    assert len(result['recent_context']) <= 3
    # Should include both roles
    assert any('user:' in c for c in result['recent_context'])


def test_extraction_system_prompt_content():
    """System prompt contains key directives."""
    assert 'kg_patch_v1' in EXTRACTION_SYSTEM_PROMPT
    assert 'user:self' in EXTRACTION_SYSTEM_PROMPT
    assert 'JSON' in EXTRACTION_SYSTEM_PROMPT
    assert 'trigger' in EXTRACTION_SYSTEM_PROMPT.lower()
