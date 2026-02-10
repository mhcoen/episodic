"""Tests for KG ablation evaluation harness."""

import json
import sqlite3
import pytest

from episodic.kg.eval_ablation import (
    score_response,
    build_messages,
    EvalResult,
    evaluate_prompt,
    _create_eval_db,
)
from episodic.kg.schema import ensure_kg_schema


# ---------------------------------------------------------------------------
# score_response
# ---------------------------------------------------------------------------

class TestScoring:
    def test_perfect_score(self):
        answer_key = {
            "required_facts": ["Emma", "MacBook"],
            "expected_answer_contains": ["yes", "64"],
        }
        result = score_response(
            "Yes, Emma's MacBook has 64 GB of RAM.", answer_key
        )
        assert result["required_facts_found"] == 2
        assert result["expected_contains_found"] == 2
        assert result["factual_score"] == 1.0

    def test_partial_score(self):
        answer_key = {
            "required_facts": ["Emma", "MacBook", "MIT"],
            "expected_answer_contains": ["yes"],
        }
        result = score_response(
            "Yes, Emma studies at a university.", answer_key
        )
        assert result["required_facts_found"] == 1  # only "Emma"
        assert result["expected_contains_found"] == 1
        assert result["factual_score"] == 2 / 4

    def test_zero_score(self):
        answer_key = {
            "required_facts": ["Emma"],
            "expected_answer_contains": ["yes"],
        }
        result = score_response("I don't know.", answer_key)
        assert result["factual_score"] == 0.0

    def test_case_insensitive(self):
        answer_key = {
            "required_facts": ["EMMA"],
            "expected_answer_contains": ["YES"],
        }
        result = score_response("emma said yes", answer_key)
        assert result["factual_score"] == 1.0

    def test_empty_answer_key(self):
        result = score_response("anything", {"required_facts": [], "expected_answer_contains": []})
        assert result["factual_score"] == 0.0  # 0/0 = 0


# ---------------------------------------------------------------------------
# build_messages (no setup_context — bare prompt only)
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_bare_prompt_only(self):
        """No KG context → single user message."""
        msgs = build_messages("What color?")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What color?"

    def test_with_kg_context(self):
        """KG context → system + user message."""
        msgs = build_messages("What color?", "Facts: red car")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Facts: red car"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What color?"

    def test_no_setup_context_in_messages(self):
        """Ensure setup_context is NOT included in messages."""
        msgs = build_messages("Hello")
        assert len(msgs) == 1
        # No assistant or prior user messages


# ---------------------------------------------------------------------------
# evaluate_prompt (dry-run only — no LLM calls)
# ---------------------------------------------------------------------------

class TestEvaluatePromptDryRun:
    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        ensure_kg_schema(conn)
        return conn

    def test_dry_run_no_kg(self, db):
        item = {
            "id": "test_01",
            "prompt": "What keyboard?",
            "setup_context": ["I use a Keychron Q1"],
            "answer_key": {
                "required_facts": ["Keychron Q1"],
                "expected_answer_contains": ["Keychron"],
                "category": "baseline_factual",
            },
        }
        result = evaluate_prompt(item, "A", db, "gpt-4o-mini", dry_run=True)
        assert result.prompt_id == "test_01"
        assert result.condition == "A"
        assert result.kg_block_tokens == 0
        assert result.llm_response == ""

    def test_dry_run_with_kg_empty_db(self, db):
        """Condition B on empty DB produces no KG context."""
        item = {
            "id": "test_02",
            "prompt": "unknown entity xyz",
            "setup_context": [],
            "answer_key": {
                "required_facts": [],
                "expected_answer_contains": [],
                "category": "baseline_factual",
            },
        }
        result = evaluate_prompt(item, "B", db, "gpt-4o-mini", dry_run=True)
        assert result.kg_block_tokens == 0
        assert result.kg_context_text == ""


# ---------------------------------------------------------------------------
# _create_eval_db
# ---------------------------------------------------------------------------

class TestEvalDb:
    def test_create_eval_db(self, tmp_path):
        """Eval DB should have nodes table + KG schema."""
        db_path = str(tmp_path / "test_eval.db")
        conn = _create_eval_db(db_path)
        # Check nodes table exists
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'"
        ).fetchone()
        assert row is not None
        # Check KG tables exist
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entities'"
        ).fetchone()
        assert row is not None
        conn.close()
