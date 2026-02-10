"""Tests for KG ablation evaluation harness."""

import json
import sqlite3
import pytest

from episodic.kg.eval_ablation import (
    score_response,
    build_messages,
    EvalResult,
    EvalSummary,
    evaluate_prompt,
    _create_eval_db,
    _check_closure_differentiation,
    _build_closure_analysis,
    compute_derived_relevance,
    compute_oracle_hit,
    _tokenize,
    _jaccard,
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


# ---------------------------------------------------------------------------
# Closure differentiation checks
# ---------------------------------------------------------------------------

class TestClosureChecks:
    def _make_result(self, pid, cond, closure_expected=False,
                     derived=0, rule=""):
        return EvalResult(
            prompt_id=pid, condition=cond, category="multi_hop",
            derived_edges_count=derived,
            closure_expected=closure_expected,
            closure_rule=rule,
        )

    def test_passing_check(self):
        """B=0 derived, C>=1 derived → pass."""
        summary = EvalSummary(results=[
            self._make_result("mh01", "B", True, derived=0, rule="DEVICE_SPEC"),
            self._make_result("mh01", "C", True, derived=2, rule="DEVICE_SPEC"),
        ])
        checks = _check_closure_differentiation(summary, ["B", "C"])
        assert checks["pass_count"] == 1
        assert checks["fail_count"] == 0

    def test_failing_check_b_has_derived(self):
        """B has derived edges → fail (shouldn't happen with max_derived=0)."""
        summary = EvalSummary(results=[
            self._make_result("mh01", "B", True, derived=1, rule="DEVICE_SPEC"),
            self._make_result("mh01", "C", True, derived=1, rule="DEVICE_SPEC"),
        ])
        checks = _check_closure_differentiation(summary, ["B", "C"])
        assert checks["fail_count"] == 1
        assert not checks["items"][0]["b_ok"]

    def test_failing_check_c_no_derived(self):
        """C has no derived edges → fail (closure didn't fire)."""
        summary = EvalSummary(results=[
            self._make_result("mh01", "B", True, derived=0, rule="DEVICE_SPEC"),
            self._make_result("mh01", "C", True, derived=0, rule="DEVICE_SPEC"),
        ])
        checks = _check_closure_differentiation(summary, ["B", "C"])
        assert checks["fail_count"] == 1
        assert not checks["items"][0]["c_ok"]

    def test_non_closure_items_skipped(self):
        """Items without closure_expected are not checked."""
        summary = EvalSummary(results=[
            self._make_result("bl01", "B", False, derived=0),
            self._make_result("bl01", "C", False, derived=0),
        ])
        checks = _check_closure_differentiation(summary, ["B", "C"])
        assert len(checks["items"]) == 0

    def test_dry_run_closure_tracking(self):
        """EvalResult captures closure fields from dataset item."""
        db = sqlite3.connect(":memory:")
        ensure_kg_schema(db)
        item = {
            "id": "mh01",
            "prompt": "Where does my daughter go?",
            "setup_context": ["a", "b", "c", "d"],
            "answer_key": {
                "required_facts": ["Emma"],
                "expected_answer_contains": ["MIT"],
                "category": "multi_hop",
            },
            "closure_expected": True,
            "closure_rule": "KINSHIP_LOCATION",
            "closure_derived": "Emma located_at MIT",
        }
        result = evaluate_prompt(item, "A", db, "gpt-4o-mini", dry_run=True)
        assert result.closure_expected is True
        assert result.closure_rule == "KINSHIP_LOCATION"
        assert result.derived_edges_count == 0  # A has no KG
        db.close()


# ---------------------------------------------------------------------------
# Derived relevance
# ---------------------------------------------------------------------------

class TestDerivedRelevance:
    def test_tokenize_removes_stopwords(self):
        tokens = _tokenize("What does my laptop have?")
        assert "what" not in tokens
        assert "does" not in tokens
        assert "my" not in tokens
        assert "laptop" in tokens

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        assert abs(_jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 0.01

    def test_relevance_with_facts(self):
        """Mock DerivedFact-like objects for relevance scoring."""
        class FakeDerived:
            def __init__(self, s, p, o):
                self.subj_name, self.predicate, self.obj_name = s, p, o
        facts = [
            FakeDerived("MacBook Pro M3 Max", "has", "64GB RAM"),
            FakeDerived("signal chain", "has", "SM7B"),
        ]
        # Prompt shares "MacBook" and "RAM" with the first derived fact
        result = compute_derived_relevance(
            "What RAM size does the MacBook support?", facts
        )
        assert result["max_rel"] > 0
        assert result["top_fact"] != ""
        assert result["mean_rel"] > 0

    def test_relevance_empty_facts(self):
        result = compute_derived_relevance("some prompt", [])
        assert result["max_rel"] == 0.0

    def test_oracle_hit(self):
        class FakeDerived:
            def __init__(self, s, p, o, rule):
                self.subj_name, self.predicate, self.obj_name = s, p, o
                self.rule = rule
        facts = [
            FakeDerived("Emma", "located_at", "MIT", "KINSHIP_LOCATION"),
            FakeDerived("signal chain", "has", "SM7B", "DEVICE_SPEC"),
        ]
        result = compute_oracle_hit(facts, "KINSHIP_LOCATION", ["Emma", "MIT"])
        assert result["oracle_hit"] is True
        assert "Emma" in result["oracle_fact"]

    def test_oracle_miss_wrong_rule(self):
        class FakeDerived:
            def __init__(self, s, p, o, rule):
                self.subj_name, self.predicate, self.obj_name = s, p, o
                self.rule = rule
        facts = [FakeDerived("signal chain", "has", "SM7B", "DEVICE_SPEC")]
        result = compute_oracle_hit(facts, "KINSHIP_LOCATION", ["Emma"])
        assert result["oracle_hit"] is False

    def test_oracle_miss_no_required_facts(self):
        class FakeDerived:
            def __init__(self, s, p, o, rule):
                self.subj_name, self.predicate, self.obj_name = s, p, o
                self.rule = rule
        facts = [FakeDerived("Emma", "located_at", "MIT", "KINSHIP_LOCATION")]
        result = compute_oracle_hit(facts, "KINSHIP_LOCATION", ["Stanford"])
        assert result["oracle_hit"] is False


# ---------------------------------------------------------------------------
# Closure analysis table
# ---------------------------------------------------------------------------

class TestClosureAnalysis:
    def _make_result(self, pid, cond, closure_expected=False,
                     derived=0, rule="", score=0.5,
                     max_rel=0.0, oracle_hit=False):
        return EvalResult(
            prompt_id=pid, condition=cond, category="multi_hop",
            factual_score=score, derived_edges_count=derived,
            closure_expected=closure_expected, closure_rule=rule,
            derived_max_relevance=max_rel, oracle_hit=oracle_hit,
        )

    def test_analysis_table_produced(self):
        summary = EvalSummary(results=[
            self._make_result("mh01", "B", True, derived=0, rule="DEVICE_SPEC", score=0.3),
            self._make_result("mh01", "C", True, derived=2, rule="DEVICE_SPEC",
                              score=0.8, max_rel=0.25, oracle_hit=True),
        ])
        summary.closure_checks = {"items": [], "pass_count": 0, "fail_count": 0}
        lines = _build_closure_analysis(summary, ["A", "B", "C"])
        assert any("Closure Analysis" in l for l in lines)
        assert any("DEVICE_SPEC" in l for l in lines)
        assert any("ALL closure" in l for l in lines)

    def test_no_closure_items_empty(self):
        summary = EvalSummary(results=[
            self._make_result("bl01", "B", False),
            self._make_result("bl01", "C", False),
        ])
        lines = _build_closure_analysis(summary, ["A", "B", "C"])
        assert lines == []

    def test_multiple_rules(self):
        summary = EvalSummary(results=[
            self._make_result("mh01", "B", True, 0, "DEVICE_SPEC", 0.3),
            self._make_result("mh01", "C", True, 2, "DEVICE_SPEC", 0.8, 0.2, True),
            self._make_result("mh02", "B", True, 0, "KINSHIP_LOCATION", 0.2),
            self._make_result("mh02", "C", True, 1, "KINSHIP_LOCATION", 0.6, 0.3, False),
        ])
        lines = _build_closure_analysis(summary, ["A", "B", "C"])
        rule_lines = [l for l in lines if "DEVICE_SPEC" in l or "KINSHIP_LOCATION" in l]
        assert len(rule_lines) == 2
