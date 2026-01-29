"""
Integration tests for quality evaluation system.

Tests:
- test_resume_moment_loading
- test_quality_eval_runs_all_modes
- test_export_format_valid
"""

import json
import os
import pytest
import sqlite3
import tempfile
from pathlib import Path
from typing import List

# Set test mode
os.environ["EPISODIC_TEST_MODE"] = "1"


def create_test_schema(conn: sqlite3.Connection) -> None:
    """Create minimal database schema for tests."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            role TEXT,
            content TEXT,
            parent_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topic_nodes (
            topic_start_node_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            turn_idx INTEGER NOT NULL,
            role TEXT NOT NULL,
            PRIMARY KEY(topic_start_node_id, node_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topic_working_set (
            topic_start_node_id TEXT PRIMARY KEY,
            topic_name TEXT,
            summary_md TEXT NOT NULL DEFAULT '',
            summary_json TEXT,
            decisions_json TEXT NOT NULL DEFAULT '[]',
            open_loops_json TEXT NOT NULL DEFAULT '[]',
            entities_json TEXT NOT NULL DEFAULT '[]',
            last_summarized_turn_idx INTEGER,
            last_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            summary_version INTEGER NOT NULL DEFAULT 1,
            schema_version INTEGER DEFAULT 1,
            summarizer_model_id TEXT,
            prompt_hash TEXT,
            input_start_turn_idx INTEGER,
            input_end_turn_idx INTEGER,
            input_node_ids_hash TEXT,
            summary_hash TEXT,
            canonicalizer_version INTEGER DEFAULT 1,
            last_summarized_at TIMESTAMP
        )
    """)

    conn.commit()


class TestResumeMomentLoading:
    """Tests for loading resume moments from fixtures."""

    def test_load_resume_moments_returns_list(self):
        """load_resume_moments returns a list of ResumeMoment objects."""
        from episodic.evaluation.resume_moments import (
            load_resume_moments,
            ResumeMoment,
        )

        moments = load_resume_moments()

        assert isinstance(moments, list)
        assert len(moments) > 0
        assert all(isinstance(m, ResumeMoment) for m in moments)

    def test_load_resume_moments_has_required_fields(self):
        """Each moment has all required fields."""
        from episodic.evaluation.resume_moments import load_resume_moments

        moments = load_resume_moments()

        for m in moments:
            assert m.moment_id, f"Missing moment_id"
            assert m.user_query, f"Missing user_query for {m.moment_id}"
            assert m.expected_active_topic, f"Missing expected_active_topic for {m.moment_id}"
            assert m.category, f"Missing category for {m.moment_id}"
            assert m.gap_turns >= 0, f"Invalid gap_turns for {m.moment_id}"

    def test_load_resume_moments_category_filter(self):
        """Category filter returns only matching moments."""
        from episodic.evaluation.resume_moments import load_resume_moments

        short_gap = load_resume_moments(category="short_gap")
        medium_gap = load_resume_moments(category="medium_gap")
        all_moments = load_resume_moments()

        assert len(short_gap) > 0
        assert len(medium_gap) > 0
        assert all(m.category == "short_gap" for m in short_gap)
        assert all(m.category == "medium_gap" for m in medium_gap)
        assert len(short_gap) + len(medium_gap) < len(all_moments)  # Other categories exist

    def test_validate_moments_finds_no_issues(self):
        """Fixtures should pass validation."""
        from episodic.evaluation.resume_moments import (
            load_resume_moments,
            validate_moments,
        )

        moments = load_resume_moments()
        result = validate_moments(moments)

        assert result["valid"], f"Validation failed: {result['issues']}"
        assert result["moment_count"] >= 50, f"Expected 50+ moments, got {result['moment_count']}"

    def test_moments_cover_all_categories(self):
        """Fixtures include all required categories."""
        from episodic.evaluation.resume_moments import load_resume_moments

        moments = load_resume_moments()
        categories = {m.category for m in moments}

        expected_categories = {"short_gap", "medium_gap", "long_gap", "ambiguous", "thin_topic"}
        assert expected_categories == categories, f"Missing categories: {expected_categories - categories}"

    def test_get_moment_by_id(self):
        """Can retrieve a specific moment by ID."""
        from episodic.evaluation.resume_moments import (
            load_resume_moments,
            get_moment_by_id,
        )

        moments = load_resume_moments()
        first_id = moments[0].moment_id

        retrieved = get_moment_by_id(first_id, moments)

        assert retrieved is not None
        assert retrieved.moment_id == first_id

    def test_get_moment_by_id_not_found(self):
        """Returns None for non-existent ID."""
        from episodic.evaluation.resume_moments import get_moment_by_id

        result = get_moment_by_id("nonexistent_id_12345")

        assert result is None

    def test_summarize_moments(self):
        """summarize_moments returns human-readable output."""
        from episodic.evaluation.resume_moments import summarize_moments

        summary = summarize_moments()

        assert "Resume Moments Summary" in summary
        assert "Total moments:" in summary
        assert "short_gap:" in summary


class TestQualityEvalRunsAllModes:
    """Tests for quality evaluation across modes."""

    def test_quality_eval_runs_without_error(self, tmp_path):
        """run_quality_eval completes without error."""
        from episodic.evaluation.quality_eval import run_quality_eval
        from episodic.evaluation.resume_moments import ResumeMoment

        # Create minimal test moment
        moments = [
            ResumeMoment(
                moment_id="test_moment_1",
                user_query="Test query",
                expected_active_topic="test-topic",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test moment",
            )
        ]

        # Create test database
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        create_test_schema(conn)
        conn.close()

        # Patch get_connection to use test db
        import episodic.db_connection as db_mod
        original_get_conn = db_mod.get_connection

        def mock_get_connection():
            return sqlite3.connect(str(db_path))

        db_mod.get_connection = mock_get_connection

        try:
            report = run_quality_eval(moments=moments, call_llm=False)

            assert report.moments_evaluated == 1
            assert "ancestry" in report.modes
            assert "hybrid" in report.modes
            assert "topic_local" in report.modes
        finally:
            db_mod.get_connection = original_get_conn

    def test_quality_eval_returns_results_for_all_modes(self, tmp_path):
        """Each moment has results for all tested modes."""
        from episodic.evaluation.quality_eval import run_quality_eval
        from episodic.evaluation.resume_moments import ResumeMoment

        moments = [
            ResumeMoment(
                moment_id="test_mode_1",
                user_query="What was that Python fix?",
                expected_active_topic="python-debugging",
                gap_turns=6,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test",
            )
        ]

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        create_test_schema(conn)
        conn.close()

        import episodic.db_connection as db_mod
        original_get_conn = db_mod.get_connection

        def mock_get_connection():
            return sqlite3.connect(str(db_path))

        db_mod.get_connection = mock_get_connection

        try:
            modes = ["ancestry", "hybrid", "topic_local"]
            report = run_quality_eval(moments=moments, modes=modes, call_llm=False)

            for result in report.results:
                for mode in modes:
                    assert mode in result.mode_results, f"Missing {mode} in {result.moment_id}"
                    mr = result.mode_results[mode]
                    assert mr.mode == mode
        finally:
            db_mod.get_connection = original_get_conn

    def test_quality_eval_computes_summary_stats(self, tmp_path):
        """Report includes summary statistics by mode and category."""
        from episodic.evaluation.quality_eval import run_quality_eval
        from episodic.evaluation.resume_moments import ResumeMoment

        moments = [
            ResumeMoment(
                moment_id="test_summary_1",
                user_query="Test",
                expected_active_topic="topic-1",
                gap_turns=5,
                category="short_gap",
                cross_topic_import_expected=False,
                notes="Test 1",
            ),
            ResumeMoment(
                moment_id="test_summary_2",
                user_query="Test 2",
                expected_active_topic="topic-2",
                gap_turns=30,
                category="medium_gap",
                cross_topic_import_expected=False,
                notes="Test 2",
            ),
        ]

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        create_test_schema(conn)
        conn.close()

        import episodic.db_connection as db_mod
        original_get_conn = db_mod.get_connection

        def mock_get_connection():
            return sqlite3.connect(str(db_path))

        db_mod.get_connection = mock_get_connection

        try:
            report = run_quality_eval(moments=moments, call_llm=False)

            assert "by_mode" in report.summary
            assert "by_category" in report.summary
            assert "ancestry" in report.summary["by_mode"]
            assert "short_gap" in report.summary["by_category"]
            assert "medium_gap" in report.summary["by_category"]
        finally:
            db_mod.get_connection = original_get_conn


class TestExportFormatValid:
    """Tests for export functionality."""

    def test_export_markdown_format(self, tmp_path):
        """export_for_human_review creates valid markdown."""
        from episodic.evaluation.quality_eval import (
            QualityEvalReport,
            MomentEvalResult,
            ModeResult,
            export_for_human_review,
        )

        # Create minimal report
        report = QualityEvalReport(
            timestamp="2026-01-28T00:00:00",
            config={"modes": ["ancestry", "topic_local"]},
            moments_evaluated=1,
            modes=["ancestry", "topic_local"],
            results=[
                MomentEvalResult(
                    moment_id="test_export_1",
                    user_query="What was that fix?",
                    expected_active_topic="python-debugging",
                    category="short_gap",
                    mode_results={
                        "ancestry": ModeResult(
                            mode="ancestry",
                            prompt_fingerprint="abc123",
                            included_node_ids=["node1", "node2"],
                            contamination_count=1,
                            token_breakdown={"total_tokens": 500},
                            total_tokens=500,
                            assembly_ms=10.5,
                            response="Test response for ancestry",
                        ),
                        "topic_local": ModeResult(
                            mode="topic_local",
                            prompt_fingerprint="def456",
                            included_node_ids=["node1"],
                            contamination_count=0,
                            token_breakdown={"total_tokens": 300},
                            total_tokens=300,
                            assembly_ms=8.2,
                            response="Test response for topic_local",
                        ),
                    },
                )
            ],
            summary={
                "by_mode": {
                    "ancestry": {"contamination_rate": 1.0, "avg_tokens": 500, "avg_assembly_ms": 10.5},
                    "topic_local": {"contamination_rate": 0.0, "avg_tokens": 300, "avg_assembly_ms": 8.2},
                },
                "by_category": {
                    "short_gap": {"count": 1, "modes": {"ancestry": {"total_contamination": 1}, "topic_local": {"total_contamination": 0}}},
                },
            },
        )

        output_path = tmp_path / "review.md"
        result_path = export_for_human_review(report, output_path, format="markdown")

        assert result_path == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "# Quality Evaluation Review" in content
        assert "test_export_1" in content
        assert "What was that fix?" in content
        assert "python-debugging" in content
        assert "Contamination: 1 foreign nodes" in content
        assert "Contamination: 0 foreign nodes" in content
        assert "Scoring (fill in):" in content
        assert "[ ] Stays on correct topic" in content

    def test_export_csv_format(self, tmp_path):
        """export_for_human_review creates valid CSV."""
        from episodic.evaluation.quality_eval import (
            QualityEvalReport,
            MomentEvalResult,
            ModeResult,
            export_for_human_review,
        )

        report = QualityEvalReport(
            timestamp="2026-01-28T00:00:00",
            config={},
            moments_evaluated=1,
            modes=["ancestry"],
            results=[
                MomentEvalResult(
                    moment_id="csv_test_1",
                    user_query="Test query",
                    expected_active_topic="test-topic",
                    category="short_gap",
                    mode_results={
                        "ancestry": ModeResult(
                            mode="ancestry",
                            prompt_fingerprint="abc",
                            included_node_ids=[],
                            contamination_count=0,
                            token_breakdown={},
                            total_tokens=100,
                            assembly_ms=5.0,
                            response="Response text",
                        ),
                    },
                )
            ],
            summary={},
        )

        output_path = tmp_path / "review.csv"
        result_path = export_for_human_review(report, output_path, format="csv")

        assert result_path == output_path
        assert output_path.exists()

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Check header
        assert "moment_id" in lines[0]
        assert "category" in lines[0]
        assert "mode" in lines[0]
        assert "contamination" in lines[0]

        # Check data row
        assert "csv_test_1" in lines[1]
        assert "short_gap" in lines[1]
        assert "ancestry" in lines[1]

    def test_save_report_json(self, tmp_path):
        """save_report creates valid JSON."""
        from episodic.evaluation.quality_eval import (
            QualityEvalReport,
            MomentEvalResult,
            ModeResult,
            save_report,
        )

        report = QualityEvalReport(
            timestamp="2026-01-28T00:00:00",
            config={"test": True},
            moments_evaluated=1,
            modes=["ancestry"],
            results=[
                MomentEvalResult(
                    moment_id="json_test_1",
                    user_query="Query",
                    expected_active_topic="topic",
                    category="short_gap",
                    mode_results={
                        "ancestry": ModeResult(
                            mode="ancestry",
                            prompt_fingerprint="fp",
                            included_node_ids=["n1"],
                            contamination_count=0,
                            token_breakdown={"summary_tokens": 10},
                            total_tokens=50,
                            assembly_ms=2.0,
                            response="",
                        ),
                    },
                )
            ],
            summary={"by_mode": {}},
        )

        output_path = tmp_path / "report.json"
        result_path = save_report(report, output_path)

        assert result_path == output_path
        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)

        assert data["moments_evaluated"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["moment_id"] == "json_test_1"


class TestPromptFingerprint:
    """Tests for prompt fingerprinting."""

    def test_compute_prompt_fingerprint_deterministic(self):
        """Same messages produce same fingerprint."""
        from episodic.evaluation.quality_eval import compute_prompt_fingerprint

        messages = [{"role": "user", "content": "Hello"}]

        fp1 = compute_prompt_fingerprint(messages)
        fp2 = compute_prompt_fingerprint(messages)

        assert fp1 == fp2
        assert len(fp1) == 16  # SHA256 truncated to 16 chars

    def test_compute_prompt_fingerprint_different_for_different_messages(self):
        """Different messages produce different fingerprints."""
        from episodic.evaluation.quality_eval import compute_prompt_fingerprint

        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "World"}]

        fp1 = compute_prompt_fingerprint(messages1)
        fp2 = compute_prompt_fingerprint(messages2)

        assert fp1 != fp2
