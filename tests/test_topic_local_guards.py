"""
Tests for topic_local guard protections.

Ensures test-only flags cannot be enabled in production.
"""

import os
import subprocess

import pytest

from episodic.config import config


class TestForceNoRecencyGuard:
    """Tests for force_no_recency guard."""

    def test_force_no_recency_blocked_in_production(self):
        """Ensure force_no_recency cannot be enabled in production."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.db_connection import get_connection

        # Save original values
        original_env = os.environ.get("EPISODIC_TEST_MODE")
        original_debug = config.get("debug")

        try:
            # Ensure not in test/debug mode
            os.environ.pop("EPISODIC_TEST_MODE", None)
            config.set("debug", False)

            strategy = TopicLocalStrategy()

            with pytest.raises(ValueError, match="test/debug mode"):
                with get_connection() as conn:
                    strategy.assemble(
                        user_turn_text="test",
                        user_node_id=None,
                        active_topic_start_node_id="test_topic",
                        user_embedding=None,
                        token_budget=4000,
                        conn=conn,
                        chroma_collection=None,
                        force_no_recency=True,  # Should raise
                    )
        finally:
            # Restore original values
            if original_env:
                os.environ["EPISODIC_TEST_MODE"] = original_env
            config.set("debug", original_debug if original_debug is not None else False)

    def test_force_no_recency_allowed_in_test_mode(self):
        """Ensure force_no_recency works when EPISODIC_TEST_MODE is set."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.db_connection import get_connection

        # Save original values
        original_env = os.environ.get("EPISODIC_TEST_MODE")
        original_debug = config.get("debug")

        try:
            # Set test mode
            os.environ["EPISODIC_TEST_MODE"] = "1"
            config.set("debug", False)

            strategy = TopicLocalStrategy()

            # Should not raise - test mode allows force_no_recency
            with get_connection() as conn:
                result = strategy.assemble(
                    user_turn_text="test",
                    user_node_id=None,
                    active_topic_start_node_id=None,  # No topic = empty result
                    user_embedding=None,
                    token_budget=4000,
                    conn=conn,
                    chroma_collection=None,
                    force_no_recency=True,
                )
                # Should complete without error
                assert result is not None
        finally:
            # Restore original values
            if original_env:
                os.environ["EPISODIC_TEST_MODE"] = original_env
            else:
                os.environ.pop("EPISODIC_TEST_MODE", None)
            config.set("debug", original_debug if original_debug is not None else False)

    def test_force_no_recency_allowed_in_debug_mode(self):
        """Ensure force_no_recency works when debug=True."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.db_connection import get_connection

        # Save original values
        original_env = os.environ.get("EPISODIC_TEST_MODE")
        original_debug = config.get("debug")

        try:
            # Set debug mode
            os.environ.pop("EPISODIC_TEST_MODE", None)
            config.set("debug", True)

            strategy = TopicLocalStrategy()

            # Should not raise - debug mode allows force_no_recency
            with get_connection() as conn:
                result = strategy.assemble(
                    user_turn_text="test",
                    user_node_id=None,
                    active_topic_start_node_id=None,  # No topic = empty result
                    user_embedding=None,
                    token_budget=4000,
                    conn=conn,
                    chroma_collection=None,
                    force_no_recency=True,
                )
                # Should complete without error
                assert result is not None
        finally:
            # Restore original values
            if original_env:
                os.environ["EPISODIC_TEST_MODE"] = original_env
            config.set("debug", original_debug if original_debug is not None else False)

    def test_production_assembly_never_sets_force_no_recency(self):
        """Assert the production code path never passes force_no_recency=True."""
        # Grep the codebase for force_no_recency=True outside of tests
        # -I skips binary files, -r is recursive
        result = subprocess.run(
            ["grep", "-r", "-I", "force_no_recency=True", "episodic/"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Filter out maintenance/verify_phase3.py which is a verification script
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        production_lines = [
            line for line in lines
            if line and "verify_phase3.py" not in line
        ]

        # Should find nothing (only tests and verification scripts should set it)
        assert not production_lines, (
            f"Production code sets force_no_recency=True:\n" + "\n".join(production_lines)
        )
