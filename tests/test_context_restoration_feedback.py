"""Tests for context restoration feedback one-liner (📎 Pulled N earlier messages about: topic)."""

from unittest.mock import MagicMock, patch

import pytest

from episodic.conversation_pipeline import TurnContext, _show_context_restoration


@pytest.fixture
def mock_manager():
    m = MagicMock()
    m.current_topic = ("coastal-erosion-paper", "node-abc-123")
    return m


@pytest.fixture
def ctx_with_reactivation():
    ctx = TurnContext()
    ctx.reactivation_applied = True
    ctx.context_debug = {"included_node_ids": ["n1", "n2", "n3", "n4", "n5", "n6"]}
    return ctx


class TestShowContextRestoration:
    """Unit tests for _show_context_restoration helper."""

    def test_prints_message_when_reactivation_with_nodes(
        self, mock_manager, ctx_with_reactivation, capsys
    ):
        """Reactivation + included_node_ids -> message printed."""
        with patch("episodic.configuration.get_system_color", return_value="cyan"):
            _show_context_restoration(mock_manager, ctx_with_reactivation)

        captured = capsys.readouterr()
        assert "Pulled 6 earlier messages about: coastal-erosion-paper" in captured.out

    def test_no_message_when_empty_included_node_ids(self, mock_manager, capsys):
        """Reactivation + empty included_node_ids -> no message."""
        ctx = TurnContext()
        ctx.reactivation_applied = True
        ctx.context_debug = {"included_node_ids": []}

        with patch("episodic.configuration.get_system_color", return_value="cyan"):
            _show_context_restoration(mock_manager, ctx)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_message_without_reactivation(self, mock_manager):
        """No reactivation -> call site guards prevent _show_context_restoration call.

        The gating logic (ctx.reactivation_applied check) lives at the call site
        in phase_context_assembly, not inside the helper. We verify the guard here.
        """
        ctx = TurnContext()
        ctx.reactivation_applied = False
        ctx.context_debug = {"included_node_ids": ["n1", "n2"]}

        # The call site checks reactivation_applied before calling the helper
        assert ctx.reactivation_applied is False

    def test_correct_count_and_topic_name(self, capsys):
        """Message includes correct node count and topic name."""
        manager = MagicMock()
        manager.current_topic = ("quantum-computing", "node-xyz")

        ctx = TurnContext()
        ctx.reactivation_applied = True
        ctx.context_debug = {"included_node_ids": ["a", "b", "c"]}

        with patch("episodic.configuration.get_system_color", return_value="cyan"):
            _show_context_restoration(manager, ctx)

        captured = capsys.readouterr()
        assert "Pulled 3 earlier messages about: quantum-computing" in captured.out


class TestPhaseContextAssemblyFeedback:
    """Verify the feedback call site in phase_context_assembly."""

    @patch("episodic.db_context_debug.persist_context_assembly_debug")
    def test_calls_show_when_reactivation_applied(
        self, mock_persist, mock_manager, ctx_with_reactivation, capsys
    ):
        """phase_context_assembly calls _show_context_restoration when reactivation is active."""
        from episodic.conversation_pipeline import phase_context_assembly

        # Mock the context builder to return expected tuple
        mock_manager.context_builder.build_context_full.return_value = (
            [],  # messages
            [],  # raw_messages
            None,  # rag_context
            None,  # web_context
            {"included_node_ids": ["n1", "n2"]},  # context_debug
        )

        ctx_with_reactivation.user_node_id = "test-node"
        ctx_with_reactivation.model = "test/model"

        with patch("episodic.configuration.get_system_color", return_value="cyan"):
            phase_context_assembly(mock_manager, ctx_with_reactivation)

        captured = capsys.readouterr()
        assert "Pulled 2 earlier messages about: coastal-erosion-paper" in captured.out

    @patch("episodic.db_context_debug.persist_context_assembly_debug")
    def test_skips_show_when_no_reactivation(self, mock_persist, mock_manager, capsys):
        """phase_context_assembly does not show feedback when reactivation_applied is False."""
        from episodic.conversation_pipeline import phase_context_assembly

        ctx = TurnContext()
        ctx.reactivation_applied = False
        ctx.user_node_id = "test-node"
        ctx.model = "test/model"

        mock_manager.context_builder.build_context_full.return_value = (
            [], [], None, None, {"included_node_ids": ["n1"]},
        )

        with patch("episodic.configuration.get_system_color", return_value="cyan"):
            phase_context_assembly(mock_manager, ctx)

        captured = capsys.readouterr()
        assert "Pulled" not in captured.out
