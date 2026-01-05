"""
Smoke tests for major CLI commands.

Purpose: Catch regressions where a command fails immediately on basic usage.
These tests verify that commands don't crash, not that they produce perfect output.

Each test:
- Invokes a command with minimal valid input
- Asserts no exception raised
- Asserts some output produced (captures stdout)
- Does NOT assert on exact output content (too brittle)
"""

import pytest
import sys
from io import StringIO
from unittest.mock import patch, MagicMock


class TestCommandSmoke:
    """Smoke tests for CLI commands - verify they don't crash on basic invocation."""

    @pytest.fixture(autouse=True)
    def setup_db(self, isolated_config, integration_db):
        """Ensure clean database and config for each test."""
        # isolated_config and integration_db fixtures from conftest.py handle setup
        pass

    @pytest.fixture
    def capture_output(self):
        """Fixture to capture typer/print output."""
        captured = StringIO()
        with patch('sys.stdout', captured):
            yield captured

    def _run_command(self, func, *args, **kwargs):
        """
        Run a command function and capture its output.

        Returns (output, exception) tuple.
        Exception is None if command succeeded.
        """
        output = StringIO()
        exception = None

        # Patch both stdout and typer.echo to capture all output
        with patch('sys.stdout', output):
            with patch('typer.echo', lambda msg='', **kw: output.write(str(msg) + '\n')):
                with patch('typer.secho', lambda msg='', **kw: output.write(str(msg) + '\n')):
                    try:
                        func(*args, **kwargs)
                    except SystemExit:
                        # typer commands may call sys.exit(0) on success
                        pass
                    except Exception as e:
                        exception = e

        return output.getvalue(), exception

    # =========================================================================
    # /help - HelpRAG initialization and search
    # =========================================================================

    def test_help_with_query(self):
        """Test /help with a search query initializes HelpRAG."""
        from episodic.commands.help import help_command

        output, exc = self._run_command(help_command, "how do I change settings")

        assert exc is None, f"/help <query> crashed: {exc}"
        assert len(output) > 0, "/help <query> produced no output"

    # =========================================================================
    # /topics - List topics
    # =========================================================================

    def test_topics_list(self):
        """Test /topics lists topics without crashing."""
        from episodic.commands.unified_topics import topics_command

        output, exc = self._run_command(topics_command, action=None)

        assert exc is None, f"/topics crashed: {exc}"
        # Even with no topics, should produce some output
        assert len(output) > 0, "/topics produced no output"

    # =========================================================================
    # /ls - List recent nodes
    # =========================================================================

    def test_ls_basic(self):
        """Test /ls lists nodes without crashing."""
        from episodic.commands.navigation import list as list_nodes

        output, exc = self._run_command(list_nodes, count=10)

        assert exc is None, f"/ls crashed: {exc}"
        # Even with no nodes, should produce some output (header or "no nodes" message)
        assert len(output) > 0, "/ls produced no output"

    # =========================================================================
    # /memory - Memory status
    # =========================================================================

    def test_memory_status(self):
        """Test /memory shows status without crashing."""
        from episodic.commands.memory import memory_command

        output, exc = self._run_command(memory_command, action=None)

        assert exc is None, f"/memory crashed: {exc}"
        assert len(output) > 0, "/memory produced no output"

    # =========================================================================
    # /model - Current model
    # =========================================================================

    def test_model_show(self):
        """Test /model shows current model without crashing."""
        from episodic.commands.unified_model import model_command

        output, exc = self._run_command(model_command, context=None, model_name=None)

        assert exc is None, f"/model crashed: {exc}"
        assert len(output) > 0, "/model produced no output"

    # =========================================================================
    # /settings - Show settings
    # =========================================================================

    def test_settings_show(self):
        """Test /set shows settings without crashing."""
        from episodic.commands.settings import set as settings_set

        output, exc = self._run_command(settings_set, param=None, value=None)

        assert exc is None, f"/set crashed: {exc}"
        assert len(output) > 0, "/set produced no output"

    # =========================================================================
    # /rag - RAG status
    # =========================================================================

    def test_rag_stats(self):
        """Test /rag stats shows RAG status without crashing."""
        from episodic.commands.rag import rag_stats

        output, exc = self._run_command(rag_stats)

        assert exc is None, f"/rag stats crashed: {exc}"
        assert len(output) > 0, "/rag stats produced no output"

    # =========================================================================
    # /compression - Compression status
    # =========================================================================

    def test_compression_status(self):
        """Test /compression shows status without crashing."""
        from episodic.commands.unified_compression import compression_command

        output, exc = self._run_command(compression_command, action="stats")

        assert exc is None, f"/compression crashed: {exc}"
        assert len(output) > 0, "/compression produced no output"

    # =========================================================================
    # /muse - Web search mode
    # =========================================================================

    def test_muse_toggle(self):
        """Test /muse shows toggle state without crashing."""
        from episodic.commands.muse import muse

        output, exc = self._run_command(muse, action=None)

        assert exc is None, f"/muse crashed: {exc}"
        assert len(output) > 0, "/muse produced no output"

    # =========================================================================
    # /web - Web search
    # =========================================================================

    def test_web_search_toggle(self):
        """Test /web shows appropriate message."""
        from episodic.commands.web_search import websearch_toggle

        output, exc = self._run_command(websearch_toggle, enable=None)

        assert exc is None, f"/web toggle crashed: {exc}"
        assert len(output) > 0, "/web toggle produced no output"

    # =========================================================================
    # /style - Response style
    # =========================================================================

    def test_style_show(self):
        """Test /style shows current style without crashing."""
        from episodic.commands.style import style_command

        output, exc = self._run_command(style_command, style=None)

        assert exc is None, f"/style crashed: {exc}"
        assert len(output) > 0, "/style produced no output"

    # =========================================================================
    # /save - Save conversation (smoke test only, no actual save)
    # =========================================================================

    def test_save_no_conversation(self):
        """Test /save handles empty conversation gracefully."""
        from episodic.commands.save_load import save_command

        output, exc = self._run_command(save_command, filename=None)

        # Should not crash even with no conversation
        assert exc is None, f"/save crashed: {exc}"

    # =========================================================================
    # /new - New topic
    # =========================================================================

    def test_new_topic(self):
        """Test /new creates a topic without crashing."""
        from episodic.commands.new_topic import new_command

        output, exc = self._run_command(new_command, topic_name=None)

        assert exc is None, f"/new crashed: {exc}"


class TestCommandSmokeWithData:
    """Smoke tests that require some data in the database."""

    @pytest.fixture(autouse=True)
    def setup_with_data(self, isolated_config, integration_db):
        """Set up database with some test data."""
        from episodic.db import insert_node

        # Insert a few test nodes
        insert_node("Hello, how are you?", role="user")
        insert_node("I'm doing well, thank you!", role="assistant")
        insert_node("What's the weather like?", role="user")
        insert_node("I don't have access to real-time weather data.", role="assistant")

        yield

    def _run_command(self, func, *args, **kwargs):
        """Run a command function and capture its output."""
        output = StringIO()
        exception = None

        with patch('sys.stdout', output):
            with patch('typer.echo', lambda msg='', **kw: output.write(str(msg) + '\n')):
                with patch('typer.secho', lambda msg='', **kw: output.write(str(msg) + '\n')):
                    try:
                        func(*args, **kwargs)
                    except SystemExit:
                        pass
                    except Exception as e:
                        exception = e

        return output.getvalue(), exception

    def test_ls_with_data(self):
        """Test /ls shows nodes when data exists."""
        from episodic.commands.navigation import list as list_nodes

        output, exc = self._run_command(list_nodes, count=10)

        assert exc is None, f"/ls crashed with data: {exc}"
        assert len(output) > 0, "/ls produced no output with data"

    def test_topics_with_conversation(self):
        """Test /topics works after some conversation."""
        from episodic.commands.unified_topics import topics_command

        output, exc = self._run_command(topics_command, action=None)

        assert exc is None, f"/topics crashed with data: {exc}"
        assert len(output) > 0, "/topics produced no output"

    def test_summary_no_crash(self):
        """Test /summary doesn't crash (may need LLM, so just check init)."""
        from episodic.commands.summary import summary

        # Mock LLM to avoid actual API calls
        with patch('episodic.commands.summary.query_llm') as mock_llm:
            mock_llm.return_value = ("Test summary", {"cost_usd": 0})

            output, exc = self._run_command(summary, length="short")

            # Should not crash, even if LLM call is mocked
            assert exc is None, f"/summary crashed: {exc}"
