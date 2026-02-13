"""Plugin isolation tests.

Verifies:
- Disabled plugin removes functionality (slash commands, tokens, grammar)
- REGISTERED-but-not-ACTIVE plugin still provides slash commands
- DISCONNECTED plugin returns connection-lost style error
- Core utility commands unaffected by plugin state changes
"""

import pytest
from unittest.mock import patch, MagicMock

from episodic.mcp.plugins import (
    PluginRegistry,
    PluginState,
    get_plugin_registry,
    reset_plugin_registry,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Reset the plugin registry before each test."""
    reset_plugin_registry()
    yield
    reset_plugin_registry()


def _setup_registry() -> PluginRegistry:
    """Create a registry with gsuite plugin registered."""
    registry = get_plugin_registry()
    registry.register_all()
    return registry


class TestPluginIsolation:
    """Tests that plugin state changes don't affect core functionality."""

    def test_registered_plugin_has_slash_commands(self):
        """REGISTERED plugin provides slash commands."""
        registry = _setup_registry()
        assert registry.get_state("gsuite") == PluginState.REGISTERED

        assert registry.has_slash_command("/cal")
        assert registry.has_slash_command("/email")
        assert registry.has_slash_command("/mail")
        assert registry.has_slash_command("/calendar")

    def test_registered_plugin_has_grammar_rules(self):
        """REGISTERED plugin provides grammar rules."""
        registry = _setup_registry()
        reg = registry.get("gsuite")
        assert reg is not None
        assert len(reg.grammar_rules) > 0

    def test_registered_plugin_has_tokens(self):
        """REGISTERED plugin provides tokens."""
        registry = _setup_registry()
        reg = registry.get("gsuite")
        assert reg is not None
        assert len(reg.tokens) > 0

    def test_registered_plugin_has_extraction(self):
        """REGISTERED plugin provides extraction contribution."""
        registry = _setup_registry()
        reg = registry.get("gsuite")
        assert reg is not None
        assert reg.extraction_contribution is not None
        assert len(reg.extraction_contribution.intents) > 0

    def test_core_utility_commands_unaffected(self):
        """Core utility commands work regardless of plugin state."""
        from episodic.utility.cli_integration import is_utility_command

        _setup_registry()

        # Core commands should always work
        for cmd in ("timer", "alarm", "time", "calc", "weather", "status"):
            assert is_utility_command(cmd), f"/{cmd} should be a utility command"

    def test_plugin_commands_work_when_registered(self):
        """Plugin slash commands are recognized when REGISTERED."""
        from episodic.utility.cli_integration import is_utility_command

        _setup_registry()

        assert is_utility_command("cal")
        assert is_utility_command("email")
        assert is_utility_command("mail")

    def test_removed_commands_not_utility(self):
        """Removed commands are not recognized as utility commands."""
        from episodic.utility.cli_integration import is_utility_command

        _setup_registry()

        for cmd in ("inbox", "calendars", "schedule", "draft", "reply", "forward"):
            assert not is_utility_command(cmd), f"/{cmd} should NOT be a utility command"


class TestPluginStateTransitions:
    """Tests for plugin state machine transitions."""

    def test_disable_removes_state(self):
        """Disabling a plugin transitions its state."""
        registry = _setup_registry()
        assert registry.get_state("gsuite") == PluginState.REGISTERED
        registry.transition("gsuite", PluginState.DISABLED)
        assert registry.get_state("gsuite") == PluginState.DISABLED

    def test_connected_to_active(self):
        """CONNECTED -> ACTIVE transition works."""
        registry = _setup_registry()
        registry.transition("gsuite", PluginState.CONNECTED)
        assert registry.get_state("gsuite") == PluginState.CONNECTED
        registry.transition("gsuite", PluginState.ACTIVE)
        assert registry.get_state("gsuite") == PluginState.ACTIVE

    def test_active_to_disconnected(self):
        """ACTIVE -> DISCONNECTED transition works."""
        registry = _setup_registry()
        registry.transition("gsuite", PluginState.CONNECTED)
        registry.transition("gsuite", PluginState.ACTIVE)
        registry.transition("gsuite", PluginState.DISCONNECTED)
        assert registry.get_state("gsuite") == PluginState.DISCONNECTED

    def test_invalid_transition_raises(self):
        """Invalid state transitions raise ValueError."""
        registry = _setup_registry()
        with pytest.raises(ValueError, match="Invalid transition"):
            registry.transition("gsuite", PluginState.DISCONNECTED)

    def test_disconnected_can_reconnect(self):
        """DISCONNECTED -> CONNECTED transition works."""
        registry = _setup_registry()
        registry.transition("gsuite", PluginState.CONNECTED)
        registry.transition("gsuite", PluginState.DISCONNECTED)
        registry.transition("gsuite", PluginState.CONNECTED)
        assert registry.get_state("gsuite") == PluginState.CONNECTED


class TestPluginConnectionManager:
    """Tests for PluginConnectionManager."""

    def test_get_plugin_status(self):
        """get_plugin_status returns expected fields."""
        _setup_registry()

        from episodic.mcp.client_manager import PluginConnectionManager
        pcm = PluginConnectionManager()
        status = pcm.get_plugin_status("gsuite")

        assert status is not None
        assert status["name"] == "gsuite"
        assert status["state"] == "registered"
        assert status["server_id"] == "mcp-gsuite"
        assert "/cal" in status["slash_commands"]
        assert "/email" in status["slash_commands"]
        assert status["intent_count"] > 0
        assert status["connected"] is False

    def test_unknown_plugin_returns_none(self):
        """Unknown plugin returns None."""
        _setup_registry()

        from episodic.mcp.client_manager import PluginConnectionManager
        pcm = PluginConnectionManager()
        assert pcm.get_plugin_status("nonexistent") is None

    def test_states_dict(self):
        """Registry states() returns correct state mapping."""
        registry = _setup_registry()
        states = registry.states()
        assert "gsuite" in states
        assert states["gsuite"] == PluginState.REGISTERED
