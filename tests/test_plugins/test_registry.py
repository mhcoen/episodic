"""Tests for the PluginRegistry."""

import pytest

from episodic.mcp.plugins import (
    PluginRegistry,
    PluginState,
    get_plugin_registry,
    reset_plugin_registry,
)
from episodic.mcp.plugins._protocol import (
    PluginRegistration,
    ServerManifest,
    SlashCommand,
    TokenDefinition,
)


def _make_manifest(server_id: str = "test-server") -> ServerManifest:
    return ServerManifest(
        server_id=server_id,
        display_name="Test",
        command="cmd",
    )


def _make_registration(
    name: str = "test",
    slash_commands: list | None = None,
    tokens: list | None = None,
) -> PluginRegistration:
    return PluginRegistration(
        name=name,
        manifest=_make_manifest(),
        slash_commands=slash_commands or [],
        tokens=tokens or [],
    )


class TestDiscovery:
    def test_discover_and_register(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        assert reg.get_state("test") == PluginState.DISCOVERED
        result = reg.register("test")
        assert result.name == "test"
        assert reg.get_state("test") == PluginState.REGISTERED

    def test_discover_idempotent(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        reg.discover("test", lambda: _make_registration("other"))
        # First registration wins
        result = reg.register("test")
        assert result.name == "test"

    def test_register_unknown_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="not discovered"):
            reg.register("nope")

    def test_register_idempotent(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        r1 = reg.register("test")
        r2 = reg.register("test")
        assert r1 is r2


class TestStateMachine:
    def test_valid_transitions(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        assert reg.get_state("test") == PluginState.DISCOVERED

        reg.register("test")
        assert reg.get_state("test") == PluginState.REGISTERED

        reg.transition("test", PluginState.CONNECTED)
        assert reg.get_state("test") == PluginState.CONNECTED

        reg.transition("test", PluginState.ACTIVE)
        assert reg.get_state("test") == PluginState.ACTIVE

        reg.transition("test", PluginState.DISCONNECTED)
        assert reg.get_state("test") == PluginState.DISCONNECTED

        # Can reconnect
        reg.transition("test", PluginState.CONNECTED)
        assert reg.get_state("test") == PluginState.CONNECTED

    def test_invalid_transition_raises(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        with pytest.raises(ValueError, match="Invalid transition"):
            reg.transition("test", PluginState.ACTIVE)

    def test_disable_from_any_state(self):
        for start_state in (
            PluginState.DISCOVERED,
            PluginState.REGISTERED,
            PluginState.CONNECTED,
            PluginState.ACTIVE,
            PluginState.DISCONNECTED,
        ):
            reg = PluginRegistry()
            reg.discover("test", lambda: _make_registration())
            # Force state for testing
            reg._plugins["test"].state = start_state
            reg.transition("test", PluginState.DISABLED)
            assert reg.get_state("test") == PluginState.DISABLED

    def test_re_enable_from_disabled(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        reg._plugins["test"].state = PluginState.DISABLED
        reg.transition("test", PluginState.DISCOVERED)
        assert reg.get_state("test") == PluginState.DISCOVERED

    def test_transition_unknown_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.transition("nope", PluginState.ACTIVE)


class TestSlashCommands:
    def test_slash_command_indexing(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration(
            slash_commands=[
                SlashCommand(name="/cal", aliases=["/calendar"], domain="calendar"),
                SlashCommand(name="/email", aliases=["/mail", "/gmail"], domain="email"),
            ],
        ))
        reg.register("test")

        assert reg.has_slash_command("/cal")
        assert reg.has_slash_command("/calendar")
        assert reg.has_slash_command("/email")
        assert reg.has_slash_command("/mail")
        assert reg.has_slash_command("/gmail")
        assert not reg.has_slash_command("/timer")

    def test_get_slash_command(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration(
            slash_commands=[
                SlashCommand(name="/cal", aliases=["/calendar"], domain="calendar"),
            ],
        ))
        reg.register("test")

        sc = reg.get_slash_command("/cal")
        assert sc is not None
        assert sc.name == "/cal"

        sc2 = reg.get_slash_command("/calendar")
        assert sc2 is not None
        assert sc2.name == "/cal"  # Resolves to canonical

        assert reg.get_slash_command("/nope") is None

    def test_get_plugin_for_command(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration(
            slash_commands=[SlashCommand(name="/cal")],
        ))
        reg.register("test")
        assert reg.get_plugin_for_command("/cal") == "test"
        assert reg.get_plugin_for_command("/nope") is None


class TestListMethods:
    def test_registered_and_names(self):
        reg = PluginRegistry()
        reg.discover("a", lambda: _make_registration("a"))
        reg.discover("b", lambda: _make_registration("b"))
        reg.register("a")

        assert len(reg.registered()) == 1
        assert reg.registered()[0].name == "a"
        assert set(reg.names()) == {"a", "b"}

    def test_states(self):
        reg = PluginRegistry()
        reg.discover("a", lambda: _make_registration("a"))
        reg.discover("b", lambda: _make_registration("b"))
        reg.register("a")
        states = reg.states()
        assert states["a"] == PluginState.REGISTERED
        assert states["b"] == PluginState.DISCOVERED


class TestRegisterAll:
    def test_idempotent(self):
        reg = PluginRegistry()
        reg.discover("test", lambda: _make_registration())
        reg.register_all()
        assert reg.initialized
        assert reg.get_state("test") == PluginState.REGISTERED
        # Second call is a no-op
        reg.register_all()
        assert reg.initialized


class TestSingleton:
    def test_singleton(self):
        reset_plugin_registry()
        r1 = get_plugin_registry()
        r2 = get_plugin_registry()
        assert r1 is r2
        reset_plugin_registry()

    def test_reset(self):
        reset_plugin_registry()
        r1 = get_plugin_registry()
        reset_plugin_registry()
        r2 = get_plugin_registry()
        assert r1 is not r2
        reset_plugin_registry()
