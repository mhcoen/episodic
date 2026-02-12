"""Tests for plugin protocol types."""

from episodic.mcp.plugins._protocol import (
    PluginRegistration,
    ServerManifest,
    SlashCommand,
    TokenDefinition,
)


class TestServerManifest:
    def test_fields(self):
        m = ServerManifest(
            server_id="test-server",
            display_name="Test Server",
            command="npx",
            args=["-y", "@test/server"],
            env_vars=["TEST_API_KEY"],
            connect_policy="manual",
        )
        assert m.server_id == "test-server"
        assert m.display_name == "Test Server"
        assert m.command == "npx"
        assert m.args == ["-y", "@test/server"]
        assert m.env_vars == ["TEST_API_KEY"]
        assert m.connect_policy == "manual"

    def test_defaults(self):
        m = ServerManifest(server_id="s", display_name="S", command="cmd")
        assert m.args == []
        assert m.env_vars == []
        assert m.connect_policy == "manual"

    def test_frozen(self):
        m = ServerManifest(server_id="s", display_name="S", command="cmd")
        try:
            m.server_id = "other"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestSlashCommand:
    def test_fields(self):
        sc = SlashCommand(
            name="/cal",
            aliases=["/calendar"],
            category="Calendar",
            description="Calendar commands",
            domain="calendar",
            completions=["today", "tomorrow"],
        )
        assert sc.name == "/cal"
        assert sc.aliases == ["/calendar"]
        assert sc.domain == "calendar"
        assert sc.completions == ["today", "tomorrow"]

    def test_defaults(self):
        sc = SlashCommand(name="/test")
        assert sc.aliases == []
        assert sc.category == ""
        assert sc.completions == []


class TestTokenDefinition:
    def test_fields(self):
        td = TokenDefinition(word="calendar", token_kind="KW_CALENDAR")
        assert td.word == "calendar"
        assert td.token_kind == "KW_CALENDAR"


class TestPluginRegistration:
    def test_minimal(self):
        reg = PluginRegistration(
            name="test",
            manifest=ServerManifest(
                server_id="test", display_name="Test", command="cmd"
            ),
        )
        assert reg.name == "test"
        assert reg.slash_commands == []
        assert reg.tokens == []
        assert reg.grammar_rules == []
        assert reg.tool_map == {}
        assert reg.adapter_map == {}
        assert reg.help_fn is None
        assert reg.extraction_contribution is None

    def test_full(self):
        reg = PluginRegistration(
            name="gsuite",
            manifest=ServerManifest(
                server_id="mcp-gsuite",
                display_name="Google Workspace",
                command="npx",
            ),
            slash_commands=[SlashCommand(name="/cal")],
            tokens=[TokenDefinition(word="calendar", token_kind="KW_CAL")],
            help_category="Calendar & Email",
        )
        assert len(reg.slash_commands) == 1
        assert len(reg.tokens) == 1
        assert reg.help_category == "Calendar & Email"
