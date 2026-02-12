"""Tests for the gsuite plugin registration."""

from episodic.mcp.plugins.gsuite import register
from episodic.mcp.plugins._protocol import PluginRegistration


class TestGsuiteRegister:
    def test_returns_registration(self):
        reg = register()
        assert isinstance(reg, PluginRegistration)
        assert reg.name == "gsuite"

    def test_manifest(self):
        reg = register()
        assert reg.manifest.server_id == "mcp-gsuite"
        assert reg.manifest.display_name == "Google Workspace"
        assert reg.manifest.command == "npx"

    def test_only_cal_and_email_slash_commands(self):
        reg = register()
        names = [sc.name for sc in reg.slash_commands]
        assert names == ["/cal", "/email"]

    def test_cal_command(self):
        reg = register()
        cal = [sc for sc in reg.slash_commands if sc.name == "/cal"][0]
        assert cal.aliases == ["/calendar"]
        assert cal.domain == "calendar"
        assert "today" in cal.completions
        assert "tomorrow" in cal.completions

    def test_email_command(self):
        reg = register()
        email = [sc for sc in reg.slash_commands if sc.name == "/email"][0]
        assert email.aliases == ["/mail", "/gmail"]
        assert email.domain == "email"
        assert "unread" in email.completions

    def test_extraction_contribution_has_6_intents(self):
        reg = register()
        ec = reg.extraction_contribution
        assert ec is not None
        assert len(ec.intents) == 6

    def test_extraction_contribution_gate_keywords(self):
        reg = register()
        ec = reg.extraction_contribution
        kw = set(ec.gate_keywords)
        assert "calendar" in kw
        assert "email" in kw
        assert "meeting" in kw
        assert "inbox" in kw

    def test_extraction_contribution_gate_phrases(self):
        reg = register()
        ec = reg.extraction_contribution
        # Should have phrases from both calendar and email
        assert len(ec.gate_phrases) >= 4

    def test_tokens_no_collision_with_core(self):
        """Tokens should not collide with core non-plugin tokens."""
        from episodic.mcp.plugins._token_registry import TokenRegistry
        reg = register()

        tr = TokenRegistry()
        # Register some core tokens
        core = {
            "timer": "KW_TIMER", "alarm": "KW_ALARM",
            "set": "ACTION_SET", "cancel": "ACTION_CANCEL",
        }
        tr.register_core(core)
        # Plugin tokens should not shadow core
        warnings = tr.register_plugin("gsuite", reg.tokens)
        for w in warnings:
            assert "shadows core" in w

    def test_grammar_rules_not_empty(self):
        reg = register()
        assert len(reg.grammar_rules) > 0

    def test_tool_map_not_empty(self):
        reg = register()
        assert len(reg.tool_map) > 0
        assert "calendar.query" in reg.tool_map
        assert "email.search" in reg.tool_map

    def test_adapter_map_not_empty(self):
        reg = register()
        assert len(reg.adapter_map) > 0
        assert "calendar.query" in reg.adapter_map
        assert "email.search" in reg.adapter_map

    def test_help_fn(self):
        reg = register()
        assert reg.help_fn is not None
        text = reg.help_fn()
        assert "/cal" in text
        assert "/email" in text

    def test_help_category(self):
        reg = register()
        assert reg.help_category == "Calendar & Email"

    def test_arg_extractors(self):
        reg = register()
        assert "calendar." in reg.arg_extractors
        assert "email." in reg.arg_extractors
