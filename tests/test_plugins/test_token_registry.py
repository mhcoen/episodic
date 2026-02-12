"""Tests for the TokenRegistry."""

import pytest

from episodic.mcp.plugins._token_registry import TokenRegistry
from episodic.mcp.plugins._protocol import TokenDefinition


class TestCoreTokens:
    def test_register_core(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER", "alarm": "KW_ALARM"})
        wm = tr.get_word_map()
        assert wm["timer"] == "KW_TIMER"
        assert wm["alarm"] == "KW_ALARM"

    def test_core_source(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        assert tr.source_of("timer") == "__core__"

    def test_has_word(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        assert tr.has_word("timer")
        assert not tr.has_word("calendar")


class TestPluginTokens:
    def test_register_plugin(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        tokens = [
            TokenDefinition(word="calendar", token_kind="KW_CALENDAR"),
            TokenDefinition(word="meeting", token_kind="KW_MEETING"),
        ]
        warnings = tr.register_plugin("gsuite", tokens)
        assert warnings == []
        wm = tr.get_word_map()
        assert wm["calendar"] == "KW_CALENDAR"
        assert wm["meeting"] == "KW_MEETING"
        assert tr.source_of("calendar") == "gsuite"

    def test_get_plugin_tokens(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        tr.register_plugin("gsuite", [
            TokenDefinition(word="calendar", token_kind="KW_CALENDAR"),
        ])
        pt = tr.get_plugin_tokens("gsuite")
        assert pt == {"calendar": "KW_CALENDAR"}
        assert tr.get_plugin_tokens("other") == {}


class TestCollisionDetection:
    def test_core_shadow_warning(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        warnings = tr.register_plugin("bad_plugin", [
            TokenDefinition(word="timer", token_kind="KW_PLUGIN_TIMER"),
        ])
        assert len(warnings) == 1
        assert "shadows core" in warnings[0]
        # Core token preserved
        assert tr.get_word_map()["timer"] == "KW_TIMER"

    def test_plugin_collision_raises(self):
        tr = TokenRegistry()
        tr.register_plugin("plugin_a", [
            TokenDefinition(word="calendar", token_kind="KW_CAL_A"),
        ])
        with pytest.raises(ValueError, match="Token collision"):
            tr.register_plugin("plugin_b", [
                TokenDefinition(word="calendar", token_kind="KW_CAL_B"),
            ])

    def test_same_plugin_reregister_ok(self):
        tr = TokenRegistry()
        tr.register_plugin("gsuite", [
            TokenDefinition(word="calendar", token_kind="KW_CALENDAR"),
        ])
        # Same plugin, same word: no error
        warnings = tr.register_plugin("gsuite", [
            TokenDefinition(word="calendar", token_kind="KW_CALENDAR_V2"),
        ])
        assert warnings == []
        assert tr.get_word_map()["calendar"] == "KW_CALENDAR_V2"


class TestFreeze:
    def test_freeze_prevents_registration(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        tr.freeze()
        assert tr.frozen

        with pytest.raises(RuntimeError, match="frozen"):
            tr.register_core({"alarm": "KW_ALARM"})

        with pytest.raises(RuntimeError, match="frozen"):
            tr.register_plugin("test", [
                TokenDefinition(word="cal", token_kind="KW_CAL"),
            ])

    def test_get_word_map_after_freeze(self):
        tr = TokenRegistry()
        tr.register_core({"timer": "KW_TIMER"})
        tr.register_plugin("gsuite", [
            TokenDefinition(word="calendar", token_kind="KW_CALENDAR"),
        ])
        tr.freeze()
        wm = tr.get_word_map()
        assert "timer" in wm
        assert "calendar" in wm
