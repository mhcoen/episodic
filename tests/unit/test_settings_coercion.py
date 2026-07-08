"""
Tests for generic /set value coercion.

Regression guard for the config boolean bug: an unhandled boolean parameter
set via `/set <param> false` used to be stored as the (truthy) string
"false", silently leaving the feature enabled.
"""

import pytest

from episodic.commands.settings import _coerce_generic_value, _str2bool, set as set_cmd
from episodic.config import config


class TestStr2Bool:
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on", " on "])
    def test_truthy(self, value):
        assert _str2bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "", "nonsense"])
    def test_falsy(self, value):
        assert _str2bool(value) is False


class TestCoerceTemplateTyped:
    """Params that appear in the template are coerced to the default's type."""

    @pytest.mark.parametrize("param", [
        "enable_relevance_truncation",
        "conversation_retrieval_enabled",
        "topic_context_retrieval",
        "kg_realtime",
    ])
    def test_bool_param_false_becomes_bool_false(self, param):
        # Sanity: these really are booleans in the template.
        assert isinstance(config.get_template_defaults().get(param), bool)

        result = _coerce_generic_value(param, "false")
        assert result is False
        assert isinstance(result, bool)

    @pytest.mark.parametrize("param", [
        "enable_relevance_truncation",
        "kg_realtime",
    ])
    def test_bool_param_true_becomes_bool_true(self, param):
        result = _coerce_generic_value(param, "true")
        assert result is True
        assert isinstance(result, bool)


class TestCoerceMissingFromTemplate:
    """Params absent from the template are inferred from the literal."""

    @pytest.mark.parametrize("param", [
        "enable_memory_rag",
        "skip_llm_response",
        "enable_smart_memory",
    ])
    def test_missing_bool_param_coerced(self, param):
        # These are read as booleans in the pipeline but aren't in the template.
        assert param not in config.get_template_defaults()

        assert _coerce_generic_value(param, "false") is False
        assert _coerce_generic_value(param, "true") is True

    def test_numeric_inference(self):
        assert _coerce_generic_value("some_unknown_int", "42") == 42
        assert isinstance(_coerce_generic_value("some_unknown_int", "42"), int)
        assert _coerce_generic_value("some_unknown_float", "1.5") == 1.5
        assert isinstance(_coerce_generic_value("some_unknown_float", "1.5"), float)

    def test_plain_string_preserved(self):
        assert _coerce_generic_value("some_unknown_str", "hello") == "hello"


class TestSetCommandEndToEnd:
    """Driving the actual /set command stores a real bool, not a string."""

    def test_set_false_disables_feature(self):
        original = config.get("enable_relevance_truncation")
        try:
            set_cmd("enable_relevance_truncation", "true")
            assert config.get("enable_relevance_truncation") is True

            set_cmd("enable_relevance_truncation", "false")
            stored = config.get("enable_relevance_truncation")
            assert stored is False
            # The core bug: a truthy string would pass `if config.get(...)`.
            assert not stored
        finally:
            config.set("enable_relevance_truncation", original)

    def test_set_missing_param_false_is_falsy(self):
        original = config.get("skip_llm_response")
        try:
            set_cmd("skip_llm_response", "false")
            stored = config.get("skip_llm_response", False)
            assert stored is False
            assert not stored
        finally:
            config.set("skip_llm_response", original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
