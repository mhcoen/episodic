"""
Tests for MCP Resolver and Schema Utils.

Spec tests 13-16b from CFG_MCP_DISPATCH_EXTENSION.md §9.2.
"""

import pytest
from episodic.mcp.dispatch import MCPResolver
from episodic.mcp.dispatch_types import DEFAULT_INTENT_MAPPING
from episodic.mcp.schema_utils import (
    normalize_schema,
    canonical_json,
    schema_fingerprint,
)


@pytest.fixture
def resolver():
    return MCPResolver()


class TestMCPResolver:
    """Spec tests 13-16b."""

    def test_13_mapped_intent_resolves(self, resolver):
        """Test 13: Mapped intent returns correct MCPResolution."""
        res = resolver.resolve("email.search")
        assert res is not None
        assert res.tool_name == "query_gmail_emails"
        assert res.sensitivity == "read"
        assert res.requires_auth_event is False

    def test_14_unmapped_intent_returns_none(self, resolver):
        """Test 14: Unmapped intent returns None (falls through)."""
        res = resolver.resolve("unknown.command")
        assert res is None

    def test_15_decomposed_intent_returns_none(self, resolver):
        """Test 15: Decomposed intent (no single tool) returns None."""
        # calendar.reschedule has tool=None in default mapping
        res = resolver.resolve("calendar.reschedule")
        assert res is None

    def test_16a_schema_required_false_normalization(self):
        """Test 16a: Schema with required=false (boolean) is normalized."""
        schema_bad = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "required": False}
            }
        }
        schema_good = {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
        assert schema_fingerprint(schema_bad) == schema_fingerprint(schema_good)

    def test_16b_freebusy_same_tool_as_query(self, resolver):
        """Test 16b: calendar.freebusy and calendar.query both resolve to same tool."""
        query_res = resolver.resolve("calendar.query")
        freebusy_res = resolver.resolve("calendar.freebusy")
        assert query_res is not None
        assert freebusy_res is not None
        assert query_res.tool_name == freebusy_res.tool_name
        assert query_res.tool_name == "get_calendar_events"

    def test_write_intent_requires_auth(self, resolver):
        """Write intents require authorization events."""
        res = resolver.resolve("calendar.create")
        assert res is not None
        assert res.sensitivity == "write"
        assert res.requires_auth_event is True

    def test_destructive_intent(self, resolver):
        """Destructive intents are marked correctly."""
        res = resolver.resolve("calendar.delete")
        assert res is not None
        assert res.sensitivity == "destructive"
        assert res.requires_auth_event is True


class TestSchemaUtils:
    """Schema normalization and fingerprinting tests."""

    def test_normalize_strips_boolean_required(self):
        """Strips 'required': false (boolean) from properties."""
        schema = {"properties": {"q": {"type": "string", "required": False}}}
        normalized = normalize_schema(schema)
        assert "required" not in normalized["properties"]["q"]

    def test_normalize_preserves_array_required(self):
        """Preserves 'required': ['field'] (array) at top level."""
        schema = {"required": ["name"], "properties": {"name": {"type": "string"}}}
        normalized = normalize_schema(schema)
        assert normalized["required"] == ["name"]

    def test_canonical_json_deterministic(self):
        """Same data produces same canonical JSON regardless of dict order."""
        a = canonical_json({"b": 2, "a": 1})
        b = canonical_json({"a": 1, "b": 2})
        assert a == b

    def test_fingerprint_stable(self):
        """Same schema produces same fingerprint."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        fp1 = schema_fingerprint(schema)
        fp2 = schema_fingerprint(schema)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_fingerprint_differs_for_different_schemas(self):
        """Different schemas produce different fingerprints."""
        s1 = {"type": "object", "properties": {"x": {"type": "string"}}}
        s2 = {"type": "object", "properties": {"y": {"type": "integer"}}}
        assert schema_fingerprint(s1) != schema_fingerprint(s2)
