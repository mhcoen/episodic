"""
Unit tests for Token Guard Event Logging (Category D).

Tests cover:
1. Required fields present in all events
2. Exactly-once constraint per assembly_id
3. Monotonic event_seq per run_id
4. Hash chain validation
5. Hash chain tamper detection
6. Canonicalization determinism
7. Unknown fields allowed (forward compatibility)
"""

import copy
import hashlib
import json
import pytest
from io import StringIO
from typing import List, Dict, Any

from episodic.token_guard_events import (
    SCHEMA_VERSION,
    EventType,
    TokenGuardEvent,
    EventLogger,
    EventVerifier,
    canonical_json,
    compute_hash,
    get_event_logger,
    reset_event_logger,
    configure_event_logger,
)
from episodic.token_guard import (
    validate_assembly,
    TokenBudget,
)


@pytest.fixture
def logger():
    """Create a fresh event logger for each test."""
    output = StringIO()
    log = EventLogger(output=output, run_id="test-run-001")
    yield log


@pytest.fixture
def reset_global_logger():
    """Reset global logger before and after test."""
    reset_event_logger()
    yield
    reset_event_logger()


class TestTokenGuardEvent:
    """Tests for TokenGuardEvent dataclass."""

    def test_required_fields_defined(self):
        """Required fields set is defined and complete."""
        required = TokenGuardEvent.required_fields()
        assert "schema_version" in required
        assert "event_type" in required
        assert "run_id" in required
        assert "turn_id" in required
        assert "assembly_id" in required
        assert "ts" in required
        assert "event_seq" in required
        assert "counter_backend" in required
        assert "counter_exact" in required
        assert "applied_safety_factor" in required
        assert "raw_tokens" in required
        assert "effective_tokens" in required
        assert "cap" in required
        assert "budget_breakdown" in required
        assert "prev_hash" in required
        assert "hash" in required

    def test_to_dict_includes_all_fields(self):
        """to_dict includes all fields."""
        event = TokenGuardEvent(
            schema_version="1.0",
            event_type="token_ok",
            run_id="run-1",
            turn_id="turn-1",
            assembly_id="asm-1",
            ts="2024-01-01T00:00:00Z",
            event_seq=1,
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor="1.200000",
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={"user_message": 100},
            prev_hash=None,
            hash="abc123",
        )
        d = event.to_dict()

        for field in TokenGuardEvent.required_fields():
            assert field in d

    def test_from_dict_preserves_unknown_fields(self):
        """from_dict preserves unknown fields in extra."""
        d = {
            "schema_version": "1.0",
            "event_type": "token_ok",
            "run_id": "run-1",
            "turn_id": "turn-1",
            "assembly_id": "asm-1",
            "ts": "2024-01-01T00:00:00Z",
            "event_seq": 1,
            "counter_backend": "heuristic",
            "counter_exact": False,
            "applied_safety_factor": None,
            "raw_tokens": 100,
            "effective_tokens": 100,
            "cap": 1000,
            "budget_breakdown": {},
            "prev_hash": None,
            "hash": "abc123",
            "custom_field": "custom_value",  # Unknown field
            "another_custom": 42,
        }
        event = TokenGuardEvent.from_dict(d)

        assert event.extra.get("custom_field") == "custom_value"
        assert event.extra.get("another_custom") == 42

        # to_dict should include extra fields
        d_out = event.to_dict()
        assert d_out.get("custom_field") == "custom_value"


class TestCanonicalization:
    """Tests for canonical JSON and hashing."""

    def test_canonical_json_sorted_keys(self):
        """Canonical JSON has sorted keys."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        # Keys should appear in sorted order
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_canonical_json_compact(self):
        """Canonical JSON is compact (no extra whitespace)."""
        obj = {"key": "value", "num": 123}
        result = canonical_json(obj)
        assert " " not in result
        assert "\n" not in result

    def test_canonical_json_deterministic(self):
        """Same input produces same output across calls."""
        obj = {"a": 1, "b": [2, 3], "c": {"d": 4}}

        result1 = canonical_json(obj)
        result2 = canonical_json(obj)
        result3 = canonical_json(copy.deepcopy(obj))

        assert result1 == result2 == result3

    def test_compute_hash_deterministic(self):
        """Same inputs produce same hash."""
        prev = "prevhash123"
        event = {"a": 1, "b": 2}

        h1 = compute_hash(prev, event)
        h2 = compute_hash(prev, event)

        assert h1 == h2

    def test_compute_hash_different_with_different_prev(self):
        """Different prev_hash produces different hash."""
        event = {"a": 1}

        h1 = compute_hash("prev1", event)
        h2 = compute_hash("prev2", event)

        assert h1 != h2

    def test_compute_hash_is_sha256(self):
        """Hash is 64-char hex (SHA-256)."""
        h = compute_hash(None, {"test": "data"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestEventLogger:
    """Tests for EventLogger class."""

    def test_emit_returns_event(self, logger):
        """emit() returns the emitted event."""
        event = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={"user_message": 100}
        )

        assert isinstance(event, TokenGuardEvent)
        assert event.event_type == "token_ok"
        assert event.assembly_id == "asm-1"

    def test_emit_assigns_monotonic_event_seq(self, logger):
        """event_seq is monotonically increasing."""
        e1 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )
        e2 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-2",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )
        e3 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-3",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        assert e1.event_seq < e2.event_seq < e3.event_seq

    def test_emit_enforces_exactly_once(self, logger):
        """Same assembly_id cannot emit twice."""
        logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        with pytest.raises(ValueError, match="(?i)exactly-once"):
            logger.emit(
                event_type=EventType.TOKEN_OK,
                assembly_id="asm-1",  # Same assembly_id!
                turn_id="turn-1",
                counter_backend="heuristic",
                counter_exact=False,
                applied_safety_factor=1.2,
                raw_tokens=100,
                effective_tokens=120,
                cap=1000,
                budget_breakdown={}
            )

    def test_emit_builds_hash_chain(self, logger):
        """Events form a hash chain."""
        e1 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )
        e2 = logger.emit(
            event_type=EventType.TOKEN_OVERFLOW_RECOVERED,
            assembly_id="asm-2",
            turn_id="turn-2",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=200,
            effective_tokens=180,
            cap=1000,
            budget_breakdown={}
        )

        assert e1.prev_hash is None
        assert e2.prev_hash == e1.hash

    def test_emit_writes_jsonl_to_output(self, logger):
        """Events are written as JSONL to output."""
        logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        output = logger._output.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 1

        # Parse the line
        event_dict = json.loads(lines[0])
        assert event_dict["event_type"] == "token_ok"

    def test_events_property_returns_copy(self, logger):
        """events property returns a copy of buffered events."""
        logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        events1 = logger.events
        events2 = logger.events

        assert events1 is not events2
        assert len(events1) == 1


class TestEventVerifier:
    """Tests for EventVerifier class."""

    def test_verify_required_fields_pass(self):
        """Valid event passes required fields check."""
        event = {
            "schema_version": "1.0",
            "event_type": "token_ok",
            "run_id": "run-1",
            "turn_id": "turn-1",
            "assembly_id": "asm-1",
            "ts": "2024-01-01T00:00:00Z",
            "event_seq": 1,
            "counter_backend": "heuristic",
            "counter_exact": False,
            "applied_safety_factor": None,
            "raw_tokens": 100,
            "effective_tokens": 100,
            "cap": 1000,
            "budget_breakdown": {},
            "prev_hash": None,
            "hash": "abc123",
        }
        missing = EventVerifier.verify_required_fields(event)
        assert missing == []

    def test_verify_required_fields_fail(self):
        """Missing fields are reported."""
        event = {
            "schema_version": "1.0",
            "event_type": "token_ok",
            # Missing many fields
        }
        missing = EventVerifier.verify_required_fields(event)
        assert "run_id" in missing
        assert "assembly_id" in missing
        assert "hash" in missing

    def test_verify_hash_pass(self):
        """Valid hash passes verification."""
        prev_hash = "prevhash"
        event_without_hash = {
            "schema_version": "1.0",
            "prev_hash": prev_hash,
            "other": "data",
        }
        expected_hash = compute_hash(prev_hash, event_without_hash)
        event = {**event_without_hash, "hash": expected_hash}

        assert EventVerifier.verify_hash(event, prev_hash) is True

    def test_verify_hash_fail_wrong_hash(self):
        """Wrong hash fails verification."""
        prev_hash = "prevhash"
        event = {
            "schema_version": "1.0",
            "prev_hash": prev_hash,
            "hash": "wronghash",
        }
        assert EventVerifier.verify_hash(event, prev_hash) is False

    def test_verify_hash_fail_wrong_prev(self):
        """Wrong prev_hash fails verification."""
        event = {
            "schema_version": "1.0",
            "prev_hash": "different_prev",
            "hash": "somehash",
        }
        assert EventVerifier.verify_hash(event, "expected_prev") is False

    def test_verify_stream_valid(self, logger):
        """Valid event stream passes verification."""
        logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={"user_message": 100}
        )
        logger.emit(
            event_type=EventType.TOKEN_OVERFLOW_RECOVERED,
            assembly_id="asm-2",
            turn_id="turn-2",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=200,
            effective_tokens=180,
            cap=1000,
            budget_breakdown={"user_message": 200}
        )

        events = [e.to_dict() for e in logger.events]
        result = EventVerifier.verify_stream(events)

        assert result["valid"] is True
        assert result["event_count"] == 2
        assert len(result["errors"]) == 0

    def test_verify_stream_detects_tampered_hash(self, logger):
        """Tampering with event hash is detected."""
        logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )
        logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-2",
            turn_id="turn-2",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        events = [e.to_dict() for e in logger.events]

        # Tamper with first event
        events[0]["raw_tokens"] = 999

        result = EventVerifier.verify_stream(events)

        # First event hash is now invalid
        assert result["valid"] is False
        assert any("hash chain" in err for err in result["errors"])

    def test_verify_stream_detects_duplicate_assembly_id(self, logger):
        """Duplicate assembly_id is detected."""
        e1 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        events = [e1.to_dict(), e1.to_dict()]  # Duplicate!

        result = EventVerifier.verify_stream(events)

        assert result["valid"] is False
        assert any("duplicate" in err.lower() or "exactly-once" in err.lower()
                   for err in result["errors"])

    def test_verify_stream_detects_non_monotonic_seq(self, logger):
        """Non-monotonic event_seq is detected."""
        e1 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-1",
            turn_id="turn-1",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )
        e2 = logger.emit(
            event_type=EventType.TOKEN_OK,
            assembly_id="asm-2",
            turn_id="turn-2",
            counter_backend="heuristic",
            counter_exact=False,
            applied_safety_factor=1.2,
            raw_tokens=100,
            effective_tokens=120,
            cap=1000,
            budget_breakdown={}
        )

        events = [e1.to_dict(), e2.to_dict()]
        # Swap event_seq values
        events[0]["event_seq"] = 5
        events[1]["event_seq"] = 3  # Not monotonic!

        result = EventVerifier.verify_stream(events)

        # Hash chain will be broken and/or monotonic check will fail
        assert result["valid"] is False


class TestValidateAssemblyEventEmission:
    """Tests for event emission from validate_assembly."""

    def test_validate_assembly_emits_token_ok(self, reset_global_logger):
        """validate_assembly emits token_ok for valid assembly."""
        configure_event_logger(run_id="test-validate-ok")

        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        result = validate_assembly(messages, budget, emit_event=True)

        events = get_event_logger().events
        assert len(events) == 1
        assert events[0].event_type == "token_ok"

    def test_validate_assembly_emits_token_overflow_abort(self, reset_global_logger):
        """validate_assembly emits token_overflow_abort when abort triggered."""
        configure_event_logger(run_id="test-validate-abort")

        huge_content = "X" * 50000
        messages = [{"role": "user", "content": huge_content}]
        budget = TokenBudget(full_cap=100)

        result = validate_assembly(messages, budget, emit_event=True)

        events = get_event_logger().events
        assert len(events) == 1
        assert events[0].event_type == "token_overflow_abort"

    def test_validate_assembly_emits_exactly_once(self, reset_global_logger):
        """Each validate_assembly call emits exactly one event."""
        configure_event_logger(run_id="test-exactly-once")

        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        # Multiple calls with unique assembly_ids
        validate_assembly(messages, budget, assembly_id="a1", emit_event=True)
        validate_assembly(messages, budget, assembly_id="a2", emit_event=True)
        validate_assembly(messages, budget, assembly_id="a3", emit_event=True)

        events = get_event_logger().events
        assert len(events) == 3

        # Verify uniqueness
        assembly_ids = [e.assembly_id for e in events]
        assert len(set(assembly_ids)) == 3

    def test_validate_assembly_event_has_required_fields(self, reset_global_logger):
        """Emitted events have all required fields."""
        configure_event_logger(run_id="test-fields")

        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        validate_assembly(messages, budget, emit_event=True)

        events = get_event_logger().events
        event_dict = events[0].to_dict()

        missing = EventVerifier.verify_required_fields(event_dict)
        assert missing == []

    def test_validate_assembly_emit_event_false(self, reset_global_logger):
        """emit_event=False skips event emission."""
        configure_event_logger(run_id="test-no-emit")

        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        validate_assembly(messages, budget, emit_event=False)

        events = get_event_logger().events
        assert len(events) == 0


class TestIntegrityEndToEnd:
    """End-to-end tests for log integrity."""

    def test_full_stream_integrity(self, reset_global_logger):
        """Full event stream maintains integrity through various scenarios."""
        output = StringIO()
        configure_event_logger(output=output, run_id="test-e2e")

        messages_small = [{"role": "user", "content": "Hello"}]
        messages_medium = [{"role": "user", "content": "X" * 1000}]
        messages_large = [{"role": "user", "content": "X" * 50000}]

        budget = TokenBudget(full_cap=500, overhead_reserve=100)

        # Mix of scenarios
        validate_assembly(messages_small, budget, assembly_id="a1")  # OK
        validate_assembly(messages_medium, budget, assembly_id="a2")  # Likely recovered
        validate_assembly(messages_large, budget, assembly_id="a3")  # Abort

        # Parse JSONL output
        output_str = output.getvalue()
        lines = output_str.strip().split("\n")
        events = [json.loads(line) for line in lines]

        # Verify stream integrity
        result = EventVerifier.verify_stream(events)

        assert result["valid"] is True
        assert result["event_count"] == 3
        assert len(result["assembly_ids"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
