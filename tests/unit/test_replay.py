"""
Unit tests for Snapshot-Based Replay Infrastructure (Category E).

Tests cover:
1. Snapshot schema with all required fields
2. Golden snapshot test (deterministic replay)
3. Mutation tests (divergence detection and localization)
4. Hash chain verification through replay
5. Token count verification
6. Event stream integrity
"""

import copy
import json
import pytest
import tempfile
from pathlib import Path
from io import StringIO
from typing import Dict, Any, List

from episodic.replay import (
    SNAPSHOT_SCHEMA_VERSION,
    ReplaySnapshot,
    ReplayResult,
    ReplayDiff,
    RetrievalState,
    TokenGuardConfig,
    ContextInputs,
    replay,
    create_snapshot,
    assemble_from_snapshot,
    _compare_messages,
    _compare_events,
)
from episodic.token_guard import (
    TokenBudget,
    validate_assembly,
    HeuristicTokenCounter,
)
from episodic.token_guard_events import (
    EventLogger,
    EventVerifier,
    TokenGuardEvent,
    reset_event_logger,
    configure_event_logger,
    get_event_logger,
)


@pytest.fixture
def reset_global_logger():
    """Reset global logger before and after test."""
    reset_event_logger()
    yield
    reset_event_logger()


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """Create sample assembled messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant.\n\n## Summary\nPrevious discussion about Python."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more."},
    ]


@pytest.fixture
def sample_inputs() -> ContextInputs:
    """Create sample context inputs."""
    return ContextInputs(
        user_turn_text="Tell me more.",
        summary_text="Previous discussion about Python.",
        anchor_exchanges=[],
        recency_exchanges=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ],
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def sample_snapshot(sample_messages, sample_inputs, reset_global_logger) -> ReplaySnapshot:
    """Create a complete sample snapshot with events."""
    # Configure event logger
    output = StringIO()
    configure_event_logger(output=output, run_id="test-snapshot-run")

    # Run validation to emit events
    budget = TokenBudget(full_cap=8000, summary_min=100, overhead_reserve=500)
    result = validate_assembly(
        messages=sample_messages,
        budget=budget,
        assembly_id="test-assembly-1",
        turn_id="test-turn-1",
        emit_event=True,
    )

    # Get emitted events
    events = [e.to_dict() for e in get_event_logger().events]

    # Create snapshot
    snapshot = create_snapshot(
        user_turn_text="Tell me more.",
        assembled_messages=sample_messages,
        token_guard_events=events,
        run_id="test-snapshot-run",
        turn_id="test-turn-1",
        inputs=sample_inputs,
        token_guard_config=TokenGuardConfig(
            full_cap=8000,
            summary_min=100,
            overhead_reserve=500,
            safety_factor=1.2,
        ),
    )

    reset_event_logger()
    return snapshot


class TestSnapshotSchema:
    """Tests for ReplaySnapshot schema."""

    def test_schema_version_present(self, sample_snapshot):
        """Snapshot has schema version."""
        assert sample_snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION

    def test_required_metadata_fields(self, sample_snapshot):
        """Snapshot has all required metadata fields."""
        assert sample_snapshot.run_id is not None
        assert sample_snapshot.turn_id is not None
        assert sample_snapshot.tokenizer_backend_name is not None
        assert sample_snapshot.created_at is not None

    def test_inputs_preserved(self, sample_snapshot, sample_inputs):
        """Context inputs are preserved."""
        assert sample_snapshot.inputs.user_turn_text == sample_inputs.user_turn_text
        assert sample_snapshot.inputs.summary_text == sample_inputs.summary_text

    def test_assembled_messages_preserved(self, sample_snapshot, sample_messages):
        """Assembled messages are preserved."""
        assert sample_snapshot.assembled_messages == sample_messages

    def test_token_guard_events_present(self, sample_snapshot):
        """Token guard events are present."""
        assert len(sample_snapshot.token_guard_events) > 0

    def test_final_event_hash_present(self, sample_snapshot):
        """Final event hash is captured."""
        assert sample_snapshot.final_event_hash is not None

    def test_to_dict_includes_all_fields(self, sample_snapshot):
        """to_dict includes all fields."""
        d = sample_snapshot.to_dict()
        assert "schema_version" in d
        assert "run_id" in d
        assert "turn_id" in d
        assert "inputs" in d
        assert "retrieval" in d
        assert "assembled_messages" in d
        assert "token_guard_config" in d
        assert "token_guard_events" in d

    def test_from_dict_round_trip(self, sample_snapshot):
        """from_dict(to_dict()) preserves data."""
        d = sample_snapshot.to_dict()
        restored = ReplaySnapshot.from_dict(d)

        assert restored.schema_version == sample_snapshot.schema_version
        assert restored.run_id == sample_snapshot.run_id
        assert restored.turn_id == sample_snapshot.turn_id
        assert restored.assembled_messages == sample_snapshot.assembled_messages
        assert len(restored.token_guard_events) == len(sample_snapshot.token_guard_events)

    def test_save_and_load(self, sample_snapshot, tmp_path):
        """Snapshot can be saved and loaded."""
        path = tmp_path / "snapshot.json"
        sample_snapshot.save(path)

        loaded = ReplaySnapshot.load(path)

        assert loaded.schema_version == sample_snapshot.schema_version
        assert loaded.run_id == sample_snapshot.run_id
        assert loaded.assembled_messages == sample_snapshot.assembled_messages


class TestRetrievalState:
    """Tests for RetrievalState."""

    def test_to_dict_round_trip(self):
        """to_dict/from_dict preserves data."""
        state = RetrievalState(
            embedding_model_identifier="test-model",
            query_embedding_vector=[0.1, 0.2, 0.3],
            retrieval_results=[("ex1", 0.95), ("ex2", 0.85)],
            topic_membership_mapping={"topic1": "node1"},
            promoted_topic_ids=["topic1"],
        )

        d = state.to_dict()
        restored = RetrievalState.from_dict(d)

        assert restored.embedding_model_identifier == state.embedding_model_identifier
        assert restored.query_embedding_vector == state.query_embedding_vector
        assert restored.retrieval_results == state.retrieval_results


class TestTokenGuardConfig:
    """Tests for TokenGuardConfig."""

    def test_to_budget_conversion(self):
        """Config converts to TokenBudget correctly."""
        config = TokenGuardConfig(
            full_cap=10000,
            summary_min=200,
            overhead_reserve=600,
        )
        budget = config.to_budget()

        assert budget.full_cap == 10000
        assert budget.summary_min == 200
        assert budget.overhead_reserve == 600


class TestContextInputs:
    """Tests for ContextInputs."""

    def test_to_dict_round_trip(self, sample_inputs):
        """to_dict/from_dict preserves data."""
        d = sample_inputs.to_dict()
        restored = ContextInputs.from_dict(d)

        assert restored.user_turn_text == sample_inputs.user_turn_text
        assert restored.summary_text == sample_inputs.summary_text
        assert restored.recency_exchanges == sample_inputs.recency_exchanges


class TestAssembleFromSnapshot:
    """Tests for assemble_from_snapshot function."""

    def test_assembles_user_message(self, sample_inputs):
        """User message is assembled."""
        snapshot = ReplaySnapshot(inputs=sample_inputs)
        messages = assemble_from_snapshot(snapshot)

        # Find user message
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert user_messages[-1]["content"] == sample_inputs.user_turn_text

    def test_assembles_system_prompt(self, sample_inputs):
        """System prompt is assembled."""
        snapshot = ReplaySnapshot(inputs=sample_inputs)
        messages = assemble_from_snapshot(snapshot)

        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 1

    def test_assembles_recency_exchanges(self, sample_inputs):
        """Recency exchanges are assembled."""
        snapshot = ReplaySnapshot(inputs=sample_inputs)
        messages = assemble_from_snapshot(snapshot)

        # Should have recency exchanges
        contents = [m["content"] for m in messages]
        assert "What is Python?" in contents
        assert "Python is a programming language." in contents


class TestGoldenSnapshotReplay:
    """Golden snapshot test - verify deterministic replay."""

    def test_replay_produces_identical_messages(self, sample_snapshot, tmp_path, reset_global_logger):
        """Replay produces identical messages."""
        # Save snapshot
        path = tmp_path / "golden.json"
        sample_snapshot.save(path)

        # Replay with stored messages
        result = replay(path, use_stored_messages=True)

        assert result.success
        assert result.messages_match
        assert result.replayed_messages == sample_snapshot.assembled_messages

    def test_replay_verifies_token_counts(self, sample_snapshot, tmp_path, reset_global_logger):
        """Replay verifies token counts match."""
        path = tmp_path / "golden.json"
        sample_snapshot.save(path)

        result = replay(path, use_stored_messages=True)

        assert result.success
        assert result.tokens_match
        assert result.replayed_token_count > 0

    def test_replay_verifies_hash_chain(self, sample_snapshot, tmp_path, reset_global_logger):
        """Replay verifies hash chain integrity."""
        path = tmp_path / "golden.json"
        sample_snapshot.save(path)

        result = replay(path, use_stored_messages=True)

        assert result.success
        assert result.hash_chain_valid

    def test_replay_reports_counter_backend(self, sample_snapshot, tmp_path, reset_global_logger):
        """Replay reports which counter backend was used."""
        path = tmp_path / "golden.json"
        sample_snapshot.save(path)

        result = replay(path, use_stored_messages=True)

        assert result.counter_backend_used != ""


class TestMutationDetection:
    """Mutation tests - verify divergence detection."""

    def test_detects_message_content_mutation(self, sample_snapshot, tmp_path, reset_global_logger):
        """Detects mutation in message content."""
        # Mutate a message
        mutated = copy.deepcopy(sample_snapshot)
        mutated.assembled_messages[1]["content"] = "MUTATED CONTENT"

        path = tmp_path / "mutated.json"
        mutated.save(path)

        # Replay with re-assembly (which will use original inputs)
        result = replay(path, use_stored_messages=False)

        # Should detect divergence
        assert not result.messages_match
        assert result.first_diff is not None
        assert "content" in result.first_diff.field_path

    def test_detects_message_count_mutation(self, sample_snapshot, tmp_path, reset_global_logger):
        """Detects mutation in message count."""
        mutated = copy.deepcopy(sample_snapshot)
        mutated.assembled_messages.append({"role": "user", "content": "Extra message"})

        path = tmp_path / "mutated.json"
        mutated.save(path)

        result = replay(path, use_stored_messages=False)

        assert not result.messages_match
        assert result.first_diff is not None
        assert "length" in result.first_diff.field_path

    def test_detects_hash_chain_mutation(self, sample_snapshot, tmp_path, reset_global_logger):
        """Detects mutation in hash chain."""
        mutated = copy.deepcopy(sample_snapshot)

        # Mutate an event's hash
        if mutated.token_guard_events:
            mutated.token_guard_events[0]["hash"] = "invalid_hash"

        path = tmp_path / "mutated.json"
        mutated.save(path)

        result = replay(path, use_stored_messages=True)

        assert not result.hash_chain_valid
        assert any("hash_chain" in d.field_path for d in result.all_diffs)

    def test_detects_event_field_mutation(self, sample_snapshot, tmp_path, reset_global_logger):
        """Detects mutation in event fields."""
        mutated = copy.deepcopy(sample_snapshot)

        # Mutate an event field
        if mutated.token_guard_events:
            mutated.token_guard_events[0]["raw_tokens"] = 999999

        path = tmp_path / "mutated.json"
        mutated.save(path)

        result = replay(path, use_stored_messages=True)

        # Hash chain should be invalid due to content change
        assert not result.hash_chain_valid

    def test_diff_points_to_correct_field(self, sample_snapshot, tmp_path, reset_global_logger):
        """Diff correctly identifies the mutated field."""
        mutated = copy.deepcopy(sample_snapshot)
        mutated.assembled_messages[2]["role"] = "system"  # Change role

        path = tmp_path / "mutated.json"
        mutated.save(path)

        result = replay(path, use_stored_messages=False)

        assert result.first_diff is not None
        assert "assembled_messages[2]" in result.first_diff.field_path
        assert "role" in result.first_diff.field_path


class TestHashChainIntegrity:
    """Tests for hash chain verification through replay."""

    def test_valid_chain_passes_verification(self, sample_snapshot):
        """Valid event chain passes verification."""
        verification = EventVerifier.verify_stream(sample_snapshot.token_guard_events)
        assert verification["valid"]

    def test_mutated_event_content_fails_verification(self, sample_snapshot):
        """Mutating event content breaks hash chain."""
        events = copy.deepcopy(sample_snapshot.token_guard_events)

        if events:
            events[0]["raw_tokens"] = 888888

        verification = EventVerifier.verify_stream(events)
        assert not verification["valid"]

    def test_reordered_events_fail_verification(self, sample_snapshot, reset_global_logger):
        """Create snapshot with multiple events and verify reordering fails."""
        # Create a snapshot with multiple events
        output = StringIO()
        configure_event_logger(output=output, run_id="multi-event-run")

        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "World"}]

        budget = TokenBudget(full_cap=8000)

        validate_assembly(messages1, budget, assembly_id="a1", emit_event=True)
        validate_assembly(messages2, budget, assembly_id="a2", emit_event=True)

        events = [e.to_dict() for e in get_event_logger().events]
        reset_event_logger()

        # Original should verify
        assert EventVerifier.verify_stream(events)["valid"]

        # Swap events
        if len(events) >= 2:
            swapped = [events[1], events[0]]
            verification = EventVerifier.verify_stream(swapped)
            assert not verification["valid"]


class TestMessageComparison:
    """Tests for message comparison logic."""

    def test_identical_messages_match(self):
        """Identical messages have no diffs."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        diffs = _compare_messages(msgs, msgs)
        assert len(diffs) == 0

    def test_different_content_detected(self):
        """Different content is detected."""
        expected = [{"role": "user", "content": "Hello"}]
        actual = [{"role": "user", "content": "Goodbye"}]

        diffs = _compare_messages(expected, actual)
        assert len(diffs) == 1
        assert "content" in diffs[0].field_path

    def test_different_role_detected(self):
        """Different role is detected."""
        expected = [{"role": "user", "content": "Hello"}]
        actual = [{"role": "assistant", "content": "Hello"}]

        diffs = _compare_messages(expected, actual)
        assert len(diffs) == 1
        assert "role" in diffs[0].field_path

    def test_different_length_detected(self):
        """Different message count is detected."""
        expected = [{"role": "user", "content": "Hello"}]
        actual = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        diffs = _compare_messages(expected, actual)
        assert len(diffs) == 1
        assert "length" in diffs[0].field_path


class TestEventComparison:
    """Tests for event comparison logic."""

    def test_identical_events_match(self):
        """Identical events have no diffs."""
        evt = {
            "schema_version": "1.0",
            "event_type": "token_ok",
            "event_seq": 1,
            "counter_backend": "heuristic",
            "counter_exact": False,
            "raw_tokens": 100,
            "effective_tokens": 120,
            "cap": 8000,
        }
        diffs = _compare_events([evt], [evt])
        assert len(diffs) == 0

    def test_different_field_detected(self):
        """Different field value is detected."""
        expected = [{"schema_version": "1.0", "event_type": "token_ok", "raw_tokens": 100}]
        actual = [{"schema_version": "1.0", "event_type": "token_ok", "raw_tokens": 200}]

        diffs = _compare_events(expected, actual)
        assert len(diffs) == 1
        assert "raw_tokens" in diffs[0].field_path


class TestReplayResult:
    """Tests for ReplayResult."""

    def test_success_requires_all_matches(self, sample_snapshot, tmp_path, reset_global_logger):
        """Success is True only when all checks pass."""
        path = tmp_path / "golden.json"
        sample_snapshot.save(path)

        result = replay(path, use_stored_messages=True)

        if result.success:
            assert result.messages_match
            assert result.tokens_match
            assert result.events_match
            assert result.hash_chain_valid

    def test_to_dict_serializable(self, sample_snapshot, tmp_path, reset_global_logger):
        """Result can be serialized to dict."""
        path = tmp_path / "golden.json"
        sample_snapshot.save(path)

        result = replay(path, use_stored_messages=True)
        d = result.to_dict()

        assert "success" in d
        assert "messages_match" in d
        assert "tokens_match" in d
        assert "events_match" in d
        assert "hash_chain_valid" in d


class TestCreateSnapshot:
    """Tests for create_snapshot helper."""

    def test_creates_valid_snapshot(self, reset_global_logger):
        """create_snapshot creates a valid snapshot."""
        messages = [{"role": "user", "content": "Hello"}]
        events = []  # Empty events for simple test

        snapshot = create_snapshot(
            user_turn_text="Hello",
            assembled_messages=messages,
            token_guard_events=events,
        )

        assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION
        assert snapshot.run_id is not None
        assert snapshot.turn_id is not None
        assert snapshot.inputs.user_turn_text == "Hello"
        assert snapshot.assembled_messages == messages

    def test_captures_final_hash(self, reset_global_logger):
        """create_snapshot captures final event hash."""
        output = StringIO()
        configure_event_logger(output=output, run_id="capture-hash-run")

        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=8000)

        validate_assembly(messages, budget, assembly_id="cap-hash-1", emit_event=True)
        events = [e.to_dict() for e in get_event_logger().events]

        snapshot = create_snapshot(
            user_turn_text="Hello",
            assembled_messages=messages,
            token_guard_events=events,
        )

        reset_event_logger()

        assert snapshot.final_event_hash is not None
        assert snapshot.final_event_hash == events[-1]["hash"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
