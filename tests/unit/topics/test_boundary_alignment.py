"""
Unit tests for canonical boundary alignment.

These tests ensure that boundary evaluation correctly handles different
dataset labeling conventions and prevents off-by-one errors.

Test coverage:
1. Off-by-one reproduction: Verify that predictions offset by one message
   score correctly with canonical alignment
2. Speaker-specific mapping: Verify alignment presets produce correct indices
3. W-F1 invariance: Same logical boundaries should produce same W-F1
   regardless of labeling convention
"""

import pytest
from episodic.topics.evaluation import (
    BoundaryAlignment,
    ALIGNMENT_PRESETS,
    to_canonical_boundaries,
    from_canonical_boundaries,
    normalize_strategy_output,
    compute_windowed_metrics,
    EvalCase,
    Message,
    EvaluationHarness,
    EvaluationResult,
)
from episodic.topics.strategy import TopicDecision, Confidence


class MockStrategy:
    """Mock strategy for testing that returns predetermined boundaries."""

    def __init__(self, boundary_indices: list):
        self.name = "MockStrategy"
        self.version = "1.0.0"
        self.boundary_indices = set(boundary_indices)

    def get_decision(self, query: str, messages: list, current_thread=None):
        # Determine current message index
        # This is a simplified mock - real strategies track state
        idx = len(messages)  # Current message would be at this index
        detected = idx in self.boundary_indices

        return TopicDecision(
            topic_changed=detected,
            confidence=Confidence.HIGH if detected else Confidence.UNCERTAIN,
            confidence_score=0.9 if detected else 0.1,
            signals={},
            strategy_name=self.name
        )


class TestBoundaryAlignmentBasics:
    """Test BoundaryAlignment dataclass and validation."""

    def test_valid_alignment_creation(self):
        """BoundaryAlignment accepts valid parameters."""
        align = BoundaryAlignment(
            label_type='message',
            anchor='before',
            speaker='user'
        )
        assert align.label_type == 'message'
        assert align.anchor == 'before'
        assert align.speaker == 'user'

    def test_invalid_label_type_raises(self):
        """Invalid label_type raises ValueError."""
        with pytest.raises(ValueError, match="label_type must be one of"):
            BoundaryAlignment(label_type='invalid')

    def test_invalid_anchor_raises(self):
        """Invalid anchor raises ValueError."""
        with pytest.raises(ValueError, match="anchor must be one of"):
            BoundaryAlignment(anchor='middle')

    def test_invalid_speaker_raises(self):
        """Invalid speaker raises ValueError."""
        with pytest.raises(ValueError, match="speaker must be one of"):
            BoundaryAlignment(speaker='bot')

    def test_presets_are_valid(self):
        """All preset alignments are properly configured."""
        for name, preset in ALIGNMENT_PRESETS.items():
            assert preset.label_type in {'message', 'between', 'turn_pair'}
            assert preset.anchor in {'before', 'after'}
            assert preset.speaker in {None, 'user', 'assistant'}


class TestCanonicalBoundaryConversion:
    """Test to_canonical_boundaries and from_canonical_boundaries."""

    @pytest.fixture
    def sample_messages(self):
        """Standard 4-message dialogue: U0, A1, U2, A3."""
        return [
            Message(role='user', content='Hello'),           # 0
            Message(role='assistant', content='Hi there'),    # 1
            Message(role='user', content='New topic'),        # 2
            Message(role='assistant', content='About that'),  # 3
        ]

    def test_segment_start_assistant_boundary(self, sample_messages):
        """Boundary labeled on assistant message converts to same index."""
        # Boundary at assistant message 3 (topic starts at message 3)
        boundaries = [3]
        alignment = ALIGNMENT_PRESETS['segment_start']

        canonical = to_canonical_boundaries(boundaries, sample_messages, alignment)

        assert canonical == {3}

    def test_segment_start_user_boundary(self, sample_messages):
        """Boundary labeled on user message converts to same index."""
        # Boundary at user message 2 (topic starts at message 2)
        boundaries = [2]
        alignment = ALIGNMENT_PRESETS['segment_start']

        canonical = to_canonical_boundaries(boundaries, sample_messages, alignment)

        assert canonical == {2}

    def test_after_message_anchor(self, sample_messages):
        """anchor='after' shifts boundary index by +1."""
        # Boundary AFTER message 1 means topic starts at message 2
        boundaries = [1]
        alignment = BoundaryAlignment(
            label_type='message',
            anchor='after',
            speaker=None
        )

        canonical = to_canonical_boundaries(boundaries, sample_messages, alignment)

        assert canonical == {2}

    def test_between_type_is_identity(self, sample_messages):
        """label_type='between' is already canonical."""
        boundaries = [2, 3]
        alignment = ALIGNMENT_PRESETS['canonical']

        canonical = to_canonical_boundaries(boundaries, sample_messages, alignment)

        assert canonical == {2, 3}

    def test_out_of_range_boundaries_filtered(self, sample_messages):
        """Boundaries outside [1, T-1] are filtered out."""
        boundaries = [0, 2, 10]  # 0 and 10 are invalid
        alignment = ALIGNMENT_PRESETS['segment_start']

        canonical = to_canonical_boundaries(boundaries, sample_messages, alignment)

        # Only 2 is valid (in range [1, 3])
        assert canonical == {2}

    def test_roundtrip_conversion(self, sample_messages):
        """Converting to canonical and back preserves boundaries."""
        original = [2, 3]
        alignment = ALIGNMENT_PRESETS['segment_start']

        canonical = to_canonical_boundaries(original, sample_messages, alignment)
        recovered = from_canonical_boundaries(canonical, sample_messages, alignment)

        assert set(recovered) == set(original)


class TestOffByOneDetection:
    """
    Test 1: Off-by-one reproduction.

    This is the critical test that verifies the fix for the original bug.
    When gold boundaries are on assistant turns and predictions are on
    user turns (one position later), the evaluation should:
    - Score 0 with exact matching
    - Score > 0 with windowed matching (W-F1)
    """

    @pytest.fixture
    def offset_dialogue(self):
        """Dialogue where gold is on A and predictions are on next U."""
        return [
            Message(role='user', content='Start'),           # 0
            Message(role='assistant', content='Reply'),       # 1
            Message(role='user', content='Question'),         # 2
            Message(role='assistant', content='NEW TOPIC'),   # 3 - gold boundary
            Message(role='user', content='Follow up'),        # 4 - prediction here
            Message(role='assistant', content='More info'),   # 5
        ]

    def test_exact_match_is_zero_for_offset(self, offset_dialogue):
        """Exact F1 is 0 when prediction is off by one."""
        gold_boundaries = {3}  # On assistant
        pred_boundaries = {4}  # On user (off by one)

        # With exact matching, no overlap
        tp = len(gold_boundaries & pred_boundaries)
        assert tp == 0

    def test_windowed_match_captures_offset(self, offset_dialogue):
        """W-F1 with window=1 captures off-by-one predictions."""
        gold_boundaries = {3}
        pred_boundaries = {4}

        precision, recall, f1 = compute_windowed_metrics(
            gold_boundaries, pred_boundaries,
            num_messages=len(offset_dialogue),
            window=1
        )

        # Both should be 1.0 since prediction at 4 is within window of gold at 3
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_canonical_alignment_with_strategy_offset(self, offset_dialogue):
        """
        Full integration test: gold on assistant, strategy detects on user.

        This is the scenario that originally produced zero scores.
        With canonical alignment, W-F1 should be meaningful.
        """
        # Gold boundaries on assistant turns
        gold_raw = [3]
        gold_alignment = ALIGNMENT_PRESETS['segment_start']

        # Strategy predicts on user turns (one after gold)
        pred_raw = [4]
        strategy_alignment = ALIGNMENT_PRESETS['user_starts_topic']

        # Convert both to canonical
        gold_canonical = to_canonical_boundaries(
            gold_raw, offset_dialogue, gold_alignment
        )
        pred_canonical = normalize_strategy_output(
            pred_raw, offset_dialogue, strategy_alignment
        )

        # Both canonical representations
        assert gold_canonical == {3}
        assert pred_canonical == {4}

        # W-F1 should show they match within tolerance
        _, _, f1 = compute_windowed_metrics(
            gold_canonical, pred_canonical,
            num_messages=len(offset_dialogue),
            window=1
        )
        assert f1 == 1.0


class TestSpeakerSpecificMapping:
    """
    Test 2: Speaker-specific mapping correctness.

    Verify that different alignment modes produce expected canonical indices.
    """

    @pytest.fixture
    def mixed_dialogue(self):
        """Dialogue with various speaker patterns."""
        return [
            Message(role='user', content='U0'),        # 0
            Message(role='assistant', content='A1'),   # 1
            Message(role='user', content='U2'),        # 2
            Message(role='assistant', content='A3'),   # 3
            Message(role='user', content='U4'),        # 4
            Message(role='assistant', content='A5'),   # 5
        ]

    def test_user_starts_topic_alignment(self, mixed_dialogue):
        """user_starts_topic maps user message indices directly."""
        boundaries = [2, 4]  # User messages
        alignment = ALIGNMENT_PRESETS['user_starts_topic']

        canonical = to_canonical_boundaries(boundaries, mixed_dialogue, alignment)

        assert canonical == {2, 4}

    def test_assistant_starts_topic_alignment(self, mixed_dialogue):
        """assistant_starts_topic maps assistant message indices directly."""
        boundaries = [1, 3, 5]  # Assistant messages
        alignment = ALIGNMENT_PRESETS['assistant_starts_topic']

        canonical = to_canonical_boundaries(boundaries, mixed_dialogue, alignment)

        assert canonical == {1, 3, 5}

    def test_after_message_shifts_indices(self, mixed_dialogue):
        """after_message alignment shifts all indices by +1."""
        # Boundary "after" message 1 -> canonical 2
        # Boundary "after" message 3 -> canonical 4
        boundaries = [1, 3]
        alignment = ALIGNMENT_PRESETS['after_message']

        canonical = to_canonical_boundaries(boundaries, mixed_dialogue, alignment)

        assert canonical == {2, 4}


class TestWF1Invariance:
    """
    Test 3: W-F1 invariance under label convention.

    The same logical boundaries should produce the same W-F1 regardless
    of whether they're labeled on assistant or user messages.
    """

    @pytest.fixture
    def conversation(self):
        """Standard alternating dialogue."""
        return [
            Message(role='user', content='U0'),
            Message(role='assistant', content='A1'),
            Message(role='user', content='U2'),  # Topic 2 starts
            Message(role='assistant', content='A3'),
            Message(role='user', content='U4'),
            Message(role='assistant', content='A5'),  # Topic 3 starts
            Message(role='user', content='U6'),
            Message(role='assistant', content='A7'),
        ]

    def test_same_boundaries_different_conventions(self, conversation):
        """
        Same logical boundaries labeled differently should produce same W-F1.

        Scenario: Topics change at positions 2 and 5.
        - Convention A: Label on message that starts topic (2, 5)
        - Convention B: Label on message after which topic changes (1, 4)

        Both represent the same segmentation.
        """
        # Convention A: segment_start (label on first message of new topic)
        gold_a = [2, 5]
        align_a = ALIGNMENT_PRESETS['segment_start']

        # Convention B: after_message (label on last message of old topic)
        gold_b = [1, 4]  # After 1 -> canonical 2, after 4 -> canonical 5
        align_b = ALIGNMENT_PRESETS['after_message']

        # Convert to canonical
        canonical_a = to_canonical_boundaries(gold_a, conversation, align_a)
        canonical_b = to_canonical_boundaries(gold_b, conversation, align_b)

        # Should produce identical canonical sets
        assert canonical_a == canonical_b == {2, 5}

        # Therefore W-F1 against same predictions should be identical
        predictions = {2, 5}  # Perfect predictions

        _, _, f1_a = compute_windowed_metrics(
            canonical_a, predictions, len(conversation), window=1
        )
        _, _, f1_b = compute_windowed_metrics(
            canonical_b, predictions, len(conversation), window=1
        )

        assert f1_a == f1_b == 1.0


class TestEvalCaseAlignment:
    """Test EvalCase integration with boundary alignment."""

    def test_testcase_stores_alignment(self):
        """EvalCase stores and uses boundary alignment."""
        messages = [
            Message(role='user', content='Hello'),
            Message(role='assistant', content='Hi'),
            Message(role='user', content='Bye'),
        ]
        alignment = ALIGNMENT_PRESETS['assistant_starts_topic']

        tc = EvalCase(
            id='test1',
            name='Test Case',
            description='Test',
            messages=messages,
            expected_boundaries=[1],  # Assistant message
            boundary_alignment=alignment
        )

        assert tc.boundary_alignment == alignment
        assert tc.get_canonical_boundaries() == {1}

    def test_testcase_default_alignment(self):
        """EvalCase defaults to segment_start alignment."""
        messages = [
            Message(role='user', content='Hello'),
            Message(role='assistant', content='Hi'),
        ]

        tc = EvalCase(
            id='test1',
            name='Test',
            description='Test',
            messages=messages,
            expected_boundaries=[1]
        )

        # Default is segment_start
        assert tc.boundary_alignment.label_type == 'message'
        assert tc.boundary_alignment.anchor == 'before'

    def test_testcase_serialization_preserves_alignment(self):
        """EvalCase to_dict/from_dict preserves alignment."""
        messages = [
            Message(role='user', content='Hello'),
            Message(role='assistant', content='Hi'),
        ]
        alignment = BoundaryAlignment(
            label_type='message',
            anchor='after',
            speaker='assistant'
        )

        original = EvalCase(
            id='test1',
            name='Test',
            description='Test',
            messages=messages,
            expected_boundaries=[0],
            boundary_alignment=alignment
        )

        # Serialize and deserialize
        data = original.to_dict()
        recovered = EvalCase.from_dict(data)

        assert recovered.boundary_alignment.label_type == 'message'
        assert recovered.boundary_alignment.anchor == 'after'
        assert recovered.boundary_alignment.speaker == 'assistant'
