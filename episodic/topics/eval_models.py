"""
Data models for topic evaluation.

Boundary alignment, test case definitions, evaluation result classes,
and major boundary detection heuristics.
"""

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set

from episodic.topics.strategy import TopicStrategy, Confidence


# ============================================================================
# CANONICAL BOUNDARY REPRESENTATION
# ============================================================================

@dataclass
class BoundaryAlignment:
    """
    Configuration for how a dataset labels topic boundaries.

    Canonical representation: A boundary at index t means "boundary between
    message t-1 and message t" (i.e., the topic changes AT message t).
    Valid boundary indices for T messages: [1, T-1].

    Different datasets label boundaries differently:
    - "message": boundary labeled on a specific message
    - "between": boundary labeled between messages (already canonical)
    - "turn_pair": boundary after a user+assistant pair

    The anchor specifies interpretation:
    - "after": label on message i means boundary AFTER message i (canonical: i+1)
    - "before": label on message i means boundary BEFORE message i (canonical: i)

    The speaker filter restricts which messages can have boundaries:
    - None: any message
    - "user": only user messages
    - "assistant": only assistant messages
    """
    label_type: str = "message"  # "message", "between", "turn_pair"
    anchor: str = "before"  # "after" or "before"
    speaker: Optional[str] = None  # None, "user", or "assistant"

    def __post_init__(self):
        valid_types = {"message", "between", "turn_pair"}
        valid_anchors = {"after", "before"}
        valid_speakers = {None, "user", "assistant"}

        if self.label_type not in valid_types:
            raise ValueError(f"label_type must be one of {valid_types}, got {self.label_type}")
        if self.anchor not in valid_anchors:
            raise ValueError(f"anchor must be one of {valid_anchors}, got {self.anchor}")
        if self.speaker not in valid_speakers:
            raise ValueError(f"speaker must be one of {valid_speakers}, got {self.speaker}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label_type': self.label_type,
            'anchor': self.anchor,
            'speaker': self.speaker,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundaryAlignment':
        return cls(
            label_type=data.get('label_type', 'message'),
            anchor=data.get('anchor', 'before'),
            speaker=data.get('speaker'),
        )


# Common alignment presets
ALIGNMENT_PRESETS = {
    # Boundary labeled on user message that starts new topic
    'user_starts_topic': BoundaryAlignment(
        label_type='message',
        anchor='before',
        speaker='user'
    ),
    # Boundary labeled on assistant message that starts new topic
    'assistant_starts_topic': BoundaryAlignment(
        label_type='message',
        anchor='before',
        speaker='assistant'
    ),
    # Boundary labeled on message AFTER which topic changes
    'after_message': BoundaryAlignment(
        label_type='message',
        anchor='after',
        speaker=None
    ),
    # Already canonical (between messages)
    'canonical': BoundaryAlignment(
        label_type='between',
        anchor='before',
        speaker=None
    ),
    # SuperDialseg/TIAGE style: boundary on the message that starts new segment
    'segment_start': BoundaryAlignment(
        label_type='message',
        anchor='before',
        speaker=None
    ),
}


def to_canonical_boundaries(
    boundaries: List[int],
    messages: List[Any],
    alignment: BoundaryAlignment
) -> Set[int]:
    """
    Convert boundaries from dataset format to canonical representation.

    Canonical: boundary at t means "topic changes AT message t"
    (between message t-1 and message t).

    Args:
        boundaries: List of boundary indices in dataset format
        messages: List of messages (need role info for speaker filtering)
        alignment: How the dataset labels boundaries

    Returns:
        Set of canonical boundary indices
    """
    num_messages = len(messages)
    canonical = set()

    for b in boundaries:
        if b < 0 or b >= num_messages:
            continue  # Skip out-of-range

        # Get the message role at this position
        msg = messages[b]
        role = msg.role if hasattr(msg, 'role') else msg.get('role', 'user')

        # Check speaker filter
        if alignment.speaker and role != alignment.speaker:
            # This boundary is on wrong speaker type - might need adjustment
            # For now, still include it but log warning
            pass

        if alignment.label_type == 'between':
            # Already canonical
            canonical_idx = b
        elif alignment.label_type == 'message':
            if alignment.anchor == 'before':
                # Boundary at message b means topic starts at b
                canonical_idx = b
            else:  # anchor == 'after'
                # Boundary after message b means topic starts at b+1
                canonical_idx = b + 1
        elif alignment.label_type == 'turn_pair':
            # Boundary after turn pair - find next user message
            canonical_idx = b + 1
        else:
            canonical_idx = b

        # Validate range: canonical boundaries are in [1, T-1]
        if 1 <= canonical_idx < num_messages:
            canonical.add(canonical_idx)

    return canonical


def from_canonical_boundaries(
    canonical_boundaries: Set[int],
    messages: List[Any],
    alignment: BoundaryAlignment
) -> List[int]:
    """
    Convert canonical boundaries back to dataset format.

    Useful for comparing predictions in the format expected by a dataset.

    Args:
        canonical_boundaries: Set of canonical boundary indices
        messages: List of messages
        alignment: Target dataset format

    Returns:
        List of boundary indices in dataset format
    """
    result = []
    num_messages = len(messages)

    for c in sorted(canonical_boundaries):
        if alignment.label_type == 'between':
            idx = c
        elif alignment.label_type == 'message':
            if alignment.anchor == 'before':
                idx = c
            else:  # anchor == 'after'
                idx = c - 1
        elif alignment.label_type == 'turn_pair':
            idx = c - 1
        else:
            idx = c

        # Apply speaker filter if needed
        if alignment.speaker and 0 <= idx < num_messages:
            msg = messages[idx]
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'user')
            if role != alignment.speaker:
                # Find nearest message with correct speaker
                # For now, just use the index anyway
                pass

        if 0 <= idx < num_messages:
            result.append(idx)

    return result


def normalize_strategy_output(
    predictions: List[int],
    messages: List[Any],
    strategy_alignment: BoundaryAlignment = None
) -> Set[int]:
    """
    Normalize strategy predictions to canonical boundary indices.

    Strategies typically detect boundaries at user messages where they
    decide a topic change occurred. This converts to canonical form.

    Args:
        predictions: List of message indices where strategy detected boundaries
        messages: List of messages
        strategy_alignment: How the strategy reports boundaries
                           (default: user_starts_topic)

    Returns:
        Set of canonical boundary indices
    """
    if strategy_alignment is None:
        # Default: strategies detect on user messages, boundary means topic starts
        strategy_alignment = ALIGNMENT_PRESETS['user_starts_topic']

    return to_canonical_boundaries(predictions, messages, strategy_alignment)


# ============================================================================
# TEST CASE AND EVALUATION CLASSES
# ============================================================================


@dataclass
class Message:
    """A single message in a test conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'node_id': self.node_id,
            **self.metadata
        }


@dataclass
class EvalCase:
    """
    A labeled test case for evaluating topic detection.

    Contains a conversation with labeled topic boundaries and
    expected retrieval behavior.

    Boundary alignment specifies how expected_boundaries are labeled
    in the source dataset. Use get_canonical_boundaries() to convert
    to the canonical representation for evaluation.
    """
    id: str
    name: str
    description: str
    messages: List[Message]

    # Expected topic boundaries (indices in dataset's format)
    expected_boundaries: List[int]

    # How boundaries are labeled in this test case
    # Default: boundaries mark message where new topic starts (canonical)
    boundary_alignment: BoundaryAlignment = field(
        default_factory=lambda: ALIGNMENT_PRESETS['segment_start']
    )

    # Expected topic names at each boundary (optional)
    expected_topic_names: Dict[int, str] = field(default_factory=dict)

    # For retrieval testing: query and expected thread to retrieve
    retrieval_tests: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "synthetic"  # synthetic, real, imported

    def get_canonical_boundaries(self) -> Set[int]:
        """
        Convert expected_boundaries to canonical representation.

        Canonical: boundary at t means topic changes AT message t
        (between message t-1 and message t).

        Returns:
            Set of canonical boundary indices
        """
        return to_canonical_boundaries(
            self.expected_boundaries,
            self.messages,
            self.boundary_alignment
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'messages': [m.to_dict() for m in self.messages],
            'expected_boundaries': self.expected_boundaries,
            'boundary_alignment': self.boundary_alignment.to_dict(),
            'expected_topic_names': self.expected_topic_names,
            'retrieval_tests': self.retrieval_tests,
            'tags': self.tags,
            'difficulty': self.difficulty,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalCase':
        messages = [
            Message(
                role=m['role'],
                content=m['content'],
                node_id=m.get('node_id'),
                metadata={k: v for k, v in m.items() if k not in ['role', 'content', 'node_id']}
            )
            for m in data['messages']
        ]

        # Parse boundary alignment if present
        alignment_data = data.get('boundary_alignment')
        if alignment_data:
            boundary_alignment = BoundaryAlignment.from_dict(alignment_data)
        else:
            # Default to segment_start for backwards compatibility
            boundary_alignment = ALIGNMENT_PRESETS['segment_start']

        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            messages=messages,
            expected_boundaries=data['expected_boundaries'],
            boundary_alignment=boundary_alignment,
            expected_topic_names=data.get('expected_topic_names', {}),
            retrieval_tests=data.get('retrieval_tests', []),
            tags=data.get('tags', []),
            difficulty=data.get('difficulty', 'medium'),
            source=data.get('source', 'synthetic'),
        )


@dataclass
class BoundaryResult:
    """Result of boundary detection for a single message."""
    message_index: int
    expected_boundary: bool
    detected_boundary: bool
    confidence: Confidence
    confidence_score: float
    processing_time_ms: float
    signals: Dict[str, float]

    @property
    def is_correct(self) -> bool:
        return self.expected_boundary == self.detected_boundary

    @property
    def is_true_positive(self) -> bool:
        return self.expected_boundary and self.detected_boundary

    @property
    def is_false_positive(self) -> bool:
        return not self.expected_boundary and self.detected_boundary

    @property
    def is_true_negative(self) -> bool:
        return not self.expected_boundary and not self.detected_boundary

    @property
    def is_false_negative(self) -> bool:
        return self.expected_boundary and not self.detected_boundary


@dataclass
class EvaluationResult:
    """Result of evaluating a strategy on a test case."""
    test_case_id: str
    strategy_name: str
    strategy_version: str
    boundary_results: List[BoundaryResult]
    total_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Computed metrics
    @property
    def true_positives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_true_positive)

    @property
    def false_positives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_false_positive)

    @property
    def true_negatives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_true_negative)

    @property
    def false_negatives(self) -> int:
        return sum(1 for r in self.boundary_results if r.is_false_negative)

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)"""
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / total"""
        total = len(self.boundary_results)
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_case_id': self.test_case_id,
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'total_time_ms': self.total_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'metrics': {
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'accuracy': self.accuracy,
            },
            'boundary_results': [
                {
                    'message_index': r.message_index,
                    'expected': r.expected_boundary,
                    'detected': r.detected_boundary,
                    'correct': r.is_correct,
                    'confidence': r.confidence.value,
                    'confidence_score': r.confidence_score,
                    'processing_time_ms': r.processing_time_ms,
                }
                for r in self.boundary_results
            ]
        }


# ============================================================================
# MAJOR BOUNDARY DETECTION HEURISTICS
# ============================================================================

MAJOR_BOUNDARY_PATTERNS = [
    r'\b(by the way|btw)\b',
    r'\b(anyway|anyhow)\b',
    r'\b(on (a |an )?other (note|topic|subject))\b',
    r'\b(changing (topic|subject|gears))\b',
    r'\b(new question|different question)\b',
    r'\b(moving on|let\'?s move on)\b',
    r'\b(back to|getting back to)\b',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MAJOR_BOUNDARY_PATTERNS]


def is_likely_major_boundary(
    message_content: str,
    prev_messages: List[Dict[str, Any]],
    semantic_distance: Optional[float] = None
) -> bool:
    """
    Heuristically determine if a message is likely a major topic boundary.

    Major boundaries are high-cost transitions that should not be missed.

    Heuristics:
    1. Explicit transition markers ("by the way", "new question", etc.)
    2. High semantic distance from recent context (top 20th percentile)
    3. Return to previously discussed entity/keyword

    Args:
        message_content: The message text
        prev_messages: Previous messages for context
        semantic_distance: Optional pre-computed semantic distance (0-1)

    Returns:
        True if this is likely a major boundary
    """
    content_lower = message_content.lower()

    # Check explicit markers
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(content_lower):
            return True

    # Check semantic distance (if provided)
    if semantic_distance is not None and semantic_distance > 0.6:
        return True

    # Check for question after long assistant response (often indicates new topic)
    if prev_messages and len(prev_messages) >= 2:
        last_msg = prev_messages[-1]
        if last_msg.get('role') == 'assistant':
            last_content = last_msg.get('content', '')
            if len(last_content) > 500 and content_lower.rstrip().endswith('?'):
                return True

    return False
