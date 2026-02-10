"""
Online evaluation harness for topic detection strategies.

Replays dialogues turn-by-turn, tracking:
- State machine transitions (STABLE/SUSPECT)
- Frozen reference context
- Evidence accumulation
- Commit/abort events

This complements the batch EvaluationHarness in evaluation.py by
preserving the temporal dynamics of online topic detection.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from episodic.topics.strategy import TopicStrategy
from episodic.topics.evaluation import (
    EvalCase,
    BoundaryAlignment,
    ALIGNMENT_PRESETS,
    normalize_strategy_output,
)

logger = logging.getLogger(__name__)


@dataclass
class TurnTrace:
    """
    Trace of a single turn during online replay.

    Captures signals, state machine state, and events for analysis.
    """
    turn_idx: int
    node_id: str                          # Unique message identifier
    role: str
    content_preview: str                  # First 50 chars for debugging

    # Signals
    semantic_drift: Optional[float]       # Embedding drift from previous user msg
    neural_confidence: float              # Raw confidence from neural model

    # State machine state (from CommitmentPolicyStrategy._state)
    state: str                            # "STABLE" | "SUSPECT"
    suspect_entry_turn: Optional[int]     # Turn index when SUSPECT was entered
    suspect_entry_node_id: Optional[str]  # Node ID when entering SUSPECT
    frozen_before_ids: List[str]          # Node IDs of frozen before context
    frozen_straddle_id: Optional[str]     # Node ID of frozen straddle message
    accumulated_evidence: float           # Evidence accumulated so far

    # Events
    event: Optional[str]                  # "COMMIT" | "ABORT" | None
    # Canonical boundary: between-message index (matches evaluation.py gold format)
    # boundary_idx=5 means boundary is BEFORE message at index 5
    canonical_boundary_idx: Optional[int]
    commit_node_id: Optional[str]         # Node ID at COMMIT event (backdated)

    processing_time_ms: float

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class DialogueTrace:
    """
    Complete trace of replaying a dialogue.

    Contains per-turn traces and aggregate boundary information.
    """
    dialogue_id: str
    turns: List[TurnTrace]
    gold_boundaries: Set[int]             # Canonical boundary indices
    predicted_boundaries: Set[int]        # Canonical boundary indices

    # Events for timing/churn analysis: (suspect_entry_turn, commit_turn, cause)
    commit_events: List[Tuple[int, int, str]] = field(default_factory=list)
    # (suspect_entry_turn, abort_turn, abort_reason)
    abort_events: List[Tuple[int, int, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            'dialogue_id': self.dialogue_id,
            'turns': [t.to_dict() for t in self.turns],
            'gold_boundaries': list(self.gold_boundaries),
            'predicted_boundaries': list(self.predicted_boundaries),
            'commit_events': self.commit_events,
            'abort_events': self.abort_events,
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL format (one line per turn)."""
        lines = []
        for turn in self.turns:
            turn_dict = turn.to_dict()
            turn_dict['dialogue_id'] = self.dialogue_id
            lines.append(json.dumps(turn_dict))
        return '\n'.join(lines)


class OnlineReplayHarness:
    """
    Replay dialogues turn-by-turn for online evaluation.

    Tracks state machine transitions and produces detailed traces
    for analysis of timing, delay, and churn metrics.
    """

    def __init__(
        self,
        strategy: TopicStrategy,
        compute_drift: bool = True,
        drift_provider: str = "sentence-transformers",
        drift_model: str = "paraphrase-mpnet-base-v2",
    ):
        """
        Initialize replay harness.

        Args:
            strategy: TopicStrategy to evaluate
            compute_drift: Whether to compute semantic drift (requires embeddings)
            drift_provider: Embedding provider for drift computation
            drift_model: Embedding model for drift computation
        """
        self.strategy = strategy
        self.compute_drift = compute_drift
        self._drift_calculator = None

        if compute_drift:
            try:
                from episodic.ml.drift import ConversationalDrift
                self._drift_calculator = ConversationalDrift(
                    embedding_provider=drift_provider,
                    embedding_model=drift_model,
                )
            except ImportError:
                logger.warning("Could not import ConversationalDrift, disabling drift")
                self.compute_drift = False

    def replay_dialogue(
        self,
        test_case: EvalCase,
        strategy_alignment: BoundaryAlignment = None
    ) -> DialogueTrace:
        """
        Replay a dialogue turn-by-turn, producing a complete trace.

        Args:
            test_case: EvalCase with messages and gold boundaries
            strategy_alignment: How strategy reports boundaries (default: user_starts_topic)

        Returns:
            DialogueTrace with per-turn traces and boundary results
        """
        if strategy_alignment is None:
            strategy_alignment = ALIGNMENT_PRESETS['user_starts_topic']

        # Reset strategy state
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()

        traces: List[TurnTrace] = []
        message_history: List[Dict[str, Any]] = []
        predicted_indices: List[int] = []

        # Track SUSPECT episodes for commit/abort events
        current_suspect_entry: Optional[int] = None
        current_suspect_cause: Optional[str] = None
        commit_events: List[Tuple[int, int, str]] = []
        abort_events: List[Tuple[int, int]] = []

        # Track previous user message for drift computation
        prev_user_content: Optional[str] = None

        for i, message in enumerate(test_case.messages):
            start_time = time.time()

            # Build node_id
            node_id = message.node_id or f"msg_{i}"
            content = message.content
            content_preview = content[:50] + "..." if len(content) > 50 else content

            # Compute drift for user messages
            semantic_drift = None
            if self.compute_drift and message.role == 'user' and prev_user_content:
                semantic_drift = self._compute_drift(prev_user_content, content)

            if message.role == 'user':
                prev_user_content = content

            # Default values for non-decision turns
            neural_confidence = 0.0
            state = "STABLE"
            accumulated_evidence = 0.0
            frozen_before_ids: List[str] = []
            frozen_straddle_id: Optional[str] = None
            event: Optional[str] = None
            canonical_boundary_idx: Optional[int] = None
            commit_node_id: Optional[str] = None

            # Only run detection on user messages with sufficient history
            if message.role == 'user' and len(message_history) >= 2:
                # IMPORTANT: The strategy expects messages[-1] to be the current query
                # (as done by topic_management.py in production). Build messages_with_query
                # to match this expectation.
                messages_with_query = message_history + [message.to_dict()]

                # Pass semantic_drift to strategy if it supports it
                # CommitmentPolicyStrategy and DefaultStrategy accept semantic_drift
                # NeuralStrategy does not (it's passed through the wrapper)
                try:
                    decision = self.strategy.get_decision(
                        query=content,
                        messages=messages_with_query,
                        current_thread=None,
                        semantic_drift=semantic_drift,
                    )
                except TypeError:
                    # Fallback for strategies that don't accept semantic_drift
                    decision = self.strategy.get_decision(
                        query=content,
                        messages=messages_with_query,
                        current_thread=None,
                    )

                # Extract state info
                state_info = self._extract_state_info()
                neural_confidence = decision.confidence_score
                state = state_info.get('state', 'STABLE')
                accumulated_evidence = state_info.get('accumulated_evidence', 0.0)
                frozen_before_ids = state_info.get('frozen_before_ids', [])
                frozen_straddle_id = state_info.get('frozen_straddle_id')

                # Detect state transitions
                prev_state = traces[-1].state if traces else "STABLE"

                # Check if strategy has state machine (CommitmentPolicyStrategy)
                # A strategy has a state machine if it reports 'committed' signal
                has_state_machine = 'committed' in decision.signals

                if has_state_machine:
                    # Check for immediate commit (STABLE with topic_changed=True)
                    # This happens when min_evidence is met immediately
                    if state == "STABLE" and decision.topic_changed:
                        event = "COMMIT"
                        # Get boundary node from signals if backdated, else current
                        boundary_node = decision.signals.get('boundary_node_id')
                        if boundary_node and current_suspect_entry is not None:
                            canonical_boundary_idx = current_suspect_entry
                            predicted_indices.append(current_suspect_entry)
                            commit_events.append((
                                current_suspect_entry, i,
                                current_suspect_cause or "neural"
                            ))
                        else:
                            canonical_boundary_idx = i
                            predicted_indices.append(i)
                            commit_events.append((i, i, "neural"))
                        commit_node_id = boundary_node

                        current_suspect_entry = None
                        current_suspect_cause = None

                    # Same-turn SUSPECT→ABORT (invisible state transition)
                    # The state machine entered SUSPECT and ABORTed in same turn,
                    # so we see STABLE→STABLE but with aborted=True signal
                    elif state == "STABLE" and decision.signals.get('aborted', False):
                        event = "ABORT"
                        abort_reason = decision.signals.get('abort_reason', 'unknown')
                        # Record same-turn abort: entry and exit both at current turn
                        abort_events.append((i, i, abort_reason))

                    # STABLE -> SUSPECT transition
                    elif prev_state == "STABLE" and state == "SUSPECT":
                        current_suspect_entry = i
                        # Determine cause: check if drift triggered
                        drift_triggered = decision.signals.get('drift_triggered', False)
                        current_suspect_cause = "drift" if drift_triggered else "neural"

                    # SUSPECT -> STABLE transition (COMMIT or ABORT)
                    elif prev_state == "SUSPECT" and state == "STABLE":
                        if decision.topic_changed:
                            event = "COMMIT"
                            # Boundary is at SUSPECT entry, not current turn (backdate)
                            canonical_boundary_idx = current_suspect_entry
                            commit_node_id = decision.signals.get('boundary_node_id')
                            predicted_indices.append(current_suspect_entry)
                            if current_suspect_entry is not None:
                                commit_events.append((
                                    current_suspect_entry,
                                    i,
                                    current_suspect_cause or "neural"
                                ))
                        else:
                            event = "ABORT"
                            abort_reason = decision.signals.get('abort_reason', 'unknown')
                            if current_suspect_entry is not None:
                                abort_events.append((current_suspect_entry, i, abort_reason))

                        current_suspect_entry = None
                        current_suspect_cause = None
                else:
                    # Simple strategy without state machine (e.g., raw NeuralStrategy)
                    # Emit boundary directly when topic_changed=True
                    if decision.topic_changed:
                        event = "COMMIT"
                        canonical_boundary_idx = i
                        predicted_indices.append(i)
                        commit_events.append((i, i, "neural"))

            # Add message to history
            message_history.append(message.to_dict())

            processing_time = (time.time() - start_time) * 1000

            # Build trace
            trace = TurnTrace(
                turn_idx=i,
                node_id=node_id,
                role=message.role,
                content_preview=content_preview,
                semantic_drift=semantic_drift,
                neural_confidence=neural_confidence,
                state=state,
                suspect_entry_turn=current_suspect_entry,
                suspect_entry_node_id=traces[current_suspect_entry].node_id if current_suspect_entry and current_suspect_entry < len(traces) else None,
                frozen_before_ids=frozen_before_ids,
                frozen_straddle_id=frozen_straddle_id,
                accumulated_evidence=accumulated_evidence,
                event=event,
                canonical_boundary_idx=canonical_boundary_idx,
                commit_node_id=commit_node_id,
                processing_time_ms=processing_time,
            )
            traces.append(trace)

        # Convert predictions to canonical format
        pred_canonical = normalize_strategy_output(
            predicted_indices,
            test_case.messages,
            strategy_alignment
        )

        return DialogueTrace(
            dialogue_id=test_case.id,
            turns=traces,
            gold_boundaries=test_case.get_canonical_boundaries(),
            predicted_boundaries=pred_canonical,
            commit_events=commit_events,
            abort_events=abort_events,
        )

    def _compute_drift(self, text1: str, text2: str) -> float:
        """
        Compute semantic drift between two texts.

        CRITICAL: Uses same embedding and normalization as production.
        Embeds raw user content without role prefixes or normalization.

        Args:
            text1: Previous user message content (raw)
            text2: Current user message content (raw)

        Returns:
            Drift score (0 = identical, 1 = maximally different)
        """
        if not self._drift_calculator:
            return 0.0

        try:
            # Use raw text content - matches production behavior
            node1 = {'message': text1}
            node2 = {'message': text2}
            return self._drift_calculator.calculate_drift(node1, node2, text_field='message')
        except Exception as e:
            logger.warning(f"Drift computation failed: {e}")
            return 0.0

    def _extract_state_info(self) -> Dict[str, Any]:
        """
        Extract state machine info from strategy.

        Works with CommitmentPolicyStrategy and DefaultStrategy.
        """
        info: Dict[str, Any] = {
            'state': 'STABLE',
            'accumulated_evidence': 0.0,
            'frozen_before_ids': [],
            'frozen_straddle_id': None,
        }

        # Check for CommitmentPolicyStrategy
        strategy = self.strategy

        # If DefaultStrategy, get inner strategy
        if hasattr(strategy, '_strategy'):
            strategy = strategy._strategy

        # If CommitmentPolicyStrategy, extract state
        if hasattr(strategy, '_state'):
            state_obj = strategy._state
            info['state'] = getattr(state_obj, 'state', 'STABLE')
            info['accumulated_evidence'] = getattr(state_obj, 'accumulated_evidence', 0.0)

            # Extract frozen context node IDs
            frozen_before = getattr(state_obj, 'frozen_before', None)
            if frozen_before:
                info['frozen_before_ids'] = [
                    msg.get('node_id', f'msg_{i}')
                    for i, msg in enumerate(frozen_before)
                    if isinstance(msg, dict)
                ]

            frozen_straddle = getattr(state_obj, 'frozen_straddle_msg', None)
            if frozen_straddle and isinstance(frozen_straddle, dict):
                info['frozen_straddle_id'] = frozen_straddle.get('node_id')

        return info


def write_traces_jsonl(traces: List[DialogueTrace], output_path: Path) -> None:
    """
    Write dialogue traces to JSONL file (one line per turn).

    Args:
        traces: List of DialogueTrace objects
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        for trace in traces:
            f.write(trace.to_jsonl())
            f.write('\n')


def load_traces_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load turn traces from JSONL file.

    Args:
        input_path: Path to input file

    Returns:
        List of turn trace dicts
    """
    traces = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces
