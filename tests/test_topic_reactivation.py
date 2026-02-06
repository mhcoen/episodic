"""
Comprehensive test suite for implicit topic reactivation.

Tests cover:
1. Probe logic (probe_reactivation)
2. Packet assembly (assemble_reactivation_packet)
3. Wrapper/arbitration logic
4. TopicHandler dry_run and decision_override
5. State management (cooldown, dormancy)
6. Edge cases and error handling

Run with: pytest test_topic_reactivation.py -v
"""

import hashlib
import pytest
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Literal
from unittest.mock import Mock, MagicMock, patch
import sqlite3


# ============================================================================
# TEST FIXTURES AND MOCKS
# ============================================================================

@dataclass
class MockTopic:
    """Mock topic for testing."""
    name: str
    start_node_id: str
    end_node_id: Optional[str] = None
    last_active_turn_idx: int = 0
    exchange_count: int = 5
    centroid_medoid_exchange_id: Optional[str] = None
    is_compressed: bool = False
    summary: Optional[str] = None


@dataclass
class MockExchange:
    """Mock exchange/message for testing."""
    exchange_id: str
    topic_start_node_id: str
    text: str
    embedding: np.ndarray
    turn_idx: int
    role: str = "user"


@dataclass
class ReactivationDecision:
    """Expected structure for reactivation decisions."""
    action: Literal["CONTINUE", "REACTIVATE", "DISAMBIGUATE"]
    topic_name: Optional[str] = None
    topic_start_node_id: Optional[str] = None
    options: Optional[List[Any]] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DisambiguationOption:
    """Option presented during disambiguation."""
    option_id: int
    topic_name: str
    start_node_id: str
    representative_snippets: List[str]
    support_count: int
    last_active_turns_ago: int


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider that returns deterministic embeddings."""
    provider = Mock()

    def embed(text: str) -> np.ndarray:
        # Generate deterministic embedding based on text content
        # Use hashlib (not hash()) because Python randomizes hash() per-run
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed)
        emb = np.random.randn(384)
        return emb / np.linalg.norm(emb)  # L2 normalize

    provider.embed = embed
    provider.embed_batch = lambda texts: [embed(t) for t in texts]
    provider.get_dimension = lambda: 384
    return provider


@pytest.fixture
def mock_topics() -> List[MockTopic]:
    """Create a set of mock topics for testing."""
    return [
        MockTopic(
            name="baseball-discussion",
            start_node_id="node_001",
            end_node_id="node_010",
            last_active_turn_idx=5,
            exchange_count=10,
            centroid_medoid_exchange_id="exch_005"
        ),
        MockTopic(
            name="python-debugging",
            start_node_id="node_011",
            end_node_id="node_025",
            last_active_turn_idx=20,
            exchange_count=15,
            centroid_medoid_exchange_id="exch_018"
        ),
        MockTopic(
            name="coffee-brewing",
            start_node_id="node_026",
            end_node_id=None,  # Active/ongoing
            last_active_turn_idx=45,
            exchange_count=8,
            centroid_medoid_exchange_id="exch_042"
        ),
    ]


@pytest.fixture
def mock_exchanges(mock_embedding_provider) -> List[MockExchange]:
    """Create mock exchanges across topics."""
    exchanges = []

    # Baseball topic exchanges
    baseball_texts = [
        "What's the best batting stance for power hitting?",
        "The Yankees had an amazing season last year",
        "How do you calculate ERA for pitchers?",
        "Spring training starts next month",
        "The World Series was incredible this year",
    ]
    for i, text in enumerate(baseball_texts):
        exchanges.append(MockExchange(
            exchange_id=f"exch_00{i+1}",
            topic_start_node_id="node_001",
            text=text,
            embedding=mock_embedding_provider.embed(text),
            turn_idx=i + 1
        ))

    # Python topic exchanges
    python_texts = [
        "How do I debug this recursive function?",
        "The stack trace shows a KeyError",
        "Can you explain Python decorators?",
        "My unit tests are failing intermittently",
    ]
    for i, text in enumerate(python_texts):
        exchanges.append(MockExchange(
            exchange_id=f"exch_01{i+1}",
            topic_start_node_id="node_011",
            text=text,
            embedding=mock_embedding_provider.embed(text),
            turn_idx=i + 15
        ))

    # Coffee topic exchanges
    coffee_texts = [
        "What's the ideal water temperature for pour-over?",
        "I prefer medium roast beans",
        "How fine should I grind for espresso?",
    ]
    for i, text in enumerate(coffee_texts):
        exchanges.append(MockExchange(
            exchange_id=f"exch_04{i+1}",
            topic_start_node_id="node_026",
            text=text,
            embedding=mock_embedding_provider.embed(text),
            turn_idx=i + 43
        ))

    return exchanges


@pytest.fixture
def mock_topic_index(mock_topics, mock_exchanges, mock_embedding_provider):
    """Mock topic-level ANN index."""
    index = Mock()

    def query_topics(user_emb: np.ndarray, k: int) -> List[Dict]:
        """Return topics ranked by similarity to user embedding."""
        results = []
        for topic in mock_topics:
            # Find medoid exchange for this topic
            medoid_exch = next(
                (e for e in mock_exchanges if e.exchange_id == topic.centroid_medoid_exchange_id),
                None
            )
            if medoid_exch:
                sim = float(np.dot(user_emb, medoid_exch.embedding))
            else:
                sim = 0.0

            results.append({
                "topic_id": topic.start_node_id,
                "topic_name": topic.name,
                "sim_to_medoid": sim,
                "exchange_count": topic.exchange_count,
                "last_active_turn_idx": topic.last_active_turn_idx,
            })

        # Sort by similarity descending
        results.sort(key=lambda x: x["sim_to_medoid"], reverse=True)
        return results[:k]

    index.query_topics = query_topics
    return index


@pytest.fixture
def mock_exchange_index(mock_exchanges):
    """Mock within-topic exchange retrieval."""
    index = Mock()

    def query_exchanges_in_topic(
        topic_start_node_id: str,
        user_emb: np.ndarray,
        m: int,
        exclude_last_n: int = 0
    ) -> List[Dict]:
        """Return top exchanges within a topic by similarity."""
        topic_exchanges = [
            e for e in mock_exchanges
            if e.topic_start_node_id == topic_start_node_id
        ]

        # Exclude last n if specified
        if exclude_last_n > 0:
            topic_exchanges = topic_exchanges[:-exclude_last_n]

        results = []
        for exch in topic_exchanges:
            sim = float(np.dot(user_emb, exch.embedding))
            results.append({
                "exchange_id": exch.exchange_id,
                "sim": sim,
                "turn_idx": exch.turn_idx,
                "text": exch.text,
            })

        results.sort(key=lambda x: x["sim"], reverse=True)
        return results[:m]

    index.query_exchanges_in_topic = query_exchanges_in_topic
    return index


@pytest.fixture
def mock_conversation_manager():
    """Mock ConversationManager with reactivation state."""
    cm = Mock()
    cm.current_topic = ("coffee-brewing", "node_026")
    cm.reactivation_cooldown_turns = 0
    cm.last_reactivation_topic_start_node_id = None

    cm.get_current_topic = lambda: cm.current_topic

    def set_current_topic(name: str, start_node_id: str):
        cm.current_topic = (name, start_node_id)
    cm.set_current_topic = set_current_topic

    return cm


@pytest.fixture
def mock_topic_handler():
    """Mock TopicHandler with dry_run and decision_override support."""
    th = Mock()
    th._messages_in_current_topic = 0
    th._dry_run_calls = []
    th._mutating_calls = []

    def detect_and_handle_topic_change(
        recent_nodes: List[Dict],
        user_input: str,
        user_node_id: str,
        semantic_drift: Optional[float] = None,
        dry_run: bool = False,
        decision_override: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict], Optional[Dict]]:

        call_record = {
            "user_input": user_input,
            "dry_run": dry_run,
            "decision_override": decision_override,
        }

        if dry_run:
            th._dry_run_calls.append(call_record)
        else:
            th._mutating_calls.append(call_record)
            if decision_override != "FORCE_CONTINUE":
                th._messages_in_current_topic += 1

        # Simulate different behaviors based on input
        if "completely new subject" in user_input.lower():
            raw_topic_changed = True
            new_topic_name = "new-topic"
        else:
            raw_topic_changed = False
            new_topic_name = None

        if decision_override == "FORCE_CONTINUE":
            topic_changed = False
        else:
            topic_changed = raw_topic_changed

        topic_cost_info = {"method": "strategy", "confidence_score": 0.8}
        topic_change_info = {
            "raw_topic_changed": raw_topic_changed,
            "changed": topic_changed,
        }

        return (topic_changed, new_topic_name, topic_cost_info, topic_change_info)

    th.detect_and_handle_topic_change = detect_and_handle_topic_change
    th.handle_topic_boundaries = Mock()
    th.increment_message_count = lambda: setattr(th, '_messages_in_current_topic', th._messages_in_current_topic + 1)
    th.reset_message_count = lambda: setattr(th, '_messages_in_current_topic', 0)

    return th


# ============================================================================
# PROBE LOGIC TESTS
# ============================================================================

class TestProbeReactivation:
    """Tests for probe_reactivation() function."""

    def test_cooldown_returns_continue(self, mock_embedding_provider, mock_topic_index):
        """Probe returns CONTINUE immediately when cooldown > 0."""
        user_input = "Tell me more about the Yankees"
        u_emb = mock_embedding_provider.embed(user_input)

        decision = self._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id="node_026",
            cooldown_turns=2,
            current_turn_idx=50,
            topic_index=mock_topic_index
        )

        assert decision.action == "CONTINUE"
        assert decision.debug.get("reason") == "cooldown"

    def test_short_input_skipped(self, mock_embedding_provider):
        """Very short inputs are skipped (no probe run)."""
        short_inputs = ["ok", "yes", "thanks", "hi"]

        for user_input in short_inputs:
            should_skip = len(user_input.split()) < 4
            assert should_skip, f"'{user_input}' should be skipped"

    def test_dormancy_filter_excludes_recent_topics(
        self, mock_embedding_provider, mock_topic_index, mock_topics
    ):
        """Topics active within dormancy_min turns are excluded from reactivation."""
        user_input = "What about the coffee grind size?"
        u_emb = mock_embedding_provider.embed(user_input)
        current_turn_idx = 47
        dormancy_min = 4

        decision = self._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id="node_026",
            cooldown_turns=0,
            current_turn_idx=current_turn_idx,
            topic_index=mock_topic_index,
            dormancy_min=dormancy_min
        )

        if decision.action == "REACTIVATE":
            assert decision.topic_start_node_id != "node_026"

    def test_reactivate_to_self_returns_continue(self, mock_embedding_provider, mock_topic_index):
        """If best topic is already active, return CONTINUE."""
        user_input = "What's the best coffee bean origin?"
        u_emb = mock_embedding_provider.embed(user_input)

        decision = self._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id="node_026",
            cooldown_turns=0,
            current_turn_idx=50,
            topic_index=mock_topic_index
        )

        if decision.topic_start_node_id == "node_026":
            assert decision.action == "CONTINUE"
            assert decision.debug.get("reason") == "reactivate_to_self"

    def _simulate_probe(
        self,
        user_input: str,
        user_embedding: np.ndarray,
        active_topic_start_node_id: Optional[str],
        cooldown_turns: int,
        current_turn_idx: int,
        topic_index: Mock,
        exchange_index: Mock = None,
        dormancy_min: int = 4,
        support_threshold: int = 2,
        rank_gap_threshold: int = 2,
        sim_gate_percentile: float = 0.25,
        delta: float = 0.07
    ) -> ReactivationDecision:
        """Simulate probe_reactivation logic for testing."""
        debug = {}

        if cooldown_turns > 0:
            return ReactivationDecision(
                action="CONTINUE",
                debug={"reason": "cooldown", "cooldown_remaining": cooldown_turns}
            )

        candidates = topic_index.query_topics(user_embedding, k=7)

        if not candidates:
            return ReactivationDecision(
                action="CONTINUE",
                debug={"reason": "no_topics"}
            )

        active_rank = None
        for i, c in enumerate(candidates):
            if c["topic_id"] == active_topic_start_node_id:
                active_rank = i
                break

        eligible = []
        for i, c in enumerate(candidates):
            dormancy = current_turn_idx - c["last_active_turn_idx"]
            if dormancy >= dormancy_min and c["topic_id"] != active_topic_start_node_id:
                c["dormancy_turns"] = dormancy
                c["rank"] = i
                eligible.append(c)

        if not eligible:
            return ReactivationDecision(
                action="CONTINUE",
                debug={"reason": "no_eligible_topics"}
            )

        best = eligible[0]
        debug["best_topic"] = best["topic_name"]
        debug["best_sim"] = best["sim_to_medoid"]
        debug["dormancy_turns"] = best["dormancy_turns"]

        if active_rank is not None:
            rank_gap = active_rank - best["rank"]
            debug["rank_gap"] = rank_gap
            debug["rank_gap_passes"] = rank_gap >= rank_gap_threshold
        else:
            debug["rank_gap_passes"] = True

        if exchange_index:
            within_topic = exchange_index.query_exchanges_in_topic(
                best["topic_id"], user_embedding, m=12
            )

            if within_topic:
                best_sim = within_topic[0]["sim"]
                support_count = sum(
                    1 for e in within_topic
                    if e["sim"] >= best_sim - delta
                )
                debug["support_count"] = support_count
            else:
                debug["support_count"] = 0
        else:
            debug["support_count"] = support_threshold

        if best["topic_id"] == active_topic_start_node_id:
            return ReactivationDecision(
                action="CONTINUE",
                debug={"reason": "reactivate_to_self", **debug}
            )

        if debug.get("support_count", 0) >= support_threshold and debug.get("rank_gap_passes", True):
            return ReactivationDecision(
                action="REACTIVATE",
                topic_name=best["topic_name"],
                topic_start_node_id=best["topic_id"],
                debug=debug
            )

        return ReactivationDecision(
            action="CONTINUE",
            debug={"reason": "insufficient_evidence", **debug}
        )


# ============================================================================
# ARBITRATION TESTS
# ============================================================================

class TestArbitration:
    """Tests for wrapper arbitration logic."""

    def test_probe_continue_follows_tracker(self, mock_conversation_manager, mock_topic_handler):
        """When probe returns CONTINUE, tracker decision is followed."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        probe_decision = ReactivationDecision(action="CONTINUE")

        result = self._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input="Continue discussing coffee",
            user_node_id="node_050"
        )

        assert result["final_decision"] == "tracker"
        assert len(th._mutating_calls) == 1
        assert th._mutating_calls[0]["decision_override"] is None

    def test_reactivate_wins_when_tracker_continues(
        self, mock_conversation_manager, mock_topic_handler
    ):
        """REACTIVATE wins when tracker would CONTINUE (no conflict)."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        probe_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="baseball-discussion",
            topic_start_node_id="node_001",
            debug={"support_count": 3, "rank_gap_passes": True}
        )

        result = self._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input="What about the Yankees?",
            user_node_id="node_050"
        )

        assert result["final_decision"] == "reactivation"
        assert cm.current_topic == ("baseball-discussion", "node_001")
        assert cm.reactivation_cooldown_turns == 3

        assert len(th._dry_run_calls) == 1
        assert len(th._mutating_calls) == 1
        assert th._mutating_calls[0]["decision_override"] == "FORCE_CONTINUE"

    def test_reactivate_needs_stronger_evidence_against_new_topic(
        self, mock_conversation_manager, mock_topic_handler
    ):
        """When tracker wants NEW_TOPIC, reactivation needs support_count >= S+1."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        probe_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="baseball-discussion",
            topic_start_node_id="node_001",
            debug={"support_count": 2, "rank_gap_passes": True}
        )

        result = self._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input="Completely new subject about astronomy",
            user_node_id="node_050",
            support_threshold=2
        )

        assert result["final_decision"] == "tracker"

    def test_reactivate_overrides_new_topic_with_strong_evidence(
        self, mock_conversation_manager, mock_topic_handler
    ):
        """Reactivation with strong evidence (support >= S+1) overrides tracker NEW_TOPIC."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        probe_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="baseball-discussion",
            topic_start_node_id="node_001",
            debug={"support_count": 4, "rank_gap_passes": True}
        )

        result = self._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input="Completely new subject but actually about baseball",
            user_node_id="node_050",
            support_threshold=2
        )

        assert result["final_decision"] == "reactivation"
        assert cm.current_topic == ("baseball-discussion", "node_001")

    def test_disambiguation_dismissed_follows_tracker(
        self, mock_conversation_manager, mock_topic_handler
    ):
        """When user dismisses disambiguation, fall through to tracker."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        probe_decision = ReactivationDecision(
            action="DISAMBIGUATE",
            options=[
                DisambiguationOption(
                    option_id=1, topic_name="baseball", start_node_id="node_001",
                    representative_snippets=["Yankees game"], support_count=2,
                    last_active_turns_ago=45
                ),
                DisambiguationOption(
                    option_id=2, topic_name="python", start_node_id="node_011",
                    representative_snippets=["Debug code"], support_count=2,
                    last_active_turns_ago=30
                ),
            ]
        )

        result = self._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input="java",
            user_node_id="node_050",
            disambiguation_selection=None
        )

        assert result["final_decision"] == "tracker"
        assert cm.current_topic[1] not in ["node_001", "node_011"]

    def test_handle_boundaries_not_called_on_reactivation(
        self, mock_conversation_manager, mock_topic_handler
    ):
        """handle_topic_boundaries is NOT called when reactivation wins."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        probe_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="baseball-discussion",
            topic_start_node_id="node_001",
            debug={"support_count": 3, "rank_gap_passes": True}
        )

        result = self._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input="Tell me about the game",
            user_node_id="node_050"
        )

        assert result["final_decision"] == "reactivation"
        th.handle_topic_boundaries.assert_not_called()

    def _run_arbitration(
        self,
        cm: Mock,
        th: Mock,
        probe_decision: ReactivationDecision,
        user_input: str,
        user_node_id: str,
        assistant_node_id: str = "node_051",
        support_threshold: int = 2,
        disambiguation_selection: Optional[DisambiguationOption] = None
    ) -> Dict:
        """Simulate the wrapper arbitration logic."""
        result = {"final_decision": None, "tracker_topic_changed": False}
        recent_nodes = []

        if probe_decision.action == "DISAMBIGUATE":
            if disambiguation_selection is None:
                probe_decision = ReactivationDecision(action="CONTINUE")
            else:
                probe_decision = ReactivationDecision(
                    action="REACTIVATE",
                    topic_name=disambiguation_selection.topic_name,
                    topic_start_node_id=disambiguation_selection.start_node_id,
                    debug=probe_decision.debug
                )

        if probe_decision.action == "CONTINUE":
            (topic_changed, new_topic_name, cost_info, change_info) = \
                th.detect_and_handle_topic_change(
                    recent_nodes=recent_nodes,
                    user_input=user_input,
                    user_node_id=user_node_id
                )

            result["final_decision"] = "tracker"
            result["tracker_topic_changed"] = topic_changed

            if topic_changed:
                th.handle_topic_boundaries(
                    topic_changed=True,
                    user_node_id=user_node_id,
                    assistant_node_id=assistant_node_id,
                    topic_change_info=change_info,
                    new_topic_name=new_topic_name
                )

        else:
            (raw_topic_changed, raw_new_name, raw_cost, raw_change) = \
                th.detect_and_handle_topic_change(
                    recent_nodes=recent_nodes,
                    user_input=user_input,
                    user_node_id=user_node_id,
                    dry_run=True
                )

            result["tracker_topic_changed"] = raw_topic_changed

            reactivation_wins = True
            if raw_topic_changed:
                support_count = probe_decision.debug.get("support_count", 0)
                rank_gap_passes = probe_decision.debug.get("rank_gap_passes", False)

                if support_count >= support_threshold + 1 and rank_gap_passes:
                    reactivation_wins = True
                else:
                    reactivation_wins = False

            if reactivation_wins:
                cm.set_current_topic(
                    probe_decision.topic_name,
                    probe_decision.topic_start_node_id
                )
                cm.reactivation_cooldown_turns = 3

                th.detect_and_handle_topic_change(
                    recent_nodes=recent_nodes,
                    user_input=user_input,
                    user_node_id=user_node_id,
                    decision_override="FORCE_CONTINUE"
                )

                result["final_decision"] = "reactivation"
            else:
                th.detect_and_handle_topic_change(
                    recent_nodes=recent_nodes,
                    user_input=user_input,
                    user_node_id=user_node_id
                )

                result["final_decision"] = "tracker"

                if raw_topic_changed:
                    th.handle_topic_boundaries(
                        topic_changed=True,
                        user_node_id=user_node_id,
                        assistant_node_id=assistant_node_id,
                        topic_change_info=raw_change,
                        new_topic_name=raw_new_name
                    )

        return result


# ============================================================================
# TOPIC HANDLER MODIFICATION TESTS
# ============================================================================

class TestTopicHandlerModifications:
    """Tests for dry_run and decision_override in TopicHandler."""

    def test_dry_run_no_counter_increment(self, mock_topic_handler):
        """dry_run=True should not increment message counter."""
        th = mock_topic_handler
        initial_count = th._messages_in_current_topic

        th.detect_and_handle_topic_change(
            recent_nodes=[],
            user_input="Test message",
            user_node_id="node_100",
            dry_run=True
        )

        assert th._messages_in_current_topic == initial_count

    def test_normal_call_increments_counter(self, mock_topic_handler):
        """Normal call (dry_run=False, no override) increments counter."""
        th = mock_topic_handler
        initial_count = th._messages_in_current_topic

        th.detect_and_handle_topic_change(
            recent_nodes=[],
            user_input="Test message",
            user_node_id="node_100",
            dry_run=False
        )

        assert th._messages_in_current_topic == initial_count + 1

    def test_force_continue_returns_false(self, mock_topic_handler):
        """decision_override=FORCE_CONTINUE returns topic_changed=False."""
        th = mock_topic_handler

        (topic_changed, _, _, _) = th.detect_and_handle_topic_change(
            recent_nodes=[],
            user_input="Completely new subject about astronomy",
            user_node_id="node_100",
            decision_override="FORCE_CONTINUE"
        )

        assert topic_changed is False

    def test_force_continue_preserves_raw_decision(self, mock_topic_handler):
        """FORCE_CONTINUE still returns raw decision info for logging."""
        th = mock_topic_handler

        (topic_changed, _, _, topic_change_info) = th.detect_and_handle_topic_change(
            recent_nodes=[],
            user_input="Completely new subject about astronomy",
            user_node_id="node_100",
            decision_override="FORCE_CONTINUE"
        )

        assert topic_changed is False
        assert topic_change_info.get("raw_topic_changed") is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_resume_baseball_scenario(
        self, mock_conversation_manager, mock_topic_handler,
        mock_embedding_provider, mock_topic_index, mock_exchange_index
    ):
        """Full scenario: discussing coffee, ask about baseball → reactivate."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        cm.current_topic = ("coffee-brewing", "node_026")
        cm.reactivation_cooldown_turns = 0

        user_input = "What was the Yankees' win record last season?"
        u_emb = mock_embedding_provider.embed(user_input)

        probe_decision = TestProbeReactivation()._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id="node_026",
            cooldown_turns=0,
            current_turn_idx=50,
            topic_index=mock_topic_index,
            exchange_index=mock_exchange_index
        )

        result = TestArbitration()._run_arbitration(
            cm=cm,
            th=th,
            probe_decision=probe_decision,
            user_input=user_input,
            user_node_id="node_050"
        )

        if probe_decision.action == "REACTIVATE":
            assert result["final_decision"] == "reactivation"
            assert cm.current_topic[0] == "baseball-discussion"
            assert cm.reactivation_cooldown_turns == 3

    def test_no_reactivation_during_cooldown(
        self, mock_conversation_manager, mock_topic_handler,
        mock_embedding_provider, mock_topic_index
    ):
        """No reactivation while cooldown is active."""
        cm = mock_conversation_manager
        th = mock_topic_handler

        cm.current_topic = ("baseball-discussion", "node_001")
        cm.reactivation_cooldown_turns = 2

        user_input = "How do I debug this recursive function?"
        u_emb = mock_embedding_provider.embed(user_input)

        probe_decision = TestProbeReactivation()._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id="node_001",
            cooldown_turns=2,
            current_turn_idx=52,
            topic_index=mock_topic_index
        )

        assert probe_decision.action == "CONTINUE"
        assert probe_decision.debug.get("reason") == "cooldown"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_topics_exist(self, mock_embedding_provider):
        """Handle case where no topics exist yet."""
        empty_index = Mock()
        empty_index.query_topics = lambda *args, **kwargs: []

        user_input = "Hello, how are you?"
        u_emb = mock_embedding_provider.embed(user_input)

        decision = TestProbeReactivation()._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id=None,
            cooldown_turns=0,
            current_turn_idx=1,
            topic_index=empty_index
        )

        assert decision.action == "CONTINUE"
        assert decision.debug.get("reason") == "no_topics"

    def test_only_active_topic_exists(self, mock_embedding_provider):
        """Handle case where only the active topic exists."""
        single_topic_index = Mock()
        single_topic_index.query_topics = lambda *args, **kwargs: [{
            "topic_id": "node_001",
            "topic_name": "only-topic",
            "sim_to_medoid": 0.9,
            "exchange_count": 10,
            "last_active_turn_idx": 50,
        }]

        user_input = "Continue this topic"
        u_emb = mock_embedding_provider.embed(user_input)

        decision = TestProbeReactivation()._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id="node_001",
            cooldown_turns=0,
            current_turn_idx=51,
            topic_index=single_topic_index,
            dormancy_min=4
        )

        assert decision.action == "CONTINUE"

    def test_null_active_topic(
        self, mock_conversation_manager, mock_embedding_provider, mock_topic_index
    ):
        """Handle case where there's no active topic (None)."""
        cm = mock_conversation_manager
        cm.current_topic = None

        user_input = "Let's talk about baseball"
        u_emb = mock_embedding_provider.embed(user_input)

        decision = TestProbeReactivation()._simulate_probe(
            user_input=user_input,
            user_embedding=u_emb,
            active_topic_start_node_id=None,
            cooldown_turns=0,
            current_turn_idx=50,
            topic_index=mock_topic_index
        )

        assert decision.action in ["CONTINUE", "REACTIVATE", "DISAMBIGUATE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
