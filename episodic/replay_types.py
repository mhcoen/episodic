"""Replay snapshot/diff/result dataclasses.

Split out of replay.py; re-exported there so external imports
(from episodic.replay import ReplaySnapshot, ...) are unchanged.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

from episodic.token_guard_events import EventLogger, EventVerifier, canonical_json
from episodic.token_guard import TokenBudget, validate_assembly, token_counter_registry

# Snapshot schema version for forward compatibility
SNAPSHOT_SCHEMA_VERSION = "1.0"


@dataclass
class RetrievalState:
    """
    Frozen retrieval state for replay.

    Captures all inputs and outputs of the retrieval process
    so it can be replayed without network calls or database access.
    """
    # Embedding model used
    embedding_model_identifier: Optional[str] = None

    # Query embedding vector (actual floats for replay)
    query_embedding_vector: Optional[List[float]] = None

    # Retrieval results: list of (exchange_id, score) in final order
    retrieval_results: List[Tuple[str, float]] = field(default_factory=list)

    # Topic membership mapping used for promotion
    topic_membership_mapping: Dict[str, str] = field(default_factory=dict)

    # Promoted topic IDs (result of topic promotion logic)
    promoted_topic_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "embedding_model_identifier": self.embedding_model_identifier,
            "query_embedding_vector": self.query_embedding_vector,
            "retrieval_results": self.retrieval_results,
            "topic_membership_mapping": self.topic_membership_mapping,
            "promoted_topic_ids": self.promoted_topic_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalState":
        """Create from dictionary."""
        return cls(
            embedding_model_identifier=d.get("embedding_model_identifier"),
            query_embedding_vector=d.get("query_embedding_vector"),
            retrieval_results=[tuple(r) for r in d.get("retrieval_results", [])],
            topic_membership_mapping=d.get("topic_membership_mapping", {}),
            promoted_topic_ids=d.get("promoted_topic_ids", []),
        )


@dataclass
class TokenGuardConfig:
    """Token guard configuration for replay."""
    full_cap: int = 8000
    summary_min: int = 100
    overhead_reserve: int = 500
    safety_factor: float = 1.2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenGuardConfig":
        """Create from dictionary."""
        return cls(**d)

    def to_budget(self) -> TokenBudget:
        """Convert to TokenBudget for validate_assembly."""
        return TokenBudget(
            full_cap=self.full_cap,
            summary_min=self.summary_min,
            overhead_reserve=self.overhead_reserve,
        )


@dataclass
class ContextInputs:
    """
    All inputs used for context assembly.

    Captures everything needed to reproduce the assembly.
    """
    # User's current message
    user_turn_text: str

    # Conversation context (summaries, anchors, recency lists)
    summary_text: Optional[str] = None
    anchor_exchanges: List[Dict[str, str]] = field(default_factory=list)
    recency_exchanges: List[Dict[str, str]] = field(default_factory=list)

    # System prompt components
    system_prompt: Optional[str] = None

    # RAG context (if any)
    rag_context: Optional[str] = None

    # Web context (if any)
    web_context: Optional[Dict[str, Any]] = None

    # Topic context (if any)
    topic_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_turn_text": self.user_turn_text,
            "summary_text": self.summary_text,
            "anchor_exchanges": self.anchor_exchanges,
            "recency_exchanges": self.recency_exchanges,
            "system_prompt": self.system_prompt,
            "rag_context": self.rag_context,
            "web_context": self.web_context,
            "topic_context": self.topic_context,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextInputs":
        """Create from dictionary."""
        return cls(
            user_turn_text=d.get("user_turn_text", ""),
            summary_text=d.get("summary_text"),
            anchor_exchanges=d.get("anchor_exchanges", []),
            recency_exchanges=d.get("recency_exchanges", []),
            system_prompt=d.get("system_prompt"),
            rag_context=d.get("rag_context"),
            web_context=d.get("web_context"),
            topic_context=d.get("topic_context"),
        )


@dataclass
class ReplaySnapshot:
    """
    Complete snapshot for deterministic replay.

    Contains all state needed to reproduce:
    - Selected memories / topic IDs
    - Assembled message list (exact text)
    - Token counts (per backend)
    - Emitted TokenGuardEvent stream
    """
    # Schema version for forward compatibility
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    # Identification
    run_id: str = ""
    turn_id: str = ""

    # Provider/model info
    provider_id: Optional[str] = None
    model_id: Optional[str] = None
    tokenizer_backend_name: str = "heuristic_chars_div_4"
    exact_flag: bool = False
    safety_factor_config: float = 1.2

    # Timestamp (ISO 8601, for reference only - not used in equality)
    created_at: str = ""

    # Inputs
    inputs: ContextInputs = field(default_factory=ContextInputs)

    # Retrieval state
    retrieval: RetrievalState = field(default_factory=RetrievalState)

    # Outputs: assembled messages exactly as sent to model
    assembled_messages: List[Dict[str, Any]] = field(default_factory=list)

    # Token guard configuration
    token_guard_config: TokenGuardConfig = field(default_factory=TokenGuardConfig)

    # Token guard events (JSONL lines or list of event dicts)
    token_guard_events: List[Dict[str, Any]] = field(default_factory=list)

    # Final hash for quick integrity check
    final_event_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "tokenizer_backend_name": self.tokenizer_backend_name,
            "exact_flag": self.exact_flag,
            "safety_factor_config": self.safety_factor_config,
            "created_at": self.created_at,
            "inputs": self.inputs.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "assembled_messages": self.assembled_messages,
            "token_guard_config": self.token_guard_config.to_dict(),
            "token_guard_events": self.token_guard_events,
            "final_event_hash": self.final_event_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReplaySnapshot":
        """Create from dictionary."""
        return cls(
            schema_version=d.get("schema_version", SNAPSHOT_SCHEMA_VERSION),
            run_id=d.get("run_id", ""),
            turn_id=d.get("turn_id", ""),
            provider_id=d.get("provider_id"),
            model_id=d.get("model_id"),
            tokenizer_backend_name=d.get("tokenizer_backend_name", "heuristic_chars_div_4"),
            exact_flag=d.get("exact_flag", False),
            safety_factor_config=d.get("safety_factor_config", 1.2),
            created_at=d.get("created_at", ""),
            inputs=ContextInputs.from_dict(d.get("inputs", {})),
            retrieval=RetrievalState.from_dict(d.get("retrieval", {})),
            assembled_messages=d.get("assembled_messages", []),
            token_guard_config=TokenGuardConfig.from_dict(d.get("token_guard_config", {})),
            token_guard_events=d.get("token_guard_events", []),
            final_event_hash=d.get("final_event_hash"),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save snapshot to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReplaySnapshot":
        """Load snapshot from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class ReplayDiff:
    """
    Describes a divergence between expected and actual values.
    """
    field_path: str
    expected_snippet: str
    actual_snippet: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_path": self.field_path,
            "expected_snippet": self.expected_snippet,
            "actual_snippet": self.actual_snippet,
            "message": self.message,
        }


@dataclass
class ReplayResult:
    """
    Result of a replay operation.
    """
    success: bool
    snapshot: ReplaySnapshot

    # Replayed outputs
    replayed_messages: List[Dict[str, Any]] = field(default_factory=list)
    replayed_events: List[Dict[str, Any]] = field(default_factory=list)
    replayed_token_count: int = 0

    # Verification results
    messages_match: bool = True
    tokens_match: bool = True
    events_match: bool = True
    hash_chain_valid: bool = True

    # First divergence (if any)
    first_diff: Optional[ReplayDiff] = None

    # All diffs found
    all_diffs: List[ReplayDiff] = field(default_factory=list)

    # Token counter availability
    counter_verified: bool = True  # False if backend not available
    counter_backend_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "messages_match": self.messages_match,
            "tokens_match": self.tokens_match,
            "events_match": self.events_match,
            "hash_chain_valid": self.hash_chain_valid,
            "counter_verified": self.counter_verified,
            "counter_backend_used": self.counter_backend_used,
            "replayed_token_count": self.replayed_token_count,
            "first_diff": self.first_diff.to_dict() if self.first_diff else None,
            "all_diffs": [d.to_dict() for d in self.all_diffs],
        }


