"""
Token Guard Event Logging: Structured, schema-versioned events with hash chain integrity.

Category D Implementation:
- Structured events with all required fields
- Exactly-once emission per assembly_id
- Monotonic event_seq per run_id
- Rolling hash chain for tamper evidence
- Canonicalization for deterministic hashing
"""

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, TextIO, Set

# Schema version for forward compatibility
SCHEMA_VERSION = "1.0"


class EventType(Enum):
    """Event types for token guard decisions."""
    TOKEN_OK = "token_ok"
    TOKEN_OVERFLOW_RECOVERED = "token_overflow_recovered"
    TOKEN_OVERFLOW_ABORT = "token_overflow_abort"


@dataclass
class TokenGuardEvent:
    """
    Structured event for token guard decisions.

    All required fields must be present. Unknown fields allowed for forward compatibility.
    """
    # Schema and identification
    schema_version: str
    event_type: str  # EventType value
    run_id: str
    turn_id: str
    assembly_id: str

    # Timing and ordering
    ts: str  # ISO 8601 timestamp
    event_seq: int  # Monotonic per run_id

    # Counter info
    counter_backend: str
    counter_exact: bool
    applied_safety_factor: Optional[str]  # String to avoid float precision issues

    # Token counts
    raw_tokens: int
    effective_tokens: int
    cap: int
    budget_breakdown: Dict[str, int]

    # Hash chain
    prev_hash: Optional[str]
    hash: str

    # Optional extra fields for forward compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def required_fields(cls) -> Set[str]:
        """Return set of required field names."""
        return {
            "schema_version", "event_type", "run_id", "turn_id", "assembly_id",
            "ts", "event_seq", "counter_backend", "counter_exact",
            "applied_safety_factor", "raw_tokens", "effective_tokens", "cap",
            "budget_breakdown", "prev_hash", "hash"
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Flatten extra fields into main dict
        extra = d.pop("extra", {})
        d.update(extra)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenGuardEvent":
        """Create from dictionary, preserving unknown fields in extra."""
        known_fields = {
            "schema_version", "event_type", "run_id", "turn_id", "assembly_id",
            "ts", "event_seq", "counter_backend", "counter_exact",
            "applied_safety_factor", "raw_tokens", "effective_tokens", "cap",
            "budget_breakdown", "prev_hash", "hash"
        }
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra=extra)


def canonical_json(obj: Dict[str, Any]) -> str:
    """
    Produce canonical JSON for deterministic hashing.

    Rules:
    - Sorted keys
    - No trailing whitespace
    - Compact separators
    - UTF-8 encoding
    - Floats represented as strings in the event itself
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def compute_hash(prev_hash: Optional[str], event_dict: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash for event in chain.

    h_i = SHA256(prev_hash || canonical_json(event without hash field))
    """
    # Create copy without hash field for hashing
    d = {k: v for k, v in event_dict.items() if k != "hash"}
    d["prev_hash"] = prev_hash  # Ensure prev_hash is included

    payload = (prev_hash or "") + canonical_json(d)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EventLogger:
    """
    Token guard event logger with hash chain integrity.

    Features:
    - Writes JSONL to configurable output
    - Maintains hash chain state
    - Enforces exactly-once per assembly_id
    - Assigns monotonic event_seq
    """

    def __init__(
        self,
        output: Optional[TextIO] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize event logger.

        Args:
            output: File-like object for JSONL output (None = buffer only)
            run_id: Unique run identifier (auto-generated if None)
        """
        self._output = output
        self._run_id = run_id or str(uuid.uuid4())
        self._event_seq = 0
        self._prev_hash: Optional[str] = None
        self._emitted_assembly_ids: Set[str] = set()
        self._events: List[TokenGuardEvent] = []  # In-memory buffer
        self._lock = threading.Lock()

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run_id

    @property
    def events(self) -> List[TokenGuardEvent]:
        """Get buffered events (copy)."""
        with self._lock:
            return list(self._events)

    def emit(
        self,
        event_type: EventType,
        assembly_id: str,
        turn_id: str,
        counter_backend: str,
        counter_exact: bool,
        applied_safety_factor: Optional[float],
        raw_tokens: int,
        effective_tokens: int,
        cap: int,
        budget_breakdown: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None
    ) -> TokenGuardEvent:
        """
        Emit a token guard event.

        Args:
            event_type: The type of event (ok, recovered, abort)
            assembly_id: Unique ID for this assembly call
            turn_id: Unique ID for the conversation turn
            counter_backend: Name of token counter backend
            counter_exact: Whether counter is exact
            applied_safety_factor: Safety factor applied (or None)
            raw_tokens: Raw token count before factor
            effective_tokens: Effective tokens after factor
            cap: Token cap used
            budget_breakdown: Token breakdown by component
            extra: Additional fields (forward compatibility)

        Returns:
            The emitted event

        Raises:
            ValueError: If assembly_id has already emitted an event
        """
        with self._lock:
            # Enforce exactly-once per assembly_id
            if assembly_id in self._emitted_assembly_ids:
                raise ValueError(
                    f"Event already emitted for assembly_id={assembly_id}. "
                    "Exactly-once constraint violated."
                )

            # Increment monotonic sequence
            self._event_seq += 1
            event_seq = self._event_seq

            # Create timestamp
            ts = datetime.now(timezone.utc).isoformat()

            # Convert safety factor to string for determinism
            sf_str = f"{applied_safety_factor:.6f}" if applied_safety_factor is not None else None

            # Build event dict for hashing (without hash field)
            event_dict = {
                "schema_version": SCHEMA_VERSION,
                "event_type": event_type.value,
                "run_id": self._run_id,
                "turn_id": turn_id,
                "assembly_id": assembly_id,
                "ts": ts,
                "event_seq": event_seq,
                "counter_backend": counter_backend,
                "counter_exact": counter_exact,
                "applied_safety_factor": sf_str,
                "raw_tokens": raw_tokens,
                "effective_tokens": effective_tokens,
                "cap": cap,
                "budget_breakdown": budget_breakdown,
                "prev_hash": self._prev_hash,
            }

            # Add extra fields
            if extra:
                event_dict.update(extra)

            # Compute hash
            event_hash = compute_hash(self._prev_hash, event_dict)
            event_dict["hash"] = event_hash

            # Create event object
            event = TokenGuardEvent.from_dict(event_dict)

            # Update state
            self._prev_hash = event_hash
            self._emitted_assembly_ids.add(assembly_id)
            self._events.append(event)

            # Write to output if configured
            if self._output is not None:
                self._output.write(canonical_json(event.to_dict()) + "\n")
                self._output.flush()

            return event

    def reset(self) -> None:
        """Reset logger state (for testing)."""
        with self._lock:
            self._event_seq = 0
            self._prev_hash = None
            self._emitted_assembly_ids.clear()
            self._events.clear()


class EventVerifier:
    """
    Verifies token guard event streams.

    Checks:
    - Required fields present
    - Hash chain integrity
    - Exactly-once per assembly_id
    - Monotonic event_seq
    """

    @staticmethod
    def verify_required_fields(event: Dict[str, Any]) -> List[str]:
        """
        Verify all required fields are present.

        Returns:
            List of missing field names (empty if valid)
        """
        required = TokenGuardEvent.required_fields()
        return [f for f in required if f not in event]

    @staticmethod
    def verify_hash(event: Dict[str, Any], expected_prev_hash: Optional[str]) -> bool:
        """
        Verify event hash is correct.

        Args:
            event: Event dictionary
            expected_prev_hash: Expected prev_hash (from previous event)

        Returns:
            True if hash is valid
        """
        if event.get("prev_hash") != expected_prev_hash:
            return False

        expected_hash = compute_hash(expected_prev_hash, event)
        return event.get("hash") == expected_hash

    @staticmethod
    def verify_stream(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify an entire event stream.

        Returns:
            {
                "valid": bool,
                "errors": List[str],
                "event_count": int,
                "assembly_ids": List[str]
            }
        """
        errors = []
        assembly_ids = set()
        prev_hash = None
        prev_seq = 0
        prev_run_id = None

        for i, event in enumerate(events):
            # Check required fields
            missing = EventVerifier.verify_required_fields(event)
            if missing:
                errors.append(f"Event {i}: missing required fields: {missing}")
                continue

            # Check hash chain
            if not EventVerifier.verify_hash(event, prev_hash):
                errors.append(f"Event {i}: hash chain verification failed")

            # Check monotonic event_seq per run_id
            run_id = event.get("run_id")
            event_seq = event.get("event_seq")

            if prev_run_id is not None and run_id == prev_run_id:
                if event_seq <= prev_seq:
                    errors.append(
                        f"Event {i}: event_seq not monotonic "
                        f"(got {event_seq}, prev {prev_seq})"
                    )

            # Check exactly-once per assembly_id
            assembly_id = event.get("assembly_id")
            if assembly_id in assembly_ids:
                errors.append(
                    f"Event {i}: duplicate assembly_id={assembly_id}, "
                    "exactly-once constraint violated"
                )
            assembly_ids.add(assembly_id)

            # Update state for next iteration
            prev_hash = event.get("hash")
            prev_seq = event_seq
            prev_run_id = run_id

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "event_count": len(events),
            "assembly_ids": list(assembly_ids)
        }


# Global event logger instance (lazy initialization)
_global_logger: Optional[EventLogger] = None
_global_logger_lock = threading.Lock()


def get_event_logger() -> EventLogger:
    """Get or create the global event logger."""
    global _global_logger
    with _global_logger_lock:
        if _global_logger is None:
            _global_logger = EventLogger()
        return _global_logger


def reset_event_logger() -> None:
    """Reset the global event logger (for testing)."""
    global _global_logger
    with _global_logger_lock:
        if _global_logger is not None:
            _global_logger.reset()
        _global_logger = None


def configure_event_logger(
    output: Optional[TextIO] = None,
    run_id: Optional[str] = None
) -> EventLogger:
    """
    Configure the global event logger.

    Args:
        output: File-like object for JSONL output
        run_id: Unique run identifier

    Returns:
        The configured logger
    """
    global _global_logger
    with _global_logger_lock:
        _global_logger = EventLogger(output=output, run_id=run_id)
        return _global_logger
