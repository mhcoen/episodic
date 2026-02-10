"""
Data classes for commitment-based topic boundary detection.

CommitmentPolicy: Configuration knobs for the commitment state machine.
CommitState: State machine state constants (STABLE/SUSPECT).
CommitmentState: Mutable runtime state for tracking evidence accumulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class CommitmentPolicy:
    """Configuration for boundary commitment decisions."""

    # Minimum turns between committed boundaries
    min_gap: int = 3

    # Threshold to enter SUSPECT state (first spike detection)
    suspect_threshold: float = 0.5

    # Threshold below which we consider returning to original topic (for ABORT)
    abort_threshold: float = 0.3

    # How many consecutive low-confidence turns before aborting SUSPECT state
    abort_streak: int = 3

    # Decay factor for evidence over turns (0-1)
    # 1.0 = no decay, 0.5 = halve each turn
    evidence_decay: float = 0.8

    # Minimum accumulated evidence to commit
    min_evidence: float = 1.2

    # Drift threshold for fast-path SUSPECT entry (hybrid trigger)
    # High embedding drift can trigger SUSPECT even if neural confidence is low
    # Set to None to disable drift fast-path
    drift_suspect_threshold: Optional[float] = 0.85

    # === Persistence requirement (two-sided test) ===
    # After evidence threshold is met, require K more turns with confidence
    # above return_threshold before committing. This catches transient digressions
    # where the user returns to the original topic.
    #
    # Commit requires BOTH:
    #   1. accumulated_evidence >= min_evidence
    #   2. last commit_persistence turns all had confidence >= return_threshold

    # Number of turns that must stay "away from topic" after evidence threshold
    # before we commit. 0 = commit immediately when evidence is met.
    commit_persistence: int = 1

    # Threshold for "returned to topic" - if confidence drops below this after
    # evidence threshold is met, we ABORT. Reuses abort_threshold by default.
    # Set to None to use abort_threshold.
    return_threshold: Optional[float] = None

    # Alternative return detection: relative drop from peak confidence.
    # If confidence drops to less than (peak * return_drop_ratio), ABORT.
    # This is more adaptive than an absolute threshold.
    # Example: peak=0.9, ratio=0.55 → abort if conf < 0.495
    # Set to None to use return_threshold instead.
    return_drop_ratio: Optional[float] = None

    # === Conditional cooldown ===
    # Don't apply persistence uniformly. High-confidence neural spikes are likely
    # real boundaries; only apply cooldown to intermediate cases that look like
    # potential digressions.
    #
    # Cooldown (K) applies when:
    #   - suspect_cause == "drift" (drift is noisy, needs confirmation), OR
    #   - confidence < high_conf_commit_threshold (intermediate band)
    #
    # Cooldown bypassed (immediate commit) when:
    #   - suspect_cause == "neural" AND confidence >= high_conf_commit_threshold
    #
    # Set to None to apply K uniformly (original behavior).
    high_conf_commit_threshold: Optional[float] = None

    # === Drift-triggered SUSPECT: confirmation requirements ===
    # Drift fires on surface-level changes (new entities, tangents) that may
    # not be real topic changes. Neural model must still confirm.

    # Minimum evidence for drift-triggered SUSPECT
    # Same as neural-triggered (1.2) since drift just triggers faster entry
    # while neural must still build evidence to commit
    drift_min_evidence: float = 1.2

    # Abort threshold for drift-triggered SUSPECT (higher = faster abort)
    # Default: 0.4 vs 0.3 for neural-triggered
    drift_abort_threshold: float = 0.4

    # Abort streak for drift-triggered SUSPECT
    # Higher than neural (4 vs 3) to give more time for neural to confirm
    # when drift triggers early on a genuine topic change
    drift_abort_streak: int = 4

    # === Drift + neural fast commit ===
    # When drift triggers SUSPECT AND neural confidence exceeds this threshold,
    # commit immediately (subject to min_gap). This handles cases where:
    # 1. Both signals agree (double confirmation)
    # 2. Neural model is sensitive to phrasing, so subsequent turns may give
    #    lower confidence despite the topic having clearly changed
    # Set to None to disable fast commit (require normal evidence accumulation)
    drift_neural_fast_commit_threshold: float = 0.9

    # === Neural commit drift gate ===
    # For neural-triggered SUSPECT, require semantic drift >= this threshold to COMMIT.
    # This prevents committing subtopic changes (like carbonara within pasta topic)
    # while still allowing large displacements (pasta → AI) to commit.
    # Only applies when suspect_cause == "neural", not drift-triggered SUSPECT.
    # Set to None to disable (allow neural commits without drift requirement).
    neural_commit_drift_threshold: Optional[float] = 0.7

    # === Early commit gate ===
    # Minimum user turns before allowing COMMIT (not detection).
    # This protects against cold-start noise and early conversational setup
    # while still allowing SUSPECT entry and evidence accumulation.
    #
    # Key distinction from detection gating:
    # - SUSPECT can be entered at any time (drift/neural triggers still work)
    # - Evidence accumulates with frozen reference
    # - Only COMMIT is blocked until enough user history exists
    #
    # Set to 0 to disable (allow commit at any time)
    min_user_turns_for_commit: int = 4


# State machine states
class CommitState:
    STABLE = "STABLE"    # Normal operation, sliding window
    SUSPECT = "SUSPECT"  # Potential topic change detected, frozen reference


@dataclass
class CommitmentState:
    """Tracks state for commitment decisions."""

    # Current state machine state
    state: str = CommitState.STABLE

    # Message index of last committed boundary (None if none committed)
    last_boundary_idx: Optional[int] = None

    # Current message index
    current_idx: int = 0

    # User turns since last confirmed boundary (for commit gate)
    # - Before any boundary: counts from conversation start
    # - After a boundary: counts from that confirmed boundary
    # This is anchored to confirmed boundaries only, not topic naming
    user_turns_since_boundary: int = 0

    # === SUSPECT state fields ===
    # Frozen "before" context captured when entering SUSPECT
    frozen_before: Optional[List[Dict[str, Any]]] = None

    # Frozen "straddle" message - the last message before the suspected topic change
    # This preserves the training format: after = [straddle_msg, query]
    frozen_straddle_msg: Optional[Dict[str, Any]] = None

    # Node ID where suspicion began (for boundary emit on commit)
    suspect_start_node_id: Optional[str] = None

    # Accumulated evidence while in SUSPECT state
    accumulated_evidence: float = 0.0

    # Count of consecutive low-confidence turns (for ABORT logic)
    low_confidence_streak: int = 0

    # Cause of SUSPECT entry: "neural" or "drift"
    # Used to apply cause-conditioned policy (stricter for drift)
    suspect_cause: Optional[str] = None

    # Recent confidence values for persistence check (two-sided test)
    # After evidence threshold is met, we track confidence values to ensure
    # the user hasn't "returned" to the original topic before committing.
    recent_confidences: List[float] = field(default_factory=list)

    # Whether evidence threshold has been met (waiting for persistence)
    evidence_met: bool = False

    # Peak confidence during SUSPECT for relative drop detection
    peak_confidence: float = 0.0

    # Entry confidence when SUSPECT was entered (for bypass decision)
    # Bypass uses entry confidence, not commit-time confidence
    suspect_entry_confidence: float = 0.0

    # Whether we've committed at least one boundary in this conversation
    # Used to make min_user_turns_for_commit apply only during cold start
    has_committed_boundary: bool = False
