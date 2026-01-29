"""
Structured summary specification for topic working sets.

Strictly parseable, evolvable, with enforced size limits.
This module defines the canonical format for topic summaries,
optimized for conversation continuation after long gaps.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Current schema version - increment when format changes
SCHEMA_VERSION = 1

# Size limits (characters)
MAX_CONTEXT_CHARS = 500
MAX_DECISION_CHARS = 200
MAX_OPEN_LOOP_CHARS = 200
MAX_LAST_STATE_CHARS = 300
MAX_DECISIONS = 10
MAX_OPEN_LOOPS = 10


@dataclass
class Decision:
    """A decision with stable ID for tracking."""

    id: str  # Stable ID for updates/closures (e.g., "d_001")
    decision: str
    rationale: Optional[str] = None
    status: str = "active"  # "active", "superseded", "closed"

    def __post_init__(self):
        """Enforce size limits."""
        if len(self.decision) > MAX_DECISION_CHARS:
            self.decision = self.decision[:MAX_DECISION_CHARS]


@dataclass
class OpenLoop:
    """An open question/thread with stable ID."""

    id: str  # Stable ID (e.g., "ol_001")
    question: str
    next_step: Optional[str] = None
    priority: str = "normal"  # "high", "normal", "low"

    def __post_init__(self):
        """Enforce size limits."""
        if len(self.question) > MAX_OPEN_LOOP_CHARS:
            self.question = self.question[:MAX_OPEN_LOOP_CHARS]


@dataclass
class LastState:
    """Structured last state with stable keys."""

    current_plan: Optional[str] = None
    constraints: Optional[str] = None
    active_files: Optional[List[str]] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Enforce size limits."""
        if self.current_plan and len(self.current_plan) > MAX_LAST_STATE_CHARS:
            self.current_plan = self.current_plan[:MAX_LAST_STATE_CHARS]
        if self.notes and len(self.notes) > MAX_LAST_STATE_CHARS:
            self.notes = self.notes[:MAX_LAST_STATE_CHARS]


@dataclass
class StructuredSummary:
    """Summary format optimized for conversation continuation."""

    schema_version: int
    context: str  # 1-3 sentences, no bullets
    decisions: List[Decision] = field(default_factory=list)
    open_loops: List[OpenLoop] = field(default_factory=list)
    last_state: LastState = field(default_factory=LastState)

    def __post_init__(self):
        """Enforce size limits."""
        if len(self.context) > MAX_CONTEXT_CHARS:
            self.context = self.context[:MAX_CONTEXT_CHARS]
        self.decisions = self.decisions[:MAX_DECISIONS]
        self.open_loops = self.open_loops[:MAX_OPEN_LOOPS]

    def to_canonical_json(self) -> str:
        """Canonical JSON for hashing (sorted keys, no extra whitespace)."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """SHA256 hash of canonical JSON."""
        return hashlib.sha256(self.to_canonical_json().encode()).hexdigest()[:16]

    def to_markdown(self) -> str:
        """Convert to markdown for inclusion in context."""
        lines = []

        if self.context:
            lines.append(f"**Context:** {self.context}")

        active_decisions = [d for d in self.decisions if d.status == "active"]
        if active_decisions:
            lines.append("\n**Decisions:**")
            for d in active_decisions:
                lines.append(f"- {d.decision}")

        if self.open_loops:
            lines.append("\n**Open questions:**")
            for o in self.open_loops:
                prefix = "[HIGH] " if o.priority == "high" else ""
                lines.append(f"- {prefix}{o.question}")

        if self.last_state.current_plan:
            lines.append(f"\n**Current plan:** {self.last_state.current_plan}")

        return "\n".join(lines)

    @classmethod
    def from_json(cls, json_str: str) -> "StructuredSummary":
        """Parse from JSON string."""
        data = json.loads(json_str)

        # Validate schema version
        if data.get("schema_version", 0) > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {data.get('schema_version')}"
            )

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            context=data.get("context", ""),
            decisions=[Decision(**d) for d in data.get("decisions", [])],
            open_loops=[OpenLoop(**o) for o in data.get("open_loops", [])],
            last_state=LastState(**data.get("last_state", {})),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredSummary":
        """Create from dictionary."""
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            context=data.get("context", ""),
            decisions=[
                Decision(**d) if isinstance(d, dict) else d
                for d in data.get("decisions", [])
            ],
            open_loops=[
                OpenLoop(**o) if isinstance(o, dict) else o
                for o in data.get("open_loops", [])
            ],
            last_state=(
                LastState(**data.get("last_state", {}))
                if isinstance(data.get("last_state"), dict)
                else data.get("last_state") or LastState()
            ),
        )


# Prompt template for generating structured summaries
SUMMARY_PROMPT = '''Analyze this conversation segment and produce a structured summary as JSON.

CONVERSATION:
{exchanges}

Produce a JSON object with this exact structure:
{{
  "schema_version": 1,
  "context": "1-3 sentences describing what this conversation is about (max 500 chars)",
  "decisions": [
    {{"id": "d_001", "decision": "what was decided", "rationale": "why", "status": "active"}}
  ],
  "open_loops": [
    {{"id": "ol_001", "question": "unresolved question", "next_step": "suggested action", "priority": "normal"}}
  ],
  "last_state": {{
    "current_plan": "where we left off",
    "constraints": "any constraints established",
    "notes": "other relevant state"
  }}
}}

Rules:
- Be concise. Max 10 decisions, 10 open_loops.
- Use stable IDs (d_001, d_002, etc.) for decisions and open_loops.
- Focus on information needed to continue this conversation later.
- Output ONLY valid JSON, no markdown or explanation.

JSON:'''
