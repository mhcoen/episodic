"""
Maintenance operations for Episodic.

This package contains background/offline maintenance tasks that are
NOT on the hot path, such as:
- Topic summarization
- Index maintenance
- Cleanup operations
- Validation checks
"""

from .summarization import (
    SummaryResult,
    get_stale_topics,
    summarize_stale_topics,
    summarize_topic,
    summarize_topic_structured,
)
from .summary_spec import (
    SCHEMA_VERSION,
    SUMMARY_PROMPT,
    Decision,
    LastState,
    OpenLoop,
    StructuredSummary,
)
from .backfill_topic_metadata import (
    BackfillReport,
    backfill_topic_metadata,
    backfill_topic_metadata_with_report,
    format_backfill_report,
)
from .validators import (
    ValidationResult,
    format_validation_report,
    validate_all_topics,
    validate_anchor_invariants,
    validate_summary_provenance,
    validate_working_set_completeness,
)
from .verify_phase3 import (
    CheckResult,
    Phase3Report,
    run_all_verifications,
    save_report,
    verify_backfill_completeness,
    verify_long_gap_only,
    verify_summary_validity,
)

__all__ = [
    # Summarization
    "summarize_topic",
    "summarize_topic_structured",
    "summarize_stale_topics",
    "get_stale_topics",
    "SummaryResult",
    # Summary spec
    "SCHEMA_VERSION",
    "SUMMARY_PROMPT",
    "StructuredSummary",
    "Decision",
    "OpenLoop",
    "LastState",
    # Backfill
    "BackfillReport",
    "backfill_topic_metadata",
    "backfill_topic_metadata_with_report",
    "format_backfill_report",
    # Validators
    "ValidationResult",
    "validate_summary_provenance",
    "validate_anchor_invariants",
    "validate_all_topics",
    "validate_working_set_completeness",
    "format_validation_report",
    # Phase 3 Verification
    "CheckResult",
    "Phase3Report",
    "run_all_verifications",
    "verify_backfill_completeness",
    "verify_summary_validity",
    "verify_long_gap_only",
    "save_report",
]
