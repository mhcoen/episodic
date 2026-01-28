"""
Context recovery strategies for Episodic.

Provides different methods for assembling conversation context:
- ancestry: Traditional DAG ancestry traversal (existing behavior)
- topic_local: Topic-isolated context that excludes other topics
- hybrid: Switches strategy based on topic reactivation
"""

from .strategy import (
    ContextRecoveryMode,
    ContextAssemblyResult,
    ContextRecoveryStrategy,
    select_strategy,
)
from .ancestry import AncestryStrategy
from .topic_local import TopicLocalStrategy, ContaminationError
from .determinism import (
    ContextAssemblyFingerprint,
    compute_fingerprint,
    persist_fingerprint,
    diff_fingerprints,
    format_diff,
)

__all__ = [
    'ContextRecoveryMode',
    'ContextAssemblyResult',
    'ContextRecoveryStrategy',
    'select_strategy',
    'AncestryStrategy',
    'TopicLocalStrategy',
    'ContaminationError',
    'ContextAssemblyFingerprint',
    'compute_fingerprint',
    'persist_fingerprint',
    'diff_fingerprints',
    'format_diff',
]
