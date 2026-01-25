"""
Smart context detection for RAG memory system.
Unified referential detection with explicit and implicit pattern matching.

This module provides the canonical detect_recall_intent() function that
determines when conversation memory should be retrieved.
"""

from typing import Dict, Tuple, Optional
import re

from episodic.config import config
from episodic.debug_utils import debug_print


class SmartContextDetector:
    """Detects when context retrieval would be helpful.

    Combines explicit referential markers (e.g., "we discussed", "you mentioned")
    with implicit pattern detection (e.g., "and then", short follow-ups).
    """

    def __init__(self):
        # Explicit referential phrases (high confidence = 0.9)
        # Includes both statement and question forms
        self.explicit_phrases = [
            "we discussed", "we talked about", "you mentioned", "you said",
            "remember when", "last time", "previously", "earlier",
            "what was that", "you told me", "we covered", "go back to",
            "that command", "that example", "that explanation",
            # Question forms for recall queries
            "did we discuss", "did we talk about", "what did we discuss",
            "what did we talk", "remind me what", "remind me about",
            "what were we", "were we talking"
        ]

        # Implicit markers that suggest past context needed (medium confidence = 0.7)
        self.implicit_markers = ["what was", "which one", "that one", "those"]

        # Implicit patterns with associated confidence scores
        self.implicit_patterns = [
            # Follow-up patterns
            (r'^(and|but|also|however|although)\s+', 0.6),
            (r'^(what about|how about|and if)\s+', 0.7),
            (r'^(ok|okay|alright|got it|i see)[,.]?\s+(now|so|then)', 0.5),

            # Continuation patterns
            (r'^(continue|go on|keep going|next)', 0.8),
            (r'^(more|another|different)\s+(example|way|method)', 0.6),

            # Clarification patterns
            (r'^(wait|hold on|actually)', 0.5),
            (r'(instead|rather than|not that)', 0.6),
            (r'^(oh|hmm|uh),?\s+', 0.4),

            # Topic continuation
            (r'(in that case|if so|given that)', 0.7),
            (r'^(for|with|using)\s+(that|this|those)', 0.6),

            # Assumed context
            (r'^(try|test|run|execute|implement)\s+it', 0.8),
            (r'^(fix|update|modify|change)\s+it', 0.8),
            (r'^make it\s+', 0.7),
        ]

        # Keywords that suggest topic continuation
        self.continuation_keywords = {
            'it', 'that', 'this', 'those', 'these',
            'same', 'similar', 'related', 'above',
            'there', 'here', 'now', 'then'
        }

    def detect_explicit_reference(self, query: str) -> Tuple[bool, float]:
        """
        Detect if a query explicitly references past conversations.

        Returns:
            Tuple of (is_explicit_reference, confidence)
        """
        query_lower = query.lower()

        # Check for explicit referential language (high confidence)
        for phrase in self.explicit_phrases:
            if phrase in query_lower:
                return True, 0.9

        # Check for implicit markers (medium confidence)
        for marker in self.implicit_markers:
            if query_lower.startswith(marker):
                return True, 0.7

        # Short questions might be referential (low confidence)
        if len(query.split()) <= 5 and "?" in query:
            return True, 0.5

        return False, 0.0

    def detect_implicit_reference(self, query: str) -> Tuple[bool, float]:
        """
        Detect if a query implicitly references context without explicit markers.

        Returns:
            Tuple of (is_implicit_reference, confidence)
        """
        query_lower = query.lower().strip()

        # Check pattern matches
        max_confidence = 0.0
        for pattern, confidence in self.implicit_patterns:
            if re.search(pattern, query_lower):
                max_confidence = max(max_confidence, confidence)

        # Check for continuation keywords without clear antecedents
        words = set(query_lower.split())
        keyword_matches = words.intersection(self.continuation_keywords)

        # Short queries with continuation keywords
        if len(words) < 8 and keyword_matches:
            keyword_confidence = 0.4 + (0.1 * len(keyword_matches))
            max_confidence = max(max_confidence, keyword_confidence)

        # Very short queries are often continuations
        if len(words) <= 3:
            max_confidence = max(max_confidence, 0.5)

        # Questions without context
        if query_lower.endswith('?') and len(words) < 6:
            max_confidence = max(max_confidence, 0.4)

        return max_confidence > 0.0, max_confidence

    def should_inject_context(
        self,
        query: str,
        conversation_state: Optional[Dict] = None
    ) -> Tuple[bool, float, str]:
        """
        Determine if context should be injected based on multiple factors.

        Args:
            query: User's current query
            conversation_state: Optional conversation state (topic, message count, etc.)

        Returns:
            Tuple of (should_inject, confidence, reason)
            reason is one of: "explicit_reference", "implicit_reference",
                             "topic_continuation", "no_context_needed"
        """
        if conversation_state is None:
            conversation_state = {}

        # Check explicit references first
        is_ref, ref_confidence = self.detect_explicit_reference(query)
        if is_ref and ref_confidence >= 0.7:
            return True, ref_confidence, "explicit_reference"

        # Check implicit references
        is_implicit, impl_confidence = self.detect_implicit_reference(query)

        # Boost confidence based on conversation state
        if is_implicit:
            # Recent topic change reduces confidence
            if conversation_state.get('messages_since_topic_change', 100) < 3:
                impl_confidence *= 0.7

            # Long conversation increases confidence
            if conversation_state.get('total_messages', 0) > 10:
                impl_confidence *= 1.2

            # Clamp to [0, 1]
            impl_confidence = min(1.0, impl_confidence)

            if impl_confidence >= 0.5:
                return True, impl_confidence, "implicit_reference"

        # Check for topic-specific triggers
        if self._check_topic_triggers(query, conversation_state):
            return True, 0.6, "topic_continuation"

        return False, 0.0, "no_context_needed"

    def _check_topic_triggers(self, query: str, conversation_state: Dict) -> bool:
        """Check if query relates to current topic in a way that needs context."""
        current_topic = conversation_state.get('current_topic_name', '')
        if not current_topic:
            return False

        # Topic-specific keywords that suggest continuation
        topic_keywords = {
            'python': ['pip', 'import', 'module', 'package', 'virtualenv', 'conda'],
            'git': ['commit', 'push', 'pull', 'branch', 'merge', 'checkout'],
            'docker': ['container', 'image', 'dockerfile', 'compose', 'volume'],
            'api': ['endpoint', 'request', 'response', 'header', 'auth', 'token'],
            'database': ['query', 'table', 'schema', 'index', 'migration', 'sql'],
        }

        # Find relevant keywords for current topic
        query_lower = query.lower()
        for topic_key, keywords in topic_keywords.items():
            if topic_key in current_topic.lower():
                if any(kw in query_lower for kw in keywords):
                    return True

        return False


# Global instance
smart_detector = SmartContextDetector()


def detect_recall_intent(
    query: str,
    conversation_state: Optional[Dict] = None
) -> Tuple[bool, float, str]:
    """
    Canonical function to detect if memory retrieval is needed.

    This is the single entry point for determining whether conversation
    memory should be searched based on the user's query.

    Args:
        query: User's current query
        conversation_state: Optional conversation state dictionary with:
            - current_topic_name: Name of current topic
            - messages_since_topic_change: Messages since last topic change
            - total_messages: Total messages in conversation

    Returns:
        Tuple of (should_retrieve, confidence, reason)
        - should_retrieve: Whether to search conversation memory
        - confidence: Confidence score (0.0 to 1.0)
        - reason: One of "explicit_reference", "implicit_reference",
                 "topic_continuation", "no_context_needed"
    """
    from episodic.debug_utils import debug_print

    should_retrieve, confidence, reason = smart_detector.should_inject_context(query, conversation_state)
    debug_print(f"detect_recall_intent: should_retrieve={should_retrieve}, confidence={confidence:.2f}, reason={reason}", category="memory")
    return should_retrieve, confidence, reason


def format_memory_indicator(
    memories_used: int,
    confidence: float,
    reason: str
) -> str:
    """Format a visual indicator for memory usage."""
    # Emoji based on confidence
    if confidence >= 0.8:
        emoji = "🧠"  # Strong memory connection
    elif confidence >= 0.6:
        emoji = "💭"  # Moderate memory connection
    else:
        emoji = "💡"  # Weak memory connection

    # Reason descriptions
    reason_text = {
        'explicit_reference': 'found relevant context',
        'implicit_reference': 'detected continuation',
        'topic_continuation': 'topic-related context'
    }.get(reason, 'added context')

    return f"{emoji} Memory: {reason_text} ({memories_used} items, {confidence:.0%} confidence)"


async def enhance_with_smart_context(
    user_input: str,
    conversation_state: Dict
) -> Tuple[Optional[str], Optional[str]]:
    """
    Enhanced context injection with smart detection.

    Returns:
        Tuple of (context_to_inject, memory_indicator)
    """
    # Lazy import to avoid circular dependency
    from episodic.rag_memory_sqlite import memory_rag

    # Check if we should inject context
    should_inject, confidence, reason = detect_recall_intent(
        user_input, conversation_state
    )

    if not should_inject:
        return None, None

    # Search for relevant memories
    memories = memory_rag.search_memories(user_input, limit=3)

    if not memories:
        return None, None

    # Filter by dynamic threshold based on confidence
    threshold = 0.8 - (confidence * 0.2)  # Higher confidence = lower threshold
    relevant_memories = [m for m in memories if m['relevance_score'] > threshold]

    if not relevant_memories:
        return None, None

    # Format context
    context = memory_rag.format_for_context(relevant_memories[:2])  # Max 2 items

    # Create indicator
    indicator = format_memory_indicator(len(relevant_memories), confidence, reason)

    if config.get("debug"):
        debug_print(f"{reason}: confidence={confidence:.2f}, threshold={threshold:.2f}", category="smart_memory")
        debug_print(f"Found {len(relevant_memories)} relevant memories", category="smart_memory")

    return context, indicator


# Configuration helpers
def get_memory_config() -> Dict:
    """Get memory system configuration with defaults."""
    return {
        'explicit_threshold': config.get('memory_explicit_threshold', 0.7),
        'implicit_threshold': config.get('memory_implicit_threshold', 0.5),
        'relevance_threshold': config.get('memory_relevance_threshold', 0.7),
        'max_memories_inject': config.get('memory_max_inject', 2),
        'show_indicators': config.get('memory_show_indicators', True),
    }


def set_memory_thresholds(
    explicit: float = 0.7,
    implicit: float = 0.5,
    relevance: float = 0.7
) -> None:
    """Set memory detection thresholds."""
    config.set('memory_explicit_threshold', explicit)
    config.set('memory_implicit_threshold', implicit)
    config.set('memory_relevance_threshold', relevance)

    print(f"Memory thresholds updated:")
    print(f"   Explicit references: {explicit:.0%}")
    print(f"   Implicit references: {implicit:.0%}")
    print(f"   Relevance scoring: {relevance:.0%}")
