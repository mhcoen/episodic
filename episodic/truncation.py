"""
Relevance-Aware Truncation for Episodic.

Phase 2 Implementation:
- Importance scoring: score(e) = 100*I_anchor + 3*I_early + 2*lex_sim + 5*I_referenced
- Score-based drop policy: recency-only first (ascending score), then anchors
- Reference detection: quote overlap >= 40 chars OR explicit markers + 6 shared tokens
- Deterministic: must not break replay, must log truncation decisions
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from difflib import SequenceMatcher


@dataclass
class MessageScore:
    """
    Importance score for a message in the context.
    """
    index: int
    tokens: int
    score: float

    # Score components (for logging/debugging)
    is_anchor: bool = False
    is_early: bool = False
    lex_similarity: float = 0.0
    is_referenced: bool = False

    # Message metadata
    role: str = ""
    content_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "index": self.index,
            "tokens": self.tokens,
            "score": self.score,
            "components": {
                "is_anchor": self.is_anchor,
                "is_early": self.is_early,
                "lex_similarity": self.lex_similarity,
                "is_referenced": self.is_referenced,
            },
            "role": self.role,
            "content_preview": self.content_preview,
        }


@dataclass
class TruncationDecision:
    """
    Record of a truncation decision for logging.
    """
    message_index: int
    tokens_freed: int
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "message_index": self.message_index,
            "tokens_freed": self.tokens_freed,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass
class TruncationResult:
    """
    Result of relevance-aware truncation.
    """
    messages: List[Dict[str, Any]]
    tokens_before: int
    tokens_after: int
    decisions: List[TruncationDecision] = field(default_factory=list)
    scores: List[MessageScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_freed": self.tokens_before - self.tokens_after,
            "decisions": [d.to_dict() for d in self.decisions],
            "scores": [s.to_dict() for s in self.scores],
        }


# =============================================================================
# Score Weights (from spec)
# =============================================================================

WEIGHT_ANCHOR = 100  # I_anchor: from anchor section
WEIGHT_EARLY = 3     # I_early: first 2 exchanges
WEIGHT_LEX_SIM = 2   # lex_sim: lexical similarity to query
WEIGHT_REFERENCED = 5  # I_referenced: referenced in current turn


# =============================================================================
# Reference Detection
# =============================================================================

# Explicit reference markers
REFERENCE_MARKERS = [
    r"you\s+said",
    r"you\s+mentioned",
    r"you\s+told\s+me",
    r"we\s+discussed",
    r"we\s+agreed",
    r"as\s+we\s+talked\s+about",
    r"earlier\s+you",
    r"you\s+asked",
    r"i\s+remember",
]

COMPILED_REFERENCE_MARKERS = [re.compile(p, re.IGNORECASE) for p in REFERENCE_MARKERS]


def _find_longest_common_substring(a: str, b: str) -> int:
    """Find length of longest common substring between two strings."""
    if not a or not b:
        return 0

    # Use SequenceMatcher to find matching blocks
    s = SequenceMatcher(None, a.lower(), b.lower())
    match = s.find_longest_match(0, len(a), 0, len(b))
    return match.size


def _extract_key_words(text: str) -> Set[str]:
    """Extract key words from text (non-stop words)."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "that", "this", "these", "those", "it", "its", "to", "for", "of",
        "in", "on", "at", "by", "with", "from", "about", "into", "through",
        "over", "under", "above", "below", "between", "after", "before",
        "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
        "not", "only", "own", "same", "than", "too", "very", "just", "also",
        "now", "here", "there", "when", "where", "why", "how", "what", "which",
        "who", "whom", "whose", "if", "then", "else", "because", "although",
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they",
        "them", "his", "her", "their", "him", "its",
        "said", "something", "like", "uses",  # Common filler words in references
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in stop_words}


def detect_reference(
    current_turn: str,
    message_content: str,
    min_quote_overlap: int = 40,
    min_shared_tokens: int = 6,
) -> bool:
    """
    Detect if the current turn references a previous message.

    Reference is detected if:
    1. Quote overlap >= min_quote_overlap chars (40 default)
    OR
    2. Explicit reference markers present AND >= min_shared_tokens shared key words

    Args:
        current_turn: The current user message text
        message_content: Content of a previous message
        min_quote_overlap: Minimum chars for quote overlap detection
        min_shared_tokens: Minimum shared key words with markers

    Returns:
        True if the message appears to be referenced
    """
    if not current_turn or not message_content:
        return False

    # Check 1: Quote overlap >= 40 chars
    overlap = _find_longest_common_substring(current_turn, message_content)
    if overlap >= min_quote_overlap:
        return True

    # Check 2: Explicit markers + shared tokens
    has_marker = any(p.search(current_turn) for p in COMPILED_REFERENCE_MARKERS)
    if has_marker:
        current_words = _extract_key_words(current_turn)
        message_words = _extract_key_words(message_content)
        shared = current_words & message_words
        if len(shared) >= min_shared_tokens:
            return True

    return False


# =============================================================================
# Lexical Similarity
# =============================================================================

def compute_lexical_similarity(query: str, message_content: str) -> float:
    """
    Compute lexical similarity between query and message.

    Returns a value between 0 and 1.
    """
    if not query or not message_content:
        return 0.0

    query_words = _extract_key_words(query)
    message_words = _extract_key_words(message_content)

    if not query_words or not message_words:
        return 0.0

    # Jaccard similarity
    intersection = len(query_words & message_words)
    union = len(query_words | message_words)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# Message Scoring
# =============================================================================

def _is_from_anchor_section(message: Dict[str, Any], anchor_indices: Set[int], index: int) -> bool:
    """Check if a message is from the anchor section."""
    return index in anchor_indices


def _is_early_exchange(index: int, total_exchanges: int, early_count: int = 2) -> bool:
    """
    Check if message is in the first N exchanges.

    An exchange is a user/assistant pair.
    """
    # First 2 exchanges = first 4 messages (2 user + 2 assistant)
    return index < early_count * 2


def score_message(
    message: Dict[str, Any],
    index: int,
    current_query: str,
    anchor_indices: Set[int],
    total_recency_count: int,
    referenced_indices: Set[int],
) -> MessageScore:
    """
    Compute importance score for a message.

    score(e) = 100*I_anchor + 3*I_early + 2*lex_sim + 5*I_referenced

    Args:
        message: The message dict
        index: Index in message list
        current_query: Current user query for similarity
        anchor_indices: Set of indices that are from anchors
        total_recency_count: Total number of recency messages
        referenced_indices: Set of indices that are referenced

    Returns:
        MessageScore with computed importance
    """
    content = message.get("content", "")
    if isinstance(content, list):
        # Multimodal - extract text
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    # Compute indicators
    is_anchor = index in anchor_indices
    is_early = _is_early_exchange(index, total_recency_count)
    lex_sim = compute_lexical_similarity(current_query, content)
    is_referenced = index in referenced_indices

    # Compute score
    score = (
        WEIGHT_ANCHOR * (1 if is_anchor else 0) +
        WEIGHT_EARLY * (1 if is_early else 0) +
        WEIGHT_LEX_SIM * lex_sim +
        WEIGHT_REFERENCED * (1 if is_referenced else 0)
    )

    return MessageScore(
        index=index,
        tokens=0,  # Filled in later
        score=score,
        is_anchor=is_anchor,
        is_early=is_early,
        lex_similarity=lex_sim,
        is_referenced=is_referenced,
        role=message.get("role", ""),
        content_preview=content[:50] + "..." if len(content) > 50 else content,
    )


def score_messages(
    messages: List[Dict[str, Any]],
    current_query: str,
    anchor_indices: Optional[Set[int]] = None,
) -> List[MessageScore]:
    """
    Score all messages for importance.

    Args:
        messages: List of message dicts
        current_query: Current user query
        anchor_indices: Set of indices that are anchors (None = auto-detect)

    Returns:
        List of MessageScore objects
    """
    if anchor_indices is None:
        anchor_indices = set()

    # Detect referenced messages
    referenced_indices = set()
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if detect_reference(current_query, content):
            referenced_indices.add(i)

    # Count recency messages (non-system, non-current-query)
    recency_count = sum(
        1 for i, m in enumerate(messages)
        if m.get("role") != "system" and not (i == len(messages) - 1 and m.get("role") == "user")
    )

    # Score each message
    scores = []
    for i, msg in enumerate(messages):
        # Skip system messages and current query
        if msg.get("role") == "system":
            continue
        if i == len(messages) - 1 and msg.get("role") == "user":
            continue

        score = score_message(
            msg, i, current_query, anchor_indices, recency_count, referenced_indices
        )
        scores.append(score)

    return scores


# =============================================================================
# Score-Based Drop Policy
# =============================================================================

def drop_by_importance(
    messages: List[Dict[str, Any]],
    target_reduction: int,
    current_query: str,
    counter: Any,  # TokenCounter
    anchor_indices: Optional[Set[int]] = None,
) -> TruncationResult:
    """
    Drop messages by importance score (lowest first).

    Drop order:
    1. Recency-only messages first (ascending score)
    2. Then anchor messages if still over
    3. Ties broken by older-first (lower index)

    Args:
        messages: List of message dicts
        target_reduction: Target tokens to reduce
        current_query: Current user query
        counter: TokenCounter for counting
        anchor_indices: Set of indices that are anchors

    Returns:
        TruncationResult with modified messages and decisions
    """
    from episodic.token_guard import HeuristicTokenCounter

    if anchor_indices is None:
        anchor_indices = set()

    # Score all messages
    scores = score_messages(messages, current_query, anchor_indices)

    # Add token counts to scores
    for score in scores:
        msg = messages[score.index]
        if isinstance(counter, HeuristicTokenCounter):
            score.tokens = counter.count_message(msg)
        else:
            score.tokens = counter.count_messages([msg])

    # Separate recency-only from anchors
    recency_scores = [s for s in scores if not s.is_anchor]
    anchor_scores = [s for s in scores if s.is_anchor]

    # Sort by score ascending, then by index ascending (older first for ties)
    recency_scores.sort(key=lambda s: (s.score, s.index))
    anchor_scores.sort(key=lambda s: (s.score, s.index))

    # Track what to drop
    indices_to_drop = set()
    tokens_freed = 0
    decisions = []

    # Phase 1: Drop recency messages (lowest score first)
    for score in recency_scores:
        if tokens_freed >= target_reduction:
            break
        indices_to_drop.add(score.index)
        tokens_freed += score.tokens
        decisions.append(TruncationDecision(
            message_index=score.index,
            tokens_freed=score.tokens,
            score=score.score,
            reason="recency_low_score",
        ))

    # Phase 2: Drop anchor messages if still not enough
    if tokens_freed < target_reduction:
        for score in anchor_scores:
            if tokens_freed >= target_reduction:
                break
            indices_to_drop.add(score.index)
            tokens_freed += score.tokens
            decisions.append(TruncationDecision(
                message_index=score.index,
                tokens_freed=score.tokens,
                score=score.score,
                reason="anchor_low_score",
            ))

    # Calculate tokens before
    tokens_before = sum(s.tokens for s in scores)

    # Build new message list
    new_messages = [
        msg for i, msg in enumerate(messages)
        if i not in indices_to_drop
    ]

    # Calculate tokens after
    tokens_after = tokens_before - tokens_freed

    return TruncationResult(
        messages=new_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        decisions=decisions,
        scores=scores,
    )


def truncate_by_relevance(
    messages: List[Dict[str, Any]],
    target_tokens: int,
    current_query: str,
    counter: Any,  # TokenCounter
    anchor_indices: Optional[Set[int]] = None,
) -> TruncationResult:
    """
    Truncate messages to fit within target tokens using relevance scoring.

    Args:
        messages: List of message dicts
        target_tokens: Maximum tokens allowed
        current_query: Current user query
        counter: TokenCounter for counting
        anchor_indices: Set of indices that are anchors

    Returns:
        TruncationResult with truncated messages
    """

    # Count current tokens
    current_tokens = counter.count_messages(messages)

    if current_tokens <= target_tokens:
        # No truncation needed
        scores = score_messages(messages, current_query, anchor_indices)
        return TruncationResult(
            messages=messages,
            tokens_before=current_tokens,
            tokens_after=current_tokens,
            decisions=[],
            scores=scores,
        )

    # Calculate how much to reduce
    target_reduction = current_tokens - target_tokens

    return drop_by_importance(
        messages=messages,
        target_reduction=target_reduction,
        current_query=current_query,
        counter=counter,
        anchor_indices=anchor_indices,
    )
