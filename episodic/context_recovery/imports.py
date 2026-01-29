"""
Cross-topic context imports.

Detects when user explicitly references another topic and fetches relevant
context for injection without breaking topic isolation.
"""

import re
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)

# Patterns for detecting import intent
IMPORT_PATTERNS = [
    # "as we discussed about X" / "as we talked about X"
    r"as we (?:discussed|talked) (?:about|regarding)\s+(.+?)(?:\.|,|$|\?)",
    # "remember when we talked about X"
    r"remember (?:when|that) we (?:talked|discussed|chatted) about\s+(.+?)(?:\.|,|$|\?)",
    # "going back to what you said about X"
    r"going back to (?:what you said|our discussion|our conversation) (?:about|on|regarding)\s+(.+?)(?:\.|,|$|\?)",
    # "like in our conversation about X"
    r"like (?:in|from) (?:our|the) (?:conversation|discussion|chat) (?:about|on|regarding)\s+(.+?)(?:\.|,|$|\?)",
    # "you mentioned X earlier" / "we mentioned X"
    r"(?:you|we) mentioned\s+(.+?)\s+(?:earlier|before|previously)",
    # "recall when we discussed X"
    r"recall (?:when|that) we (?:discussed|talked about)\s+(.+?)(?:\.|,|$|\?)",
    # "what did you say about X"
    r"what did (?:you|we) say about\s+(.+?)(?:\?|$)",
    # "from our X conversation/discussion"
    r"from (?:our|the)\s+(.+?)\s+(?:conversation|discussion)",
    # "in the X topic/discussion"
    r"in the\s+(.+?)\s+(?:topic|discussion|conversation)",
]

# Compiled patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in IMPORT_PATTERNS]


@dataclass
class ImportIntent:
    """Result of import intent detection."""
    has_intent: bool
    topic_reference: Optional[str] = None
    pattern_matched: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ImportTarget:
    """Resolved import target topic."""
    topic_start_node_id: str
    topic_name: str
    confidence: float
    match_method: str  # "exact", "fuzzy", "semantic"


@dataclass
class ImportContext:
    """Fetched context from another topic."""
    context_block: str
    topic_name: str
    debug: Dict[str, Any] = field(default_factory=dict)


def detect_import_intent(user_input: str) -> ImportIntent:
    """
    Detect if user is explicitly referencing another topic.

    Args:
        user_input: The user's message text

    Returns:
        ImportIntent with detection result and extracted topic reference
    """
    if not user_input or len(user_input) < 10:
        return ImportIntent(has_intent=False)

    for i, pattern in enumerate(COMPILED_PATTERNS):
        match = pattern.search(user_input)
        if match:
            topic_ref = match.group(1).strip()
            # Clean up the reference
            topic_ref = _clean_topic_reference(topic_ref)

            if topic_ref and len(topic_ref) >= 2:
                return ImportIntent(
                    has_intent=True,
                    topic_reference=topic_ref,
                    pattern_matched=IMPORT_PATTERNS[i],
                    confidence=0.8  # High confidence for explicit patterns
                )

    return ImportIntent(has_intent=False)


def _clean_topic_reference(topic_ref: str) -> str:
    """Clean up extracted topic reference."""
    # Remove trailing punctuation and common words
    topic_ref = topic_ref.strip().rstrip('.,;:!?')

    # Remove leading articles
    for article in ['the ', 'a ', 'an ']:
        if topic_ref.lower().startswith(article):
            topic_ref = topic_ref[len(article):]

    # Remove common trailing words
    for trailing in [' earlier', ' before', ' previously', ' thing', ' stuff']:
        if topic_ref.lower().endswith(trailing):
            topic_ref = topic_ref[:-len(trailing)]

    return topic_ref.strip()


def _get_all_topics_with_names(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all topics with their names."""
    cursor = conn.execute("""
        SELECT name, start_node_id, end_node_id
        FROM topics
        ORDER BY rowid DESC
    """)

    topics = []
    for row in cursor.fetchall():
        topics.append({
            'name': row[0],
            'start_node_id': row[1],
            'end_node_id': row[2]
        })
    return topics


def _fuzzy_match_score(ref: str, topic_name: str) -> float:
    """Compute fuzzy match score between reference and topic name."""
    ref_lower = ref.lower()
    name_lower = topic_name.lower()

    # Exact match
    if ref_lower == name_lower:
        return 1.0

    # Reference is substring of topic name
    if ref_lower in name_lower:
        return 0.8 + (len(ref_lower) / len(name_lower)) * 0.2

    # Topic name is substring of reference
    if name_lower in ref_lower:
        return 0.7 + (len(name_lower) / len(ref_lower)) * 0.2

    # Word overlap
    ref_words = set(ref_lower.split())
    name_words = set(name_lower.split())

    if not ref_words or not name_words:
        return 0.0

    overlap = ref_words & name_words
    if overlap:
        jaccard = len(overlap) / len(ref_words | name_words)
        return 0.5 + jaccard * 0.4

    return 0.0


def resolve_import_target(
    topic_reference: str,
    active_topic_start_node_id: Optional[str],
    user_embedding: Optional[np.ndarray],
    conn: Optional[sqlite3.Connection] = None
) -> Optional[ImportTarget]:
    """
    Find the topic being referenced.

    Args:
        topic_reference: Extracted topic reference from user input
        active_topic_start_node_id: Currently active topic (to exclude)
        user_embedding: Embedding of current user input (for semantic matching)
        conn: Optional database connection

    Returns:
        ImportTarget with resolved topic info, or None if not found
    """
    def _resolve(c: sqlite3.Connection) -> Optional[ImportTarget]:
        topics = _get_all_topics_with_names(c)

        if not topics:
            return None

        # Try fuzzy matching first
        best_match = None
        best_score = 0.0

        for topic in topics:
            # Skip active topic
            if topic['start_node_id'] == active_topic_start_node_id:
                continue

            score = _fuzzy_match_score(topic_reference, topic['name'])
            if score > best_score and score >= 0.5:  # Minimum threshold
                best_score = score
                best_match = topic

        if best_match:
            return ImportTarget(
                topic_start_node_id=best_match['start_node_id'],
                topic_name=best_match['name'],
                confidence=best_score,
                match_method="fuzzy" if best_score < 1.0 else "exact"
            )

        # If no fuzzy match and we have embedding, try semantic matching
        if user_embedding is not None:
            semantic_target = _semantic_match_topic(
                topic_reference,
                user_embedding,
                [t for t in topics if t['start_node_id'] != active_topic_start_node_id],
                c
            )
            if semantic_target:
                return semantic_target

        return None

    if conn is not None:
        return _resolve(conn)

    with get_connection() as c:
        return _resolve(c)


def _semantic_match_topic(
    topic_reference: str,
    user_embedding: np.ndarray,
    topics: List[Dict[str, Any]],
    conn: sqlite3.Connection
) -> Optional[ImportTarget]:
    """Try to match topic using semantic similarity."""
    try:
        from episodic.rag_collections import get_multi_collection_rag, CollectionType
        from episodic.recall.reactivation import _compute_similarity

        # Get centroids for topics
        cursor = conn.execute("""
            SELECT tc.start_node_id, tc.centroid_medoid_exchange_id
            FROM topic_centroids tc
            WHERE tc.start_node_id IN ({})
        """.format(','.join('?' * len(topics))), [t['start_node_id'] for t in topics])

        centroid_map = {row[0]: row[1] for row in cursor.fetchall()}

        if not centroid_map:
            return None

        # Get centroid embeddings
        rag = get_multi_collection_rag()
        collection = rag.get_collection(CollectionType.CONVERSATION)

        centroid_ids = list(centroid_map.values())
        result = collection.get(ids=centroid_ids, include=['embeddings'])

        if not result or not result.get('ids') or not result.get('embeddings'):
            return None

        # Find best semantic match
        best_topic = None
        best_sim = 0.0

        for i, node_id in enumerate(result['ids']):
            emb = result['embeddings'][i]
            if emb is None:
                continue

            sim = _compute_similarity(user_embedding, np.array(emb))

            if sim > best_sim and sim >= 0.4:  # Threshold for semantic match
                # Find which topic this centroid belongs to
                for start_id, centroid_id in centroid_map.items():
                    if centroid_id == node_id:
                        for topic in topics:
                            if topic['start_node_id'] == start_id:
                                best_sim = sim
                                best_topic = topic
                                break

        if best_topic:
            return ImportTarget(
                topic_start_node_id=best_topic['start_node_id'],
                topic_name=best_topic['name'],
                confidence=best_sim,
                match_method="semantic"
            )

    except Exception as e:
        logger.debug(f"Semantic topic matching failed: {e}")

    return None


def fetch_import_context(
    source_topic_start_node_id: str,
    user_input: str,
    user_embedding: Optional[np.ndarray],
    token_budget: int,
    conn: Optional[sqlite3.Connection] = None,
    chroma_collection: Optional[Any] = None
) -> ImportContext:
    """
    Fetch relevant context from source topic for injection.

    Args:
        source_topic_start_node_id: Start node ID of the topic to import from
        user_input: Current user input (for anchor selection)
        user_embedding: Embedding of current user input
        token_budget: Maximum tokens for the imported context
        conn: Optional database connection
        chroma_collection: Optional Chroma collection

    Returns:
        ImportContext with formatted context block
    """
    debug: Dict[str, Any] = {
        'source_topic_start_node_id': source_topic_start_node_id,
        'token_budget': token_budget,
    }

    def _fetch(c: sqlite3.Connection) -> ImportContext:
        # Get topic info
        cursor = c.execute("""
            SELECT name, end_node_id FROM topics WHERE start_node_id = ?
        """, (source_topic_start_node_id,))
        row = cursor.fetchone()

        if not row:
            debug['error'] = 'topic_not_found'
            return ImportContext(context_block="", topic_name="", debug=debug)

        topic_name = row[0]
        debug['topic_name'] = topic_name

        parts = [f"[Imported from: {topic_name}]"]

        # Get summary from working set
        cursor = c.execute("""
            SELECT summary_md FROM topic_working_set WHERE topic_start_node_id = ?
        """, (source_topic_start_node_id,))
        summary_row = cursor.fetchone()

        if summary_row and summary_row[0] and summary_row[0].strip():
            summary = summary_row[0].strip()
            parts.append(summary)
            debug['summary_included'] = True
        else:
            debug['summary_included'] = False

        # Get top 2 anchors matching user query
        anchor_context = _get_import_anchors(
            source_topic_start_node_id,
            user_input,
            user_embedding,
            token_budget // 2,  # Reserve half budget for anchors
            c,
            chroma_collection
        )

        if anchor_context:
            parts.append("")
            parts.append(anchor_context)
            debug['anchors_included'] = True
        else:
            debug['anchors_included'] = False

        context_block = "\n".join(parts)

        # Truncate if over budget
        char_budget = token_budget * 4
        if len(context_block) > char_budget:
            context_block = context_block[:char_budget - 3] + "..."
            debug['truncated'] = True

        debug['context_length'] = len(context_block)
        debug['estimated_tokens'] = len(context_block) // 4

        return ImportContext(
            context_block=context_block,
            topic_name=topic_name,
            debug=debug
        )

    if conn is not None:
        return _fetch(conn)

    with get_connection() as c:
        return _fetch(c)


def _get_import_anchors(
    topic_start_node_id: str,
    user_input: str,
    user_embedding: Optional[np.ndarray],
    token_budget: int,
    conn: sqlite3.Connection,
    chroma_collection: Optional[Any] = None
) -> str:
    """Get relevant anchors from the source topic."""
    try:
        if chroma_collection is None:
            from episodic.rag_collections import get_multi_collection_rag, CollectionType
            rag = get_multi_collection_rag()
            chroma_collection = rag.get_collection(CollectionType.CONVERSATION)

        if chroma_collection is None:
            return ""

        # Query with topic filter
        results = chroma_collection.query(
            query_texts=[user_input],
            n_results=2,  # Top 2 anchors for imports
            where={"topic_start_node_id": topic_start_node_id}
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            return ""

        anchor_parts = []
        char_budget = token_budget * 4
        chars_used = 0

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i] if results.get('metadatas') else {}

            user_content = metadata.get('user_content', '')
            assistant_content = metadata.get('assistant_content', '')

            anchor_text = ""
            if user_content:
                anchor_text += f"User: {user_content}\n"
            if assistant_content:
                anchor_text += f"Assistant: {assistant_content}"

            if chars_used + len(anchor_text) > char_budget:
                break

            anchor_parts.append(anchor_text.strip())
            chars_used += len(anchor_text)

        return "\n\n".join(anchor_parts)

    except Exception as e:
        logger.debug(f"Import anchor retrieval failed: {e}")
        return ""
