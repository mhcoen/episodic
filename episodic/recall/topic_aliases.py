"""
Topic alias extraction for two-channel reactivation matching.

Extracts distinctive terms from topic metadata that can be used
for referential matching (e.g., "Back to that Python thing").
"""

import re
import sqlite3
from typing import List, Optional, Set

from episodic.db_connection import get_connection

# Common stopwords to filter out
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'about', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just',
    'don', 'now', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
    'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
    'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'up', 'down',
    'out', 'off', 'over', 'any', 'both', 'also', 'being', 'having',
    # Conversation-specific stopwords
    'topic', 'discussed', 'talked', 'mentioned', 'said', 'asked', 'told',
    'thing', 'stuff', 'something', 'anything', 'nothing', 'everything',
}

# Minimum term length
MIN_TERM_LENGTH = 3


def _normalize_term(term: str) -> Optional[str]:
    """Normalize a term for alias matching."""
    # Remove non-alphanumeric chars, lowercase
    normalized = re.sub(r'[^a-z0-9]', '', term.lower())

    # Skip if too short or is a stopword
    if len(normalized) < MIN_TERM_LENGTH:
        return None
    if normalized in STOPWORDS:
        return None

    return normalized


def _extract_terms(text: str) -> Set[str]:
    """Extract distinctive terms from text."""
    if not text:
        return set()

    terms = set()
    # Split on whitespace and punctuation
    words = re.split(r'[\s\-_/.,;:!?()\[\]{}]+', text)

    for word in words:
        normalized = _normalize_term(word)
        if normalized:
            terms.add(normalized)

    return terms


def extract_topic_aliases(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Set[str]:
    """
    Extract alias terms for a topic.

    Sources:
    1. Topic name tokens (normalized)
    2. Distinctive terms from structured summary (if exists)
    3. User content from topic exchanges

    Args:
        topic_start_node_id: The start_node_id that identifies the topic
        conn: Optional database connection

    Returns:
        Set of normalized alias terms
    """
    def _extract(c: sqlite3.Connection) -> Set[str]:
        aliases = set()

        # 1. Get topic name
        cursor = c.execute("""
            SELECT name FROM topics WHERE start_node_id = ?
        """, (topic_start_node_id,))
        row = cursor.fetchone()
        if row and row[0]:
            # Topic names are typically like "python-retry-patterns"
            name_terms = _extract_terms(row[0].replace('-', ' '))
            aliases.update(name_terms)

        # 2. Get structured summary if exists
        cursor = c.execute("""
            SELECT compressed_content FROM compressions_v2 c
            JOIN compression_nodes cn ON c.compressed_node_id = cn.compression_id
            WHERE cn.original_node_id IN (
                SELECT node_id FROM topic_nodes WHERE topic_start_node_id = ?
            )
            LIMIT 1
        """, (topic_start_node_id,))
        row = cursor.fetchone()
        if row and row[0]:
            # Extract terms from summary
            summary_terms = _extract_terms(row[0])
            aliases.update(summary_terms)

        # 3. Get distinctive terms from user exchanges (sample)
        cursor = c.execute("""
            SELECT n.content FROM nodes n
            JOIN topic_nodes tn ON n.id = tn.node_id
            WHERE tn.topic_start_node_id = ?
            AND n.role = 'user'
            ORDER BY n.rowid DESC
            LIMIT 5
        """, (topic_start_node_id,))

        for row in cursor.fetchall():
            if row[0]:
                exchange_terms = _extract_terms(row[0])
                aliases.update(exchange_terms)

        return aliases

    if conn is not None:
        return _extract(conn)

    with get_connection() as c:
        return _extract(c)


def compute_alias_score(
    query_text: str,
    topic_aliases: Set[str]
) -> int:
    """
    Compute alias match score between query and topic aliases.

    Args:
        query_text: The user's query text
        topic_aliases: Set of alias terms for the topic

    Returns:
        Number of distinct alias hits (query terms matching topic aliases)
    """
    if not topic_aliases:
        return 0

    query_terms = _extract_terms(query_text)

    # Count distinct matches
    hits = query_terms & topic_aliases
    return len(hits)


def get_topic_aliases_batch(
    topic_start_node_ids: List[str],
    conn: Optional[sqlite3.Connection] = None
) -> dict:
    """
    Get aliases for multiple topics efficiently.

    Args:
        topic_start_node_ids: List of topic start_node_ids
        conn: Optional database connection

    Returns:
        Dict mapping topic_start_node_id to set of aliases
    """
    def _batch(c: sqlite3.Connection) -> dict:
        result = {}
        for topic_id in topic_start_node_ids:
            result[topic_id] = extract_topic_aliases(topic_id, conn=c)
        return result

    if conn is not None:
        return _batch(conn)

    with get_connection() as c:
        return _batch(c)
