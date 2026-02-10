"""
Episodic Retrieval System - Usage Examples

This demonstrates how to use the new retrieval system for
answering, browsing, and summarizing conversation history.
"""
import sqlite3
from datetime import datetime, timezone, timedelta

# =============================================================================
# Setup
# =============================================================================

from episodic.retrieval import migrate_fts5, retrieve
from episodic.retrieval.modes import format_answer_response, format_browse_response, format_summarize_response
from episodic.retrieval.display import get_exchange_for_display
from episodic.retrieval.segment import get_cached_segment_nodes, get_all_topics

# Configuration
RETRIEVAL_CONFIG = {
    "semantic_weight": 0.6,
    "lexical_weight": 0.4,
    "over_fetch_multiplier": 3,
    "segment_filter_in_clause_max": 100,
    "sqlite_max_variable_number": 999,
}


def get_retrieval_connection(db_path: str) -> sqlite3.Connection:
    """Get a connection configured for retrieval."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_fts_migrated(db_path: str):
    """Run FTS5 migration if needed (idempotent)."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    migrate_fts5(conn)
    conn.close()


# =============================================================================
# Example 1: Answer a question about past conversations
# =============================================================================

def answer_question(conn, chroma, question: str) -> str:
    """
    Answer a question using conversation history.
    
    Returns either retrieved excerpts for LLM synthesis,
    or a fixed "not found" message.
    """
    results = retrieve(
        conn=conn,
        chroma=chroma,
        target=question,
        segment_scope=None,  # Search all segments
        temporal=None,       # No time restriction
        speaker=None,        # Both user and assistant
        mode="answer",
        max_results=5,
        config=RETRIEVAL_CONFIG
    )
    
    # Check for empty
    fixed_response = format_answer_response(results)
    if fixed_response:
        return fixed_response  # "I don't have that in our conversation history."
    
    # Format excerpts for LLM
    excerpts = []
    for r in results:
        exchange = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        excerpts.append(f"User: {exchange['user_content']}\nAssistant: {exchange['assistant_content']}")
    
    return "\n\n---\n\n".join(excerpts)


# =============================================================================
# Example 2: Browse conversations by topic
# =============================================================================

def browse_topic(conn, chroma, topic_query: str) -> dict:
    """
    Browse conversations in a specific topic.
    
    First resolves the topic name, then retrieves relevant exchanges.
    """
    # Resolve topic to segment nodes
    from episodic.retrieval.segment import get_all_topics, get_cached_segment_nodes
    
    topics = get_all_topics(conn)
    
    # Simple topic matching (production would use segment resolver)
    matched_topic = None
    for t in topics:
        if topic_query.lower() in t['name'].lower():
            matched_topic = t
            break
    
    if not matched_topic:
        return {"error": f"No topic matching '{topic_query}'"}
    
    # Get segment nodes
    nodes_list, nodes_set = get_cached_segment_nodes(conn, matched_topic['id'])
    
    if not nodes_list:
        return {"error": "Topic has no nodes"}
    
    # Retrieve within segment
    results = retrieve(
        conn=conn,
        chroma=chroma,
        target="",  # Empty = browse recent
        segment_scope=nodes_list,
        temporal=None,
        speaker=None,
        mode="browse",
        max_results=10,
        config=RETRIEVAL_CONFIG
    )
    
    return format_browse_response(conn, results)


# =============================================================================
# Example 3: Search with time filter
# =============================================================================

def search_last_week(conn, chroma, query: str) -> list:
    """Search conversations from the last 7 days."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    
    # Format as canonical ISO8601
    temporal = (
        week_ago.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )
    
    results = retrieve(
        conn=conn,
        chroma=chroma,
        target=query,
        segment_scope=None,
        temporal=temporal,
        speaker=None,
        mode="answer",
        max_results=10,
        config=RETRIEVAL_CONFIG
    )
    
    return results


# =============================================================================
# Example 4: Search only user messages
# =============================================================================

def search_user_questions(conn, chroma, query: str) -> list:
    """
    Search only user messages (not assistant responses).
    
    Note: This disables semantic search (speaker scope incompatible
    with exchange-level embeddings).
    """
    results = retrieve(
        conn=conn,
        chroma=chroma,
        target=query,
        segment_scope=None,
        temporal=None,
        speaker="user",  # Only user messages
        mode="browse",
        max_results=10,
        config=RETRIEVAL_CONFIG
    )
    
    return results


# =============================================================================
# Example 5: Summarize a topic
# =============================================================================

def summarize_topic(conn, chroma, topic_name: str) -> str:
    """
    Get excerpts for summarizing a topic.
    
    Returns excerpts for LLM summarization or fixed message if empty.
    """
    topics = get_all_topics(conn)
    matched = next((t for t in topics if topic_name.lower() in t['name'].lower()), None)
    
    if not matched:
        return "No conversations found to summarize."
    
    nodes_list, _ = get_cached_segment_nodes(conn, matched['id'])
    
    results = retrieve(
        conn=conn,
        chroma=chroma,
        target="",  # Get all from segment
        segment_scope=nodes_list,
        temporal=None,
        speaker=None,
        mode="summarize",
        max_results=20,
        config=RETRIEVAL_CONFIG
    )
    
    fixed = format_summarize_response(results)
    if fixed:
        return fixed
    
    # Return excerpts for LLM summarization
    excerpts = []
    for r in results:
        exchange = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        excerpts.append(f"User: {exchange['user_content']}\nAssistant: {exchange['assistant_content']}")
    
    return "\n\n".join(excerpts)


# =============================================================================
# Example 6: Integration with existing Episodic CLI
# =============================================================================

def handle_recall_command(query: str, db_path: str, chroma_collection) -> str:
    """
    Handle /recall command in CLI.
    
    Usage:
        /recall what did we discuss about coffee?
        /recall --topic legal when did we talk about contracts?
        /recall --last-week python errors
    """
    ensure_fts_migrated(db_path)
    conn = get_retrieval_connection(db_path)
    
    try:
        # Simple implementation - production would parse flags
        results = retrieve(
            conn=conn,
            chroma=chroma_collection,
            target=query,
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="answer",
            max_results=5,
            config=RETRIEVAL_CONFIG
        )
        
        fixed = format_answer_response(results)
        if fixed:
            return fixed
        
        # Format for display
        output = []
        for r in results:
            exchange = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
            output.append(f"📝 {exchange['user_content'][:100]}...")
            if exchange['assistant_content']:
                output.append(f"   → {exchange['assistant_content'][:100]}...")
        
        return "\n".join(output)
    
    finally:
        conn.close()


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Retrieval system loaded. See examples above for usage.")
    print("\nKey functions:")
    print("  - answer_question(conn, chroma, question)")
    print("  - browse_topic(conn, chroma, topic_query)")
    print("  - search_last_week(conn, chroma, query)")
    print("  - search_user_questions(conn, chroma, query)")
    print("  - summarize_topic(conn, chroma, topic_name)")
