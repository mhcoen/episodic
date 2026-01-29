#!/usr/bin/env python3
"""
Episodic Retrieval System - Usage Examples

Run from episodic root:
    python -m episodic.retrieval.example
"""
import sqlite3
from datetime import datetime, timezone

# =============================================================================
# Setup: Get a connection and migrate FTS5
# =============================================================================

def setup_retrieval():
    """One-time setup: migrate FTS5 for lexical search."""
    from episodic.db_connection import get_db_path
    
    # Migration requires isolation_level=None
    conn = sqlite3.connect(get_db_path(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    
    from episodic.retrieval.migration import migrate_fts5
    migrate_fts5(conn)
    conn.close()
    print("✓ FTS5 migration complete")


# =============================================================================
# Basic Usage: Search your conversations
# =============================================================================

def search_conversations(query: str, max_results: int = 5):
    """
    Search conversation history.
    
    Returns exchanges (user + assistant pairs) matching the query.
    """
    from episodic.db_connection import get_db_path
    from episodic.retrieval import retrieve
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    # For now, no Chroma - lexical only
    # To add semantic: pass your Chroma collection
    class NoOpChroma:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    results = retrieve(
        conn=conn,
        chroma=NoOpChroma(),
        target=query,
        segment_scope=None,      # None = search all segments
        temporal=None,           # None = no time filter
        speaker=None,            # None = search both user and assistant
        mode="browse",           # "browse", "answer", or "summarize"
        max_results=max_results,
        config={
            "semantic_weight": 0.0,  # Lexical only
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999
        }
    )
    
    conn.close()
    return results


def search_with_display(query: str):
    """Search and format results for display."""
    from episodic.db_connection import get_db_path
    from episodic.retrieval import retrieve
    from episodic.retrieval.display import get_exchange_for_display
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    class NoOpChroma:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    results = retrieve(
        conn=conn,
        chroma=NoOpChroma(),
        target=query,
        segment_scope=None,
        temporal=None,
        speaker=None,
        mode="browse",
        max_results=5,
        config={
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999
        }
    )
    
    print(f"\n🔍 Results for: {query}\n")
    
    for i, r in enumerate(results, 1):
        exchange = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        print(f"─── Result {i} ───")
        print(f"You: {exchange['user_content'][:100]}...")
        if exchange['assistant_content']:
            print(f"AI:  {exchange['assistant_content'][:100]}...")
        print()
    
    conn.close()


# =============================================================================
# Scoped Search: Filter by segment, time, or speaker
# =============================================================================

def search_in_topic(query: str, topic_name: str):
    """Search within a specific topic/segment."""
    from episodic.db_connection import get_db_path
    from episodic.retrieval import retrieve
    from episodic.retrieval.segment import get_all_topics, get_cached_segment_nodes
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    # Find the topic
    topics = get_all_topics(conn)
    target_topic = None
    for t in topics:
        if topic_name.lower() in t['name'].lower():
            target_topic = t
            break
    
    if not target_topic:
        print(f"Topic '{topic_name}' not found")
        conn.close()
        return []
    
    # Get segment nodes
    nodes_list, _ = get_cached_segment_nodes(conn, target_topic['id'])
    
    class NoOpChroma:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    results = retrieve(
        conn=conn,
        chroma=NoOpChroma(),
        target=query,
        segment_scope=nodes_list,  # Filter to this segment
        temporal=None,
        speaker=None,
        mode="browse",
        max_results=5,
        config={
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999
        }
    )
    
    print(f"\n🔍 Results in topic '{target_topic['name']}': {len(results)}")
    conn.close()
    return results


def search_recent(query: str, hours: int = 24):
    """Search only recent conversations."""
    from episodic.db_connection import get_db_path
    from episodic.retrieval import retrieve
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    # Calculate time window
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Format as canonical ISO8601
    start_str = start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '000Z'
    end_str = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '000Z'
    
    class NoOpChroma:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    results = retrieve(
        conn=conn,
        chroma=NoOpChroma(),
        target=query,
        segment_scope=None,
        temporal=(start_str, end_str),  # Time filter
        speaker=None,
        mode="browse",
        max_results=5,
        config={
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999
        }
    )
    
    print(f"\n🔍 Results from today: {len(results)}")
    conn.close()
    return results


def search_my_questions(query: str):
    """Search only user messages."""
    from episodic.db_connection import get_db_path
    from episodic.retrieval import retrieve
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    class NoOpChroma:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    results = retrieve(
        conn=conn,
        chroma=NoOpChroma(),
        target=query,
        segment_scope=None,
        temporal=None,
        speaker="user",  # Only user messages
        mode="browse",
        max_results=5,
        config={
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999
        }
    )
    
    print(f"\n🔍 Your questions matching '{query}': {len(results)}")
    conn.close()
    return results


# =============================================================================
# Answer Mode: Get grounded response
# =============================================================================

def answer_from_history(question: str):
    """
    Answer a question using only conversation history.
    
    Returns fixed string if nothing found (no hallucination).
    """
    from episodic.db_connection import get_db_path
    from episodic.retrieval import retrieve
    from episodic.retrieval.modes import format_answer_response
    from episodic.retrieval.display import get_exchange_for_display
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    class NoOpChroma:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    results = retrieve(
        conn=conn,
        chroma=NoOpChroma(),
        target=question,
        segment_scope=None,
        temporal=None,
        speaker=None,
        mode="answer",
        max_results=5,
        config={
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999
        }
    )
    
    # Check for empty
    response = format_answer_response(results)
    if response:
        print(f"\n💬 {response}")
        conn.close()
        return response
    
    # Build context for LLM
    print(f"\n📚 Found {len(results)} relevant exchanges:")
    excerpts = []
    for r in results:
        exchange = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        excerpts.append(f"User: {exchange['user_content']}\nAssistant: {exchange['assistant_content']}")
        print(f"  - {exchange['user_content'][:60]}...")
    
    conn.close()
    
    # Return excerpts for LLM synthesis
    return excerpts


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
Episodic Retrieval Examples

Usage:
    python -m episodic.retrieval.example setup           # Migrate FTS5
    python -m episodic.retrieval.example search "query"  # Basic search
    python -m episodic.retrieval.example answer "question"  # Answer mode
    python -m episodic.retrieval.example topic "query" "topic_name"  # Scoped search
""")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "setup":
        setup_retrieval()
    elif cmd == "search" and len(sys.argv) > 2:
        search_with_display(sys.argv[2])
    elif cmd == "answer" and len(sys.argv) > 2:
        answer_from_history(sys.argv[2])
    elif cmd == "topic" and len(sys.argv) > 3:
        search_in_topic(sys.argv[2], sys.argv[3])
    else:
        print("Unknown command or missing arguments")
