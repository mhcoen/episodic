#!/usr/bin/env python3
"""
Run this to see the retrieval system in action.

Usage:
    cd ~/proj/episodic
    python -m episodic.retrieval.demo
"""
import sqlite3
import sys

def main():
    from episodic.db_connection import get_db_path
    from episodic.retrieval.migration import migrate_fts5
    from episodic.retrieval import retrieve
    from episodic.retrieval.display import get_exchange_for_display
    from episodic.retrieval.segment import get_all_topics, get_cached_segment_nodes
    from episodic.retrieval.modes import format_answer_response
    
    print("=" * 60)
    print("EPISODIC RETRIEVAL DEMO")
    print("=" * 60)
    
    db_path = get_db_path()
    print(f"\nDatabase: {db_path}")
    
    # Step 1: Migrate FTS5
    print("\n[1] Migrating FTS5...")
    migration_conn = sqlite3.connect(db_path, isolation_level=None)
    migration_conn.row_factory = sqlite3.Row
    migrate_fts5(migration_conn)
    migration_conn.close()
    print("    ✓ FTS5 ready")
    
    # Step 2: Connect
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Check stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE role='user'")
    user_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM topics")
    topic_count = cursor.fetchone()[0]
    print(f"\n[2] Database stats:")
    print(f"    {user_count} exchanges (user messages)")
    print(f"    {topic_count} topics")
    
    if user_count == 0:
        print("\n⚠️  No conversations yet. Have some chats first!")
        conn.close()
        return
    
    # Step 3: Show topics
    print(f"\n[3] Topics:")
    topics = get_all_topics(conn)
    for t in topics[-5:]:  # Last 5
        status = "(ongoing)" if t['end_node_id'] is None else ""
        nodes, _ = get_cached_segment_nodes(conn, t['id'])
        print(f"    [{t['id']}] {t['name']} - {len(nodes)} nodes {status}")
    
    # Stub Chroma (lexical only for demo)
    class NoChroma:
        def query(self, **kw):
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    config = {
        "semantic_weight": 0.0,
        "lexical_weight": 1.0,
        "over_fetch_multiplier": 3,
        "segment_filter_in_clause_max": 100,
        "sqlite_max_variable_number": 999,
    }
    
    # Step 4: Browse recent (empty target)
    print(f"\n[4] Browse mode (recent exchanges):")
    results = retrieve(
        conn=conn,
        chroma=NoChroma(),
        target="",  # Empty = recent
        segment_scope=None,
        temporal=None,
        speaker=None,
        mode="browse",
        max_results=3,
        config=config
    )
    for r in results:
        ex = get_exchange_for_display(conn, r['exchange_id'], None)
        user_preview = (ex['user_content'] or "")[:60].replace('\n', ' ')
        print(f"    → {user_preview}...")
    
    # Step 5: Search
    print(f"\n[5] Search mode:")
    test_queries = ["python", "error", "help", "how"]
    
    for query in test_queries:
        results = retrieve(
            conn=conn,
            chroma=NoChroma(),
            target=query,
            segment_scope=None,
            temporal=None,
            speaker=None,
            mode="answer",
            max_results=3,
            config=config
        )
        if results:
            print(f"\n    Query: '{query}' → {len(results)} results")
            for r in results[:2]:
                ex = get_exchange_for_display(conn, r['exchange_id'], None)
                preview = (ex['user_content'] or "")[:50].replace('\n', ' ')
                print(f"      • {preview}...")
            break
    else:
        print(f"    (No matches for test queries)")
    
    # Step 6: Answer mode with no results
    print(f"\n[6] Answer mode (no results):")
    results = retrieve(
        conn=conn,
        chroma=NoChroma(),
        target="xyzzy_nonexistent_query_12345",
        segment_scope=None,
        temporal=None,
        speaker=None,
        mode="answer",
        max_results=5,
        config=config
    )
    response = format_answer_response(results)
    print(f"    → \"{response}\"")
    
    # Step 7: Scoped search (if topics exist)
    if topics:
        recent_topic = topics[-1]
        nodes, _ = get_cached_segment_nodes(conn, recent_topic['id'])
        
        print(f"\n[7] Scoped search (topic: {recent_topic['name']}):")
        results = retrieve(
            conn=conn,
            chroma=NoChroma(),
            target="",
            segment_scope=nodes if nodes else None,
            temporal=None,
            speaker=None,
            mode="browse",
            max_results=3,
            config=config
        )
        print(f"    → {len(results)} exchanges in topic")
    
    conn.close()
    print("\n" + "=" * 60)
    print("Demo complete. See episodic/retrieval/examples.py for API usage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
