#!/usr/bin/env python3
"""
Search your conversation history.

Usage:
    ./search.py coffee
    ./search.py "python error"
    ./search.py --user "how do I"      # Only your questions
    ./search.py --topic cooking pasta  # Search within a topic
"""
import sys
import sqlite3
sys.path.insert(0, '/Users/mhcoen/proj/episodic')

from episodic.db_connection import get_db_path
from episodic.retrieval import retrieve
from episodic.retrieval.display import get_exchange_for_display
from episodic.retrieval.segment import get_all_topics, get_cached_segment_nodes

class NoChroma:
    def query(self, **kw):
        return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

CONFIG = {
    "semantic_weight": 0.0,
    "lexical_weight": 1.0,
    "over_fetch_multiplier": 3,
    "segment_filter_in_clause_max": 100,
    "sqlite_max_variable_number": 999,
}

def search(query, speaker=None, topic_name=None, limit=5):
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    segment_scope = None
    if topic_name:
        topics = get_all_topics(conn)
        for t in topics:
            if topic_name.lower() in t['name'].lower():
                nodes, _ = get_cached_segment_nodes(conn, t['id'])
                segment_scope = nodes
                print(f"🔍 Searching in topic: {t['name']}\n")
                break
        else:
            print(f"⚠️  Topic '{topic_name}' not found")
            return
    
    results = retrieve(
        conn=conn,
        chroma=NoChroma(),
        target=query,
        segment_scope=segment_scope,
        temporal=None,
        speaker=speaker,
        mode="browse",
        max_results=limit,
        config=CONFIG
    )
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results for: {query}\n")
    print("=" * 70)
    
    for i, r in enumerate(results, 1):
        ex = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        print(f"\n[{i}] You:")
        print(f"    {ex['user_content'][:200]}")
        if ex.get('assistant_content'):
            print(f"\n    Assistant:")
            print(f"    {ex['assistant_content'][:200]}")
        print("-" * 70)
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    args = sys.argv[1:]
    speaker = None
    topic = None
    query_parts = []
    
    i = 0
    while i < len(args):
        if args[i] == "--user":
            speaker = "user"
        elif args[i] == "--assistant":
            speaker = "assistant"
        elif args[i] == "--topic" and i + 1 < len(args):
            i += 1
            topic = args[i]
        else:
            query_parts.append(args[i])
        i += 1
    
    query = " ".join(query_parts)
    search(query, speaker=speaker, topic_name=topic)
