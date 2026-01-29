#!/usr/bin/env python3
"""
Browse recent conversations.

Usage:
    ./recent.py           # Last 5 exchanges
    ./recent.py 10        # Last 10 exchanges
    ./recent.py --today   # Today's conversations only
"""
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
sys.path.insert(0, '/Users/mhcoen/proj/episodic')

from episodic.db_connection import get_db_path
from episodic.retrieval import retrieve
from episodic.retrieval.display import get_exchange_for_display

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

def browse_recent(limit=5, today_only=False):
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    temporal = None
    if today_only:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        temporal = (
            start.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            now.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        )
        print("📅 Today's conversations:\n")
    else:
        print(f"📜 Last {limit} conversations:\n")
    
    results = retrieve(
        conn=conn,
        chroma=NoChroma(),
        target="",  # Empty = browse recent
        segment_scope=None,
        temporal=temporal,
        speaker=None,
        mode="browse",
        max_results=limit,
        config=CONFIG
    )
    
    if not results:
        print("No conversations found.")
        return
    
    print("=" * 70)
    
    for i, r in enumerate(results, 1):
        ex = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        
        # Get timestamp
        cursor = conn.cursor()
        cursor.execute("SELECT created_at FROM nodes WHERE id = ?", (r['exchange_id'],))
        row = cursor.fetchone()
        ts = row['created_at'][:19] if row else "?"
        
        print(f"\n[{i}] {ts}")
        print(f"    You: {ex['user_content'][:150]}...")
        if ex.get('assistant_content'):
            print(f"    AI:  {ex['assistant_content'][:150]}...")
        print("-" * 70)
    
    conn.close()

if __name__ == "__main__":
    limit = 5
    today_only = False
    
    for arg in sys.argv[1:]:
        if arg == "--today":
            today_only = True
        elif arg.isdigit():
            limit = int(arg)
    
    browse_recent(limit, today_only)
