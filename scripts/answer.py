#!/usr/bin/env python3
"""
Ask a question about your conversation history.

Returns relevant excerpts or "I don't have that" if nothing found.

Usage:
    ./answer.py "what did we discuss about coffee?"
    ./answer.py "when did I ask about Python?"
"""
import sys
import sqlite3
sys.path.insert(0, '/Users/mhcoen/proj/episodic')

from episodic.db_connection import get_db_path
from episodic.retrieval import retrieve
from episodic.retrieval.display import get_exchange_for_display
from episodic.retrieval.modes import format_answer_response

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

def answer(question, limit=5):
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    print(f"❓ {question}\n")
    
    results = retrieve(
        conn=conn,
        chroma=NoChroma(),
        target=question,
        segment_scope=None,
        temporal=None,
        speaker=None,
        mode="answer",
        max_results=limit,
        config=CONFIG
    )
    
    # Check for empty
    fixed = format_answer_response(results)
    if fixed:
        print(f"💬 {fixed}")
        conn.close()
        return
    
    print(f"📚 Found {len(results)} relevant exchanges:\n")
    print("=" * 70)
    
    for i, r in enumerate(results, 1):
        ex = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))
        
        cursor = conn.cursor()
        cursor.execute("SELECT created_at FROM nodes WHERE id = ?", (r['exchange_id'],))
        row = cursor.fetchone()
        ts = row['created_at'][:10] if row else "?"
        
        print(f"\n[{i}] {ts}")
        print(f"    You: {ex['user_content']}")
        if ex.get('assistant_content'):
            assistant_text = ex['assistant_content']
            if len(assistant_text) > 300:
                assistant_text = assistant_text[:300] + "..."
            print(f"\n    Assistant: {assistant_text}")
        print("-" * 70)
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    answer(question)
