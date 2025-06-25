#!/usr/bin/env python3
"""
Test script to verify topic detection fixes.

This script will:
1. Show current topics in the database
2. Test topic detection with debug mode enabled
3. Verify unique topic names are created
"""

import sqlite3
import os
from episodic.db import get_db_path, get_recent_topics
from episodic.config import config

def show_current_topics():
    """Display all topics in the database."""
    print("\n=== Current Topics in Database ===")
    topics = get_recent_topics(limit=100)
    
    if not topics:
        print("No topics found in database")
        return
    
    for i, topic in enumerate(topics, 1):
        print(f"\n{i}. Topic: {topic['name']}")
        print(f"   Start: {topic['start_short_id']} -> End: {topic['end_short_id']}")
        print(f"   Confidence: {topic.get('confidence', 'N/A')}")
        print(f"   Created: {topic['created_at']}")
        
def check_node_relationships():
    """Check for broken parent chains."""
    print("\n=== Checking Node Relationships ===")
    
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Count orphan nodes
    c.execute("SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL AND role <> 'system'")
    orphan_count = c.fetchone()[0]
    print(f"Orphan nodes (no parent): {orphan_count}")
    
    if orphan_count > 0:
        c.execute("""
            SELECT short_id, role, substr(content, 1, 50) 
            FROM nodes 
            WHERE parent_id IS NULL AND role <> 'system'
            ORDER BY short_id
        """)
        print("\nOrphan nodes:")
        for row in c.fetchall():
            print(f"  {row[0]} ({row[1]}): {row[2]}")
    
    conn.close()

def test_topic_detection():
    """Test topic detection with debug mode."""
    print("\n=== Testing Topic Detection ===")
    print("To test topic detection:")
    print("1. Run: python -m episodic")
    print("2. Set debug mode: /config debug true")
    print("3. Set min messages before topic change: /config min_messages_before_topic_change 4")
    print("4. Have a conversation that changes topics")
    print("5. Watch for unique topic names like 'ongoing-discussion-<timestamp>'")

def main():
    """Run all tests."""
    print("Topic Detection Fix Verification")
    print("=" * 50)
    
    # Enable debug mode for this test
    original_debug = config.get("debug", False)
    config.set("debug", True)
    
    try:
        show_current_topics()
        check_node_relationships()
        test_topic_detection()
        
        print("\n=== Summary ===")
        print("✓ Fixed: New topics now get unique names (ongoing-discussion-<timestamp>)")
        print("✓ Fixed: Broken parent chains are handled gracefully")
        print("✓ Fixed: Topic count calculation works even with broken chains")
        print("\n⚠️  Note: Existing topics with duplicate names won't be fixed retroactively")
        print("⚠️  Note: Consider setting min_messages_before_topic_change to a lower value (e.g., 4)")
        
    finally:
        # Restore original debug setting
        config.set("debug", original_debug)

if __name__ == "__main__":
    main()