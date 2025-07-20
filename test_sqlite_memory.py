#!/usr/bin/env python3
"""Test the SQLite + ChromaDB memory integration"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.rag_memory_sqlite import memory_rag, enhance_with_memory_context

async def main():
    print("=== Testing SQLite + ChromaDB Memory Integration ===\n")
    
    # First, index recent conversations
    print("1. Indexing recent conversations from SQLite...")
    indexed = await memory_rag.index_recent_conversations(limit=50)
    print(f"   ✓ Indexed {indexed} new conversations\n")
    
    # Test referential detection
    print("2. Testing referential query detection:")
    test_queries = [
        # Clearly referential
        ("What was that command you mentioned?", True),
        ("Remember when we discussed Python?", True),
        ("You told me about virtual environments", True),
        ("Earlier you said something about decorators", True),
        
        # Not referential
        ("How do I create a list in Python?", False),
        ("Explain object-oriented programming", False),
        
        # Edge cases
        ("Which one?", True),  # Very short, likely referential
        ("Tell me more", True),  # Follow-up
    ]
    
    for query, expected in test_queries:
        is_ref, confidence = memory_rag.is_query_referential(query)
        status = "✓" if is_ref == expected else "✗"
        print(f"   {status} '{query}' -> {is_ref} (conf: {confidence:.2f})")
    
    print("\n3. Testing memory search:")
    search_queries = [
        "virtual environment",
        "save command",
        "drift detector",
        "What was that command?",
    ]
    
    for query in search_queries:
        print(f"\n   Query: '{query}'")
        memories = memory_rag.search_memories(query, limit=2)
        if memories:
            for i, mem in enumerate(memories, 1):
                print(f"   Result {i} (score: {mem['relevance_score']:.3f}):")
                print(f"      User: {mem['user_content'][:60]}...")
                print(f"      Assistant: {mem['assistant_content'][:60]}...")
        else:
            print("   No results found")
    
    print("\n4. Testing context enhancement:")
    referential_queries = [
        "What was that save command you mentioned?",
        "How did you say to activate the virtual environment?",
        "You mentioned something about drift detection",
    ]
    
    for query in referential_queries:
        print(f"\n   Query: '{query}'")
        context = await enhance_with_memory_context(query)
        if context:
            print("   Enhanced context:")
            print("   " + context.replace("\n", "\n   "))
        else:
            print("   No relevant context found")
    
    print("\n✅ SQLite + ChromaDB integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())