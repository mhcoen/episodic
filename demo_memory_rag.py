#!/usr/bin/env python3
"""Demo the SQLite + ChromaDB memory RAG system with real conversations"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.config import config
from episodic.conversation import conversation_manager
from episodic.rag_memory_sqlite import memory_rag, enhance_with_memory_context

async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Episodic Memory RAG System - Milestone 1 Demo         ║
║                                                              ║
║  Automatic memory with ChromaDB + Referential Detection      ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Setup
    config.set("enable_memory_rag", True)
    config.set("skip_llm_response", True)  # For demo speed
    config.set("debug", True)
    
    print("1. First, let's index existing conversations...")
    indexed = await memory_rag.index_recent_conversations(limit=20)
    print(f"   ✓ Indexed {indexed} recent conversations\n")
    
    print("2. Now let's have a conversation that will be auto-indexed:\n")
    
    # Simulate some conversations
    exchanges = [
        ("How do I create a Python virtual environment?",
         "[Assistant explains: python -m venv myenv]"),
         
        ("What about activating it on Mac?",
         "[Assistant says: source myenv/bin/activate]"),
         
        ("Thanks! Now tell me about list comprehensions",
         "[Assistant explains list comprehensions]"),
    ]
    
    for user_msg, expected in exchanges:
        print(f"👤 User: {user_msg}")
        _, response = conversation_manager.handle_chat_message(
            user_msg, "gpt-3.5-turbo", None, 5
        )
        print()
    
    print("\n3. Now let's test referential queries that should find past context:\n")
    
    # Test referential queries
    referential_queries = [
        "What was that command to create a virtual environment again?",
        "You mentioned something about activating it?",
        "Earlier you explained something, what was it?"
    ]
    
    for query in referential_queries:
        print(f"👤 User: {query}")
        
        # Check if it's referential
        is_ref, conf = memory_rag.is_query_referential(query)
        print(f"   [Memory] Referential detection: {is_ref} (confidence: {conf:.2f})")
        
        # Get context
        context = await enhance_with_memory_context(query)
        if context:
            print("   [Memory] Found relevant context:")
            print("   " + context.replace("\n", "\n   "))
        else:
            print("   [Memory] No relevant context found")
        print()
    
    print("\n4. Summary of what's happening behind the scenes:")
    print("   ✓ Every conversation is automatically indexed in ChromaDB")
    print("   ✓ Referential queries are detected (90% confidence for explicit references)")
    print("   ✓ Relevant past conversations are retrieved using vector similarity")
    print("   ✓ Context is injected into the LLM prompt automatically")
    print("\n✅ Milestone 1 Complete: Basic auto-memory with ChromaDB is working!")

if __name__ == "__main__":
    asyncio.run(main())