#!/usr/bin/env python3
"""Enable the SQLite + ChromaDB memory RAG system"""

import asyncio
from episodic.config import config
from episodic.rag_memory_sqlite import memory_rag

async def main():
    print("=== Enabling Memory RAG System ===\n")
    
    # Enable the memory RAG system
    config.set("enable_memory_rag", True)
    print("✓ Memory RAG enabled in config")
    
    # Index existing conversations
    print("\nIndexing existing conversations...")
    indexed = await memory_rag.index_recent_conversations(limit=100)
    print(f"✓ Indexed {indexed} conversations")
    
    # Show what's enabled
    print("\n✅ Memory RAG system is now active!")
    print("\nFeatures enabled:")
    print("- Automatic memory indexing for new conversations")
    print("- Referential query detection")
    print("- Context injection for relevant past conversations")
    print("\nExamples that will trigger memory retrieval:")
    print('- "What was that command you mentioned?"')
    print('- "Remember when we discussed Python?"')
    print('- "You told me about..."')
    print('- "Earlier you said..."')
    
    print("\nTo disable: config.set('enable_memory_rag', False)")

if __name__ == "__main__":
    asyncio.run(main())