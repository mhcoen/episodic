#!/usr/bin/env python3
"""
Episodic Memory System - Proof of Concept Demo

This demonstrates automatic conversation memory without explicit save/load commands.
"""

import time
from episodic.rag_memory import memory_system

print("""
╔══════════════════════════════════════════════════════════════╗
║           Episodic Memory System - POC Demo                  ║
║                                                              ║
║  No more /save commands! Conversations remember themselves.  ║
╚══════════════════════════════════════════════════════════════╝
""")

# Clear previous memories for clean demo
memory_system.memories = []
memory_system.message_count = 0
memory_system._save_memories()

print("Let's simulate a conversation that spans different topics...\n")

# Simulate a realistic conversation
conversations = [
    ("I'm learning Python. What IDE do you recommend?", 
     "I recommend VS Code or PyCharm. VS Code is free and lightweight with great Python extensions."),
    
    ("Can you explain list comprehensions?",
     "List comprehensions are a concise way to create lists: [x*2 for x in range(5)] creates [0, 2, 4, 6, 8]"),
    
    ("What about virtual environments?",
     "Use venv or virtualenv to isolate project dependencies: python -m venv myenv"),
    
    ("How do I activate it?",
     "On Mac/Linux: source myenv/bin/activate. On Windows: myenv\\Scripts\\activate"),
    
    ("Thanks! By the way, what's a good restaurant in San Francisco?",
     "For great dining in SF, try The Slanted Door for Vietnamese or Gary Danko for fine dining."),
    
    ("What was that command to create a virtual environment again?",
     "The command was: python -m venv myenv")
]

# Process conversations
import asyncio
for i, (user, assistant) in enumerate(conversations, 1):
    print(f"Exchange {i}:")
    print(f"👤 User: {user}")
    print(f"🤖 Assistant: {assistant[:60]}...")
    
    # Index the conversation
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memory_system.on_message(user, assistant))
    loop.close()
    
    time.sleep(0.5)  # Pause for effect
    print()

print("\n" + "="*60 + "\n")
print("Now let's test memory retrieval WITHOUT any explicit save/load...\n")

# Test queries that reference past conversations
test_queries = [
    ("virtual environment", "Should recall the venv discussion"),
    ("restaurant SF", "Should recall the SF restaurant recommendation"),
    ("VS Code", "Should recall the IDE recommendation"),
]

for query, description in test_queries:
    print(f"🔍 Query: '{query}' ({description})")
    results = memory_system.search_and_format(query)
    if results:
        print(results)
    else:
        print("   No memories found")
    print()

# Show final stats
print("\n" + "="*60)
print("📊 Memory System Statistics:")
stats = memory_system.get_stats()
print(f"   Total conversations indexed: {stats['total_memories']}")
print(f"   Memory storage used: {stats['memory_size_kb']:.1f} KB")
print(f"   Time span: {stats['oldest_memory'][:10]} to {stats['newest_memory'][:10]}")

print("""
✨ Key Benefits:
   • No manual /save commands needed
   • Automatic indexing of all conversations
   • Natural language search across all memories
   • Works seamlessly in the background
   
🚀 Next steps would include:
   • ChromaDB integration for better search
   • Automatic context injection when relevant
   • Memory consolidation over time
""")