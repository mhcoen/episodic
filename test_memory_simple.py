#!/usr/bin/env python3
"""Simple memory POC test - minimal output"""

from episodic.conversation import conversation_manager
from episodic.config import config
from episodic.rag_memory import memory_system

# Setup
config.set("debug", False)  # Turn off debug for cleaner output
config.set("enable_memory_poc", True)
config.set("skip_llm_response", True)

print("=== Memory System POC Demo ===\n")

# Clear previous memories for clean demo
memory_system.memories = []
memory_system.message_count = 0

# Simulate conversations
exchanges = [
    ("What's the capital of France?", "[Assistant would say: Paris]"),
    ("Tell me about Python lists", "[Assistant would explain lists]"),
    ("How do I install numpy?", "[Assistant would say: pip install numpy]"),
    ("What are Python decorators?", "[Assistant would explain decorators]"),
    ("Can you explain generators?", "[Assistant would explain generators]"),
    ("What was that install command?", "[Assistant would recall: pip install numpy]"),
]

# Process exchanges (simulating the integration)
for i, (user_msg, _) in enumerate(exchanges):
    print(f"User: {user_msg}")
    # This simulates what happens in handle_chat_message
    conversation_manager.handle_chat_message(user_msg, "gpt-3.5-turbo", None, 5)
    print()

print("\n=== Memory Search Demo ===\n")

# Test searches
searches = [
    "install numpy",
    "Python",
    "France capital"
]

for query in searches:
    print(f"Query: '{query}'")
    results = memory_system.search_and_format(query)
    print(results if results else "No results found")
    print()

# Final stats
print("=== Memory Statistics ===")
stats = memory_system.get_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"Memory size: {stats['memory_size_kb']:.1f} KB")
print(f"Oldest: {stats['oldest_memory']}")
print(f"Newest: {stats['newest_memory']}")