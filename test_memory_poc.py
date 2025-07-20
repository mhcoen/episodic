#!/usr/bin/env python3
"""Test the memory POC integration"""

from episodic.conversation import conversation_manager
from episodic.config import config
from episodic.rag_memory import memory_system
import typer

# Enable debug and memory POC
config.set("debug", True)
config.set("enable_memory_poc", True)
config.set("skip_llm_response", True)  # Skip actual LLM calls for testing

print("=== Testing Memory POC Integration ===\n")

# Simulate some conversations
test_messages = [
    "What's the best way to learn Python?",
    "Tell me about data structures",
    "How do I use dictionaries?",
    "What about lists and tuples?",
    "Can you explain list comprehensions?",
    "What was that about dictionaries again?"
]

# Process messages
for i, msg in enumerate(test_messages):
    print(f"\n--- Message {i+1} ---")
    print(f"User: {msg}")
    
    # This will trigger memory indexing
    conversation_manager.handle_chat_message(msg, "gpt-3.5-turbo", None, 5)
    
    # Show memory stats
    stats = memory_system.get_stats()
    print(f"\nMemory Stats: {stats['total_memories']} memories stored")

# Test retrieval
print("\n\n=== Testing Memory Retrieval ===\n")

test_queries = [
    "dictionaries",
    "Python learning",
    "data structures"
]

for query in test_queries:
    print(f"\nSearching for: '{query}'")
    context = memory_system.search_and_format(query)
    if context:
        print(context)
    else:
        print("No memories found")

print("\n\n=== Final Memory Stats ===")
final_stats = memory_system.get_stats()
for key, value in final_stats.items():
    print(f"{key}: {value}")