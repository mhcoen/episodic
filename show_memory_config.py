#!/usr/bin/env python3
"""Show the config entries needed for memory system"""

print("""
To enable the memory system, add these lines to ~/.episodic/config.json:

{
  "enable_memory_rag": true,
  "enable_smart_memory": true,
  "memory_show_indicators": true,
  "memory_explicit_threshold": 0.7,
  "memory_implicit_threshold": 0.5,
  "memory_relevance_threshold": 0.7
}

Or use these commands in Episodic:

/set enable_memory_rag true
/set enable_smart_memory true
/set memory_show_indicators true

Then restart Episodic to see the memory indicators in action!
""")