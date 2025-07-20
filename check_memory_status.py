#!/usr/bin/env python3
"""Check memory system status and configuration"""

from episodic.config import config

print("=== Memory System Status ===\n")

# Check basic memory
memory_enabled = config.get("enable_memory_rag", False)
print(f"Memory RAG: {'✓ Enabled' if memory_enabled else '✗ Disabled'}")

# Check smart memory
smart_enabled = config.get("enable_smart_memory", False)
print(f"Smart Memory: {'✓ Enabled' if smart_enabled else '✗ Disabled'}")

# Check indicators
indicators = config.get("memory_show_indicators", True)
print(f"Visual Indicators: {'✓ Shown' if indicators else '✗ Hidden'}")

# Check thresholds
print("\nThresholds:")
print(f"  Explicit: {config.get('memory_explicit_threshold', 0.7):.0%}")
print(f"  Implicit: {config.get('memory_implicit_threshold', 0.5):.0%}")
print(f"  Relevance: {config.get('memory_relevance_threshold', 0.7):.0%}")

if not memory_enabled:
    print("\n⚠️  Memory system is disabled!")
    print("Run: python enable_memory_rag.py")
elif not smart_enabled:
    print("\n⚠️  Smart detection is disabled!")
    print("Run: python enable_smart_memory.py")
else:
    print("\n✅ All systems active!")