#!/usr/bin/env python3
"""Trace when topics should be extended."""

# From the test output, let's simulate the flow
messages = [
    ("02", "user", "Tell me about the Mars rovers"),
    ("03", "assistant", "NASA has sent several..."),
    ("04", "user", "What are the main challenges"),
    ("05", "assistant", "The main challenges..."),
    ("06", "user", "How long would a trip take"),
    ("07", "assistant", "A trip would take..."),
    ("08", "user", "What kind of supplies"),
    ("09", "assistant", "Astronauts would need..."),
    ("0a", "user", "Could we terraform Mars"),  # This should be in mars-rover!
    ("0b", "assistant", "Terraforming Mars..."),   # This too!
    ("0c", "user", "I want to learn Italian pasta"),  # Topic change here
]

print("Expected topic extensions:")
print("-" * 50)

current_topic = None
for i, (node_id, role, content) in enumerate(messages):
    content_preview = content[:40] + "..."
    
    if role == "user":
        print(f"\n{node_id} ({role}): {content_preview}")
        if "Mars" in content or "terraform" in content:
            print("  -> Should be in mars-rover topic")
        elif "Italian" in content:
            print("  -> Topic change detected! Close mars-rover at previous assistant node")
            print(f"  -> Start new topic at {node_id}")
    else:
        print(f"{node_id} ({role}): {content_preview}")
        if current_topic:
            print(f"  -> Extend {current_topic} to {node_id}")
            
print("\n\nWhat actually happened:")
print("mars-rover: 02 → 09 (missing 0a, 0b)")
print("italian-cuisine: 0c → 0f (missing several nodes)")