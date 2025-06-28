---
name: topic_detection_ollama
description: Simplified prompt for detecting topic changes with Ollama models
version: 1.1
tags: [system, topic-detection, ollama]
---

Previous conversation:
{recent_conversation}

New message:
{new_message}

Is this a shift to an UNRELATED topic?

Examples:
- Mars rovers → Mars missions = No (same topic)
- Italian pasta → French cuisine = No (related)
- Cooking → Neural networks = Yes (unrelated)
- Space travel → Programming = Yes (unrelated)

Reply with only: Yes or No