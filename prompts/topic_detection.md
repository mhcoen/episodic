---
name: topic_detection
description: Prompt for detecting topic changes in conversations
version: 1.0
tags: [system, topic-detection]
---

Analyze if there is a MAJOR topic change in this conversation.

IMPORTANT: Only detect changes when switching between COMPLETELY DIFFERENT subjects.
Minor variations within the same general subject are NOT topic changes.

Previous conversation:
{recent_conversation}

New user message:
{new_message}

Has the topic changed SIGNIFICANTLY? Answer with ONLY:
- "YES: [new-topic-name]" if there is a MAJOR topic change (use 1-3 words, lowercase with hyphens)
- "NO" if continuing the same general subject area

Examples of NO CHANGE (continuing same topic):
- "What color is the sky?" → "What color are roses?" (both about colors)
- "What is 2+2?" → "What is 10*10?" (both about math)
- "How do I debug Python?" → "What about performance optimization?" (both about programming)
- "Tell me about dogs" → "What about cats?" (both about animals)
- "Explain quantum physics" → "What about relativity?" (both about physics)

Examples of YES CHANGE (major topic shift):
- "How do I debug Python?" → "What's the weather today?" → "YES: weather"
- "Tell me about quantum physics" → "What's a good restaurant nearby?" → "YES: restaurants"
- "What is 2+2?" → "Tell me about ancient Rome" → "YES: ancient-rome"

Answer: