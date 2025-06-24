---
name: topic_detection
description: Prompt for detecting topic changes in conversations
version: 1.0
tags: [system, topic-detection]
---

Previous conversation:
{recent_conversation}

New user message:
{new_message}

Is the new message about a DIFFERENT SUBJECT than the conversation?

Topic CHANGES if switching between unrelated subjects:
- Programming → weather = YES: weather
- Movies → restaurants = YES: restaurants  
- Math → history = YES: history

Topic SAME if continuing related discussion:
- Python lists → Python debugging = NO
- Paris weather → London weather = NO
- Italian food → French food = NO

Answer YES: [new-topic] or NO: