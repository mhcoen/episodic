---
name: topic_detection
description: Prompt for detecting topic changes in conversations
version: 1.1
tags: [system, topic-detection]
---

Previous conversation:
{recent_conversation}

New user message:
{new_message}

Is the new message about a DIFFERENT SUBJECT than the conversation?

Topic CHANGES if switching between unrelated subjects:
- Programming → weather = Yes
- Movies → restaurants = Yes  
- Math → history = Yes

Topic SAME if continuing related discussion:
- Python lists → Python debugging = No
- Paris weather → London weather = No
- Italian food → French food = No

Respond with a JSON object containing only an "answer" field with value "Yes" or "No".