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

Is the new message about a COMPLETELY DIFFERENT subject than the conversation?

Answer "Yes" ONLY for clear shifts to unrelated topics:
- Space/astronomy → Cooking = Yes
- Programming → Sports = Yes
- History → Personal health = Yes

Answer "No" for continuing discussions:
- Mars facts → Mars moons = No
- Mars surface → Mars rovers = No  
- Programming basics → Programming errors = No
- Italian food → French food = No

The key test: Could these topics naturally appear in the same conversation? If yes, answer "No".

Respond with a JSON object containing only an "answer" field with value "Yes" or "No".