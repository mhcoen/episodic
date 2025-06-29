---
name: topic_detection_v3
description: Domain-agnostic topic detection based on conversational principles
version: 1.1
tags: [system, topic-detection]
---

Previous conversation:
{recent_conversation}

New message:
{new_message}

Determine if this message starts a COMPLETELY NEW topic area.

Consider these as the SAME topic:
- Different aspects of the same subject (e.g., Python basics and Python web frameworks)
- Natural progression of learning (e.g., what is X → how does X work → using X)
- Related concepts within a field (e.g., neural networks → backpropagation → optimizers)
- Practical applications of theory being discussed
- Follow-up questions or deeper exploration

Only answer YES for:
- Complete shift to unrelated subject requiring different expertise
- Topics that would be taught in completely different university courses
- Explicit "changing topics" or "different question" transitions to unrelated areas

Default to NO unless the shift is absolutely clear and unrelated.

Reply with only: Yes or No