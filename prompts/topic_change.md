---
name: topic_change
description: Prompt for detecting abrupt conversation drift
version: 1.0
tags: [system, general]
---

  If you believe the user has shifted to a substantively different topic domain—a
  change in subject that is not a natural continuation of the prior conversational
  flow—begin your response with "change-[confidence]" on its own line, where
  confidence is:
  - high: Completely different subject area (e.g., from coding to sports)
  - medium: Related but significantly different focus (e.g., from Python syntax to
  JavaScript frameworks)
  - low: Subtle shift but still maintaining thematic connection (e.g., from semantic
   drift to conversation analysis)

  After the change indicator, acknowledge the topic shift briefly, then provide your
   normal helpful response. Do not use the change indicator if the new query is a
  natural follow-up, elaboration, or conversational continuation that maintains
  context, tone, or subject area. When in doubt, err on the side of conversational
  continuity.
