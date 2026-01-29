# Topic Reactivation User Guide

This guide explains how Episodic's topic reactivation system works and how to use it effectively.

---

## What Is Topic Reactivation?

Topic reactivation automatically detects when you're returning to a previously discussed topic and brings back the relevant context from that topic.

**Key idea:** When you say "remember that Python bug we discussed?", Episodic detects you're returning to the Python topic and switches context to show only that topic's history—the coffee discussion from between disappears.

---

## Mental Model

Think of your conversation as having multiple "topic threads":

```
Topic A (Python debugging)
├── "I have an IndexError"
├── "Try checking the list bounds"
└── "What about try-except?"

Topic B (Coffee brewing)
├── "Best pour-over ratio?"
└── "Use 1:15 coffee to water"

Topic A resumed
└── "What was that Python fix again?"  ← You're here
```

When you return to Topic A, the system:
1. Recognizes your message relates to Python, not coffee
2. Switches context to Topic A only
3. Excludes Topic B from the prompt entirely
4. Retrieves Topic A's summary and relevant exchanges

**Result:** The assistant responds with Python context, not coffee context.

---

## Common Scenarios

### Scenario 1: A→B→A Resume (No Ambiguity)

**Conversation:**
```
You: Help me debug this IndexError in Python
[... Python discussion for 10 exchanges ...]

You: Let's talk about coffee brewing
[... Coffee discussion for 5 exchanges ...]

You: Back to that Python error - what was the fix?
```

**What happens:**
1. System detects "Python" relates to dormant topic
2. Returns `REACTIVATE` decision
3. Context built from Python topic only
4. Assistant responds with Python context

**You see:** Assistant remembers the IndexError discussion accurately.

### Scenario 2: Ambiguous Headword → Disambiguation

**Conversation:**
```
Topic "java-programming": "How do I use Java streams?"
Topic "java-coffee": "Best Java coffee beans?"

You: "More about Java"
```

**What happens:**
1. System detects multiple topics match "Java"
2. Returns `DISAMBIGUATE` decision
3. Shows disambiguation prompt

**You see:**
```
I found multiple topics that might match:

[1] java-programming (12 turns ago)
    - "How do I use Java streams?"
    - "What about parallel streams?"
    3 matching exchanges

[2] java-coffee (45 turns ago)
    - "Best Java coffee beans?"
    2 matching exchanges

[0] Neither / Continue current topic

Which topic?
```

**Your response:**
- Enter `1` to switch to Java programming
- Enter `2` to switch to Java coffee
- Enter `0` to continue current topic

### Scenario 3: Thin Topic → Fallback

**Conversation:**
```
Topic "quick-question": Just 1 exchange, no summary
You: "Remember that quick thing I asked?"
```

**What happens:**
1. System detects "quick question" topic
2. Topic is "thin" (no summary, few exchanges)
3. Falls back to traditional context (recent messages)
4. No error, just uses more context

**You see:** Normal response using recent conversation history.

---

## Disambiguation Options Explained

When disambiguation is triggered, each option shows:

| Element | Meaning |
|---------|---------|
| **Topic name** | What the topic was about |
| **N turns ago** | How long since last activity in that topic |
| **Snippets** | Sample messages from that topic (your words) |
| **N matching exchanges** | How many of your recent messages matched this topic |

**Example:**
```
[1] python-debugging (12 turns ago)
    - "How do I fix IndexError?"
    - "What about try-except?"
    3 matching exchanges
```

This means:
- Topic is about Python debugging
- Last active 12 turns ago
- Your messages about Python include the two snippets shown
- 3 of your recent messages semantically match this topic

---

## Skipping Disambiguation

If you don't want to choose a topic:

1. **Enter `0`** - Continue with current topic
2. **Enter invalid input twice** - System auto-continues
3. **Press Ctrl+C** - Cancels and continues current topic

---

## Configuration Options

### Enable/Disable Reactivation

```bash
# Topic reactivation is enabled by default
# To disable:
/set enable_topic_reactivation false

# To re-enable:
/set enable_topic_reactivation true
```

### Context Recovery Mode

```bash
# Hybrid (recommended): Uses topic-local when reactivating
/set context_recovery_mode hybrid

# Traditional: Always uses recent messages regardless of topic
/set context_recovery_mode ancestry

# Topic-local: Always isolates to current topic
/set context_recovery_mode topic_local
```

### Token Budget

```bash
# Maximum tokens for context (default: 4000)
/set context_token_budget 4000

# Increase for longer conversations
/set context_token_budget 6000
```

### Thin Topic Thresholds

```bash
# Minimum anchors before using topic-local (default: 2)
/set min_anchors_for_topic_local 2

# Minimum tokens before using topic-local (default: 500)
/set min_tokens_for_topic_local 500
```

---

## Debug Log Interpretation

Enable memory-specific debug output to see reactivation decisions without topic detection noise:

```bash
# Memory debugging only (reactivation, recall, context recovery)
/set debug memory

# All debug output (includes topic detection, drift, etc.)
/set debug true
```

### Example Debug Output

```
[REACTIVATION] REACTIVATE to "python-debugging"
  confidence: 0.85
  best_similarity: 0.72
  support_count: 3
  dormancy_turns: 15
  gates_passed: [cooldown, length, topics_exist, dormancy, similarity, support]
  gates_failed: []
```

**Key fields:**

| Field | Meaning |
|-------|---------|
| `confidence` | How confident (0-1) the decision is correct |
| `best_similarity` | Semantic similarity to best topic (0-1) |
| `support_count` | How many recent exchanges match this topic |
| `dormancy_turns` | Turns since topic was last active |
| `gates_passed` | Which checks passed |
| `gates_failed` | Why reactivation was blocked (if CONTINUE) |

### Common Gate Failures

| Gate | Why It Failed |
|------|---------------|
| `cooldown` | Reactivation happened recently (wait 3 turns) |
| `length` | Input too short (< 4 words) |
| `topics_exist` | No topics with centroids in database |
| `dormancy` | Topic was active too recently |
| `similarity` | No topic similar enough (< 0.3) |
| `support` | Not enough matching exchanges (< 2) |
| `rank_gap` | Active topic too close to best candidate |

---

## Guarantees

### Contamination = 0%

When in topic A, **no content from topic B appears in context**.

- This is the core guarantee
- Prevents confusing the assistant with unrelated context
- Verified by automated tests

### No Accidental Switches

Topic switches only happen when:
1. You explicitly switch topics (topic boundary detected)
2. You reference a dormant topic with sufficient evidence
3. You select a disambiguation option

**Protected by:**
- Support threshold: Need 2+ matching exchanges
- Cooldown: No switch for 3 turns after last switch
- Dormancy: Topic must be inactive 4+ turns

---

## Failure Modes & Troubleshooting

### Problem: Wrong Topic Detected

**Symptom:** Reactivates to wrong topic

**Possible causes:**
1. Similar vocabulary across topics
2. Ambiguous keywords

**Solutions:**
- Be more specific in your message
- Use topic-specific terms
- If disambiguation appears, select correct option

### Problem: Reactivation Doesn't Trigger

**Symptom:** Expected to return to topic, but stayed in current

**Possible causes:**
1. Topic too recent (< 4 turns dormant)
2. Input too short (< 4 words)
3. Not enough support (< 2 matching exchanges)
4. Cooldown active (recent reactivation)

**Solutions:**
- Add more context to your message
- Wait a few more turns
- Check debug output for specific gate failure

### Problem: Thin Topic Fallback

**Symptom:** Topic reactivation falls back to traditional context

**Cause:** Topic doesn't have enough history (no summary, few exchanges)

**This is expected behavior.** The system uses the best available context.

### Problem: Too Many Disambiguation Prompts

**Symptom:** Frequently asked to choose between topics

**Possible causes:**
1. Topics have overlapping vocabulary
2. Vague messages that could match multiple topics

**Solutions:**
- Use more specific language
- Name the topic explicitly ("back to Python debugging")

---

## Commands Reference

### Evaluation Commands

```bash
# See reactivation replay stats
/evaluate reactivation

# Run quality evaluation on test moments
/evaluate quality

# Run calibration (advanced)
/evaluate calibrate
```

### Debug Commands

```bash
# Enable debug output
/set debug true

# Check current mode
/get context_recovery_mode

# Check if reactivation enabled
/get enable_topic_reactivation
```

---

## FAQ

**Q: Will reactivation lose my current context?**
A: No. Your current topic is still in the database. You can return to it by mentioning it.

**Q: How long does topic history persist?**
A: Indefinitely. Summaries are stored in the database and retrieved on demand.

**Q: Can I disable reactivation for a single message?**
A: Not directly, but you can be explicit: "Continuing our current discussion..."

**Q: What if I want to bring in context from another topic?**
A: Mention it explicitly: "Like we discussed in the Python topic, try-except blocks..."
The cross-topic import system may bring in relevant context.

**Q: How do I know which mode I'm in?**
A: Run `/get context_recovery_mode`. With `debug true`, you'll also see mode in each turn's output.
