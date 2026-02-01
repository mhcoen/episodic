# Context Replacement: Complete Mental Model

This document provides a complete mental model of how Episodic builds context for LLM calls, with concrete examples showing exactly what the model sees in each scenario.

---

## Part 1: How Episodic Models Topics

### What Is a Topic?

A **topic** is a contiguous segment of conversation about a coherent subject. Topics have:

- **Start node** - The first user message that initiated the topic
- **Boundary** - The point where conversation shifted to a new subject
- **Exchanges** - User+assistant pairs within the topic
- **Summary** - A compressed representation of the topic's content (generated on-demand)
- **Centroid** - A representative embedding for semantic matching

```
Topic: python-debugging
├── Start: Node 001 "I have an IndexError in my Python script"
├── Exchange 1: (001, 002) - Initial problem description
├── Exchange 2: (003, 004) - Traceback analysis
├── Exchange 3: (005, 006) - Solution found
├── Boundary: Topic ends, new topic begins
├── Summary: "Discussed Python IndexError. List index out of range at line 42..."
└── Centroid: embedding([Exchange 2])  # Most representative exchange
```

### The Reality: Fuzzy Boundaries

The clean model above is a simplification. Real conversations are messier:

**Gradual Transitions**

Topic shifts are often gradual, not instantaneous. A Python debugging conversation might drift toward testing, then toward CI/CD, then toward deployment—each step feeling like a natural continuation, not a clear boundary.

```
Exchange 1-4: Python IndexError
Exchange 5-6: "Maybe add a test for this?"     ← Still Python? Or testing?
Exchange 7-8: "Our CI should catch this"       ← Testing? Or DevOps?
Exchange 9-10: "Let's update the deploy config" ← Clearly different
```

The boundary detection system must pick *some* point, but the "true" boundary is fuzzy. The state machine's SUSPECT→COMMIT mechanism exists precisely because single-exchange signals are unreliable.

**Multiple Active Topics**

Conversations can have multiple topics "active" simultaneously:

```
User: "While we're fixing that Python bug, I'm also wondering about
       the coffee machine in the break room—is it broken?"
```

This message genuinely belongs to *both* topics. The system handles this by:
1. Assigning the exchange to the **current active topic** (Python)
2. Making it retrievable via **anchors** when coffee is later discussed
3. NOT creating a separate coffee topic unless the user persists

**Interleaved Topics**

Real conversations often interleave rather than cleanly switch:

```
Exchange 1: Python question
Exchange 2: Python answer
Exchange 3: Coffee aside         ← Brief interleave
Exchange 4: Back to Python
Exchange 5: Python continues
Exchange 6: "Oh, and about that coffee..." ← Callback
Exchange 7: Python resolution
```

The boundary detector sees this as **one Python topic with embedded coffee references**, not two topics. The persistence requirement (multiple consecutive turns) prevents false boundaries from brief asides.

**Human-Led Dynamics**

Human-AI conversations differ fundamentally from human-human dialog:

- **The human drives topic changes** - The AI responds to what the human says; it doesn't spontaneously introduce new subjects or steer the conversation
- **Asymmetric initiative** - In peer conversations, either party might redirect; here, redirects come almost exclusively from the human side
- **Questions dominate** - Human turns are often questions or requests; AI turns are responses to those questions

This asymmetry simplifies some detection challenges:
- Boundaries occur at **human turns**, not AI turns
- The AI's response to a topic shift confirms/follows the shift, it doesn't cause it
- We can focus boundary detection on analyzing human messages

But it creates others:
- A long AI response might span multiple sub-topics the human asked about
- The human may issue compound requests ("fix the bug and also tell me about coffee")
- Implicit context from prior human-AI sessions may make references opaque

The boundary detector focuses on human turn analysis, treating AI responses as continuations of whatever topic the human established.

**Practical Implications**

Because boundaries are fuzzy:
- **Don't expect perfect topic separation** - Some bleed-through is normal
- **Anchors compensate** - Fine-grained retrieval finds relevant content even if topic assignment is imperfect
- **Summaries capture themes** - Even fuzzy topics get coherent summaries
- **Reactivation uses similarity** - Doesn't require exact boundary matching

The system aims for "good enough" topic modeling that enables useful context assembly, not perfect semantic parsing of human discourse.

### Topic Granularity: Exchanges, Not Messages

The fundamental unit is the **exchange** (user message + assistant response), not individual messages.

- Topics contain 1-N exchanges
- Summaries compress all exchanges in a topic
- Centroids point to the most representative exchange
- Anchors are individual exchanges retrieved by semantic similarity

### How Topics Are Created

Topics are NOT created by user command. They emerge from **boundary detection**:

1. **Neural scorer** estimates boundary probability around each turn
2. **State machine** accumulates evidence before committing
3. **Commit** creates a new topic starting at the boundary point

```
Turn 1: "Help with Python"     → STABLE (no boundary)
Turn 2: "Show me the error"    → STABLE (continuing)
Turn 3: "Thanks, now about coffee" → SUSPECT (possible boundary)
Turn 4: "Pour-over ratios?"    → COMMIT (boundary confirmed)
                                  └── New topic "coffee-brewing" starts at Turn 3
```

Key properties:
- **Drift alone cannot commit** - Only triggers investigation
- **Strong signals commit fast** - High confidence = immediate commit
- **Weak signals need persistence** - Must maintain for N turns
- **Digressions abort** - If you return to old topic, SUSPECT aborts

### What About Singleton References?

A **singleton reference** is when you mention something briefly within a larger topic:

```
Topic: python-debugging (20 exchanges)
├── Exchange 5: "By the way, have you tried French press coffee?"
├── Exchange 6: "I prefer pour-over, but let's focus on Python"
└── [Back to Python discussion]
```

**How this is handled:**

1. **No separate topic created** - The coffee mention is too brief to trigger boundary detection (persistence not met)
2. **Included in Python topic** - The singleton exchange is part of python-debugging
3. **Searchable via anchors** - If you later ask "what did you say about French press?", the anchor retrieval can find Exchange 5 within the Python topic
4. **Not a recall target** - You cannot "reactivate" to a singleton - there's no standalone coffee topic

**The key insight:** Recall granularity is at the **exchange level within a topic**, not just entire topics.

---

## Part 2: The Two Levels of Recall

### Level 1: Topic Reactivation (Coarse-Grained)

When you say "back to that Python discussion", the system:

1. **Matches** your message embedding against topic centroids
2. **Switches** the active topic from current to Python
3. **Builds context** from Python topic only

This is **coarse-grained** - you get the whole topic (summary + recent exchanges).

### Level 2: Anchor Retrieval (Fine-Grained)

Within a topic, the system uses **semantic anchors** to find specific relevant exchanges:

```
Topic: python-debugging (20 exchanges)
│
User asks: "What was that thing about list bounds?"
│
Anchor query against ChromaDB (filtered to python-debugging topic):
├── Similarity 0.89: Exchange 4 "Check if the list is empty before..."
├── Similarity 0.72: Exchange 2 "Can you show me the traceback?"
└── Similarity 0.65: Exchange 8 "Try-except block for safety"
│
Top 2 anchors included in context
```

**Anchors retrieve individual exchanges**, not entire topics. This handles:

- Long topics where recency alone misses relevant history
- Specific callbacks to earlier parts of the current topic
- Singleton references embedded in larger topics

### How They Work Together

```
User: "Back to Python - what was that bounds check again?"
        │
        ├── Topic Reactivation: Switch to python-debugging
        │   └── Excludes coffee topic entirely
        │
        └── Anchor Retrieval (within python-debugging):
            └── Finds Exchange 4 (bounds check) even if it's not recent
```

**Context assembled:**
```
[system] Topic: python-debugging
         Summary: Discussed IndexError. List index out of range...

         Relevant Past Context:
         Exchange 4: "Check if the list is empty before accessing..."

[user] Exchange 19 (recent)
[assistant] Exchange 20 (recent)
[user] "Back to Python - what was that bounds check again?"
```

---

## Part 3: What Happens to Singletons

### Case 1: Singleton Within Topic (No Separate Topic)

```
Topic: python-debugging
├── Exchange 1-4: Python discussion
├── Exchange 5: "By the way, French press coffee is great"
├── Exchange 6: "Yeah, anyway back to Python..."
└── Exchange 7-20: More Python
```

**If you ask "what about coffee?":**

- **Anchor retrieval** finds Exchange 5 within python-debugging
- Exchange 5 appears in "Relevant Past Context"
- No topic switch occurs (coffee isn't a topic)

### Case 2: Brief Topic (Below Persistence Threshold)

```
Topic A: python-debugging
Exchange 1-10: Python discussion

[User mentions coffee - SUSPECT triggered]
Exchange 11: "Speaking of coffee..."
Exchange 12: "Pour-over is best"

[User returns to Python - SUSPECT aborted]
Exchange 13: "Anyway, about that error..."
Exchange 14-20: Python continues

Topic A continues (no coffee topic created)
```

**Result:** Coffee exchanges 11-12 are part of python-debugging, retrievable via anchors.

### Case 3: Established Topic (Above Persistence Threshold)

```
Topic A: python-debugging
Exchange 1-10: Python discussion

[User switches to coffee - SUSPECT triggered]
Exchange 11: "Let's talk coffee"
Exchange 12: "Pour-over ratios?"
Exchange 13: "Water temperature?"
Exchange 14: "Bloom time?"
[Persistence threshold met - COMMIT]

Topic B: coffee-brewing
Exchange 11-14: Coffee discussion (backdated to Exchange 11)

[User switches back]
Topic A resumed
Exchange 15: "Back to Python..."
```

**Result:** Coffee is now a separate topic. Reactivation works. Topic A excludes Exchanges 11-14.

---

## Part 4: What Is "Context"?

When you send a message, the LLM doesn't see your entire conversation history. It sees a **context window** - a curated list of messages assembled from your history. This assembly is called **context replacement**.

**The key insight:** Different context assembly strategies produce different message lists, which produce different LLM responses.

---

## Part 5: The Two Context Strategies

### Strategy 1: Ancestry (Traditional)

Walks backward from the current position in the conversation DAG, including recent messages regardless of topic boundaries.

```
What LLM sees:
┌────────────────────────────────────────────────────────┐
│ [system] You are a helpful assistant...               │
│ [user] Help me debug this IndexError                  │  ← 10 turns ago
│ [assistant] Check the list bounds...                  │
│ [user] What about pour-over coffee ratios?            │  ← 5 turns ago
│ [assistant] Use 1:15 coffee to water...               │
│ [user] Back to that Python error                      │  ← NOW
└────────────────────────────────────────────────────────┘
```

**Problem:** The assistant sees both Python AND coffee context mixed together.

### Strategy 2: Topic-Local

Builds context from a single topic only. Other topics are completely excluded.

```
What LLM sees (after reactivating Python topic):
┌────────────────────────────────────────────────────────┐
│ [system] You are a helpful assistant...               │
│ [system] Topic: python-debugging                      │
│ [system] Summary: Discussed IndexError in list...     │
│ [user] Help me debug this IndexError                  │
│ [assistant] Check the list bounds...                  │
│ [user] Back to that Python error                      │  ← NOW
└────────────────────────────────────────────────────────┘
```

**Result:** Coffee discussion is completely absent. The assistant responds purely in Python context.

---

## Part 6: Complete Example - A→B→A Resume

### The Conversation (as stored in database)

```
Node 001 [user]    "I have an IndexError in my Python script"
Node 002 [assistant] "Can you show me the traceback?"
Node 003 [user]    "It says list index out of range at line 42"
Node 004 [assistant] "Check if the list is empty before accessing..."
Node 005 [user]    "That fixed it, thanks!"
Node 006 [assistant] "You're welcome!"
                   ─── Topic boundary: "python-debugging" ends ───
Node 007 [user]    "What's a good pour-over coffee ratio?"
Node 008 [assistant] "A common ratio is 1:15 coffee to water..."
Node 009 [user]    "Should I use a gooseneck kettle?"
Node 010 [assistant] "Yes, for better pour control..."
                   ─── Current head ───
Node 011 [user]    "Actually, back to that Python bug - what was the fix?"
```

### What Happens at Node 011

#### Step 1: Compute Embedding
```python
user_embedding = embed("Actually, back to that Python bug - what was the fix?")
# Returns: [0.23, -0.45, 0.67, ...]  # 384-dim vector
```

#### Step 2: Reactivation Probe
```python
decision = probe_reactivation(
    user_input="Actually, back to that Python bug - what was the fix?",
    user_embedding=user_embedding,
    active_topic_start_node_id="007",  # Currently in coffee topic
    ...
)
```

**Probe checks (gates):**
1. **Cooldown** - No recent reactivation? ✅ Pass
2. **Length** - Input >= 4 words? ✅ Pass (9 words)
3. **Topics exist** - Any topics with centroids? ✅ Pass
4. **Dormancy** - Python topic dormant >= 4 turns? ✅ Pass (6 turns)
5. **Similarity** - User embedding similar to Python centroid? ✅ Pass (0.72 > 0.3)
6. **Support** - Multiple exchanges match Python? ✅ Pass (3 matches)
7. **Rank gap** - Python clearly better than coffee? ✅ Pass

**Result:**
```python
ReactivationDecision(
    action="REACTIVATE",
    topic_name="python-debugging",
    topic_start_node_id="001",
    confidence=0.85,
    reason="Strong semantic match to dormant topic"
)
```

#### Step 3: Context Assembly (Topic-Local)

Since decision is REACTIVATE, use `TopicLocalStrategy`:

```python
result = TopicLocalStrategy().assemble(
    user_turn_text="Actually, back to that Python bug - what was the fix?",
    active_topic_start_node_id="001",  # Python topic
    token_budget=4000,
    ...
)
```

**Assembly process:**

1. **Get topic summary** (from `topic_working_set`)
   ```
   "Discussed Python IndexError. User had list index out of range
    at line 42. Fixed by checking if list is empty before accessing."
   ```

2. **Get last N exchanges** (from `topic_nodes`, filtered to topic 001)
   ```
   Node 003: "It says list index out of range at line 42"
   Node 004: "Check if the list is empty before accessing..."
   Node 005: "That fixed it, thanks!"
   Node 006: "You're welcome!"
   ```

3. **Assemble final message list:**
   ```python
   [
       {"role": "system", "content": "You are a helpful assistant..."},
       {"role": "system", "content": "Topic: python-debugging\n\nSummary: Discussed Python IndexError..."},
       {"role": "user", "content": "It says list index out of range at line 42"},
       {"role": "assistant", "content": "Check if the list is empty before accessing..."},
       {"role": "user", "content": "That fixed it, thanks!"},
       {"role": "assistant", "content": "You're welcome!"},
       {"role": "user", "content": "Actually, back to that Python bug - what was the fix?"}
   ]
   ```

**Critical observation:** Nodes 007-010 (coffee discussion) are **completely absent**.

#### Step 4: LLM Call

```python
response = llm.chat(messages=result.messages)
# Returns: "The fix was to check if the list is empty before
#           accessing it. You added a check like: if mylist: value = mylist[0]"
```

The assistant correctly recalls the Python context without any coffee confusion.

---

## Ancestry Strategy: What Would Have Happened

If reactivation had NOT fired (e.g., user said "what's next?" instead), ancestry strategy would run:

```python
result = AncestryStrategy().assemble(
    user_turn_text="what's next?",
    token_budget=4000,
    ...
)
```

**Assembly process:**

1. **Walk backward from head** (Node 010)
2. **Include recent messages** until budget exhausted

**Result:**
```python
[
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Should I use a gooseneck kettle?"},
    {"role": "assistant", "content": "Yes, for better pour control..."},
    {"role": "user", "content": "What's a good pour-over coffee ratio?"},
    {"role": "assistant", "content": "A common ratio is 1:15..."},
    {"role": "user", "content": "what's next?"}
]
```

Python context is NOT included - it's too far back.

---

## Hybrid Mode: Automatic Selection

In `hybrid` mode (the default), strategy selection is automatic:

```python
def select_strategy(mode, reactivation_decision):
    if mode == HYBRID:
        if reactivation_decision == "REACTIVATE":
            return TopicLocalStrategy()
        else:
            return AncestryStrategy()
```

**Decision matrix:**

| Reactivation Decision | Strategy Used | What Happens |
|-----------------------|---------------|--------------|
| CONTINUE | Ancestry | Recent messages, may cross topics |
| REACTIVATE | Topic-Local | Single topic only, others excluded |
| DISAMBIGUATE | (user chooses) | Then applies as REACTIVATE or CONTINUE |

---

## The "B Disappears" Guarantee

When topic-local assembly is used for topic A:

1. **Only nodes in `topic_nodes` table with `topic_start_node_id=A` are considered**
2. **No ancestry walk occurs** - we don't traverse the DAG
3. **Other topics cannot "leak in"** - there's no path to them

```sql
-- This is the query that gets topic nodes:
SELECT n.* FROM nodes n
JOIN topic_nodes tn ON n.id = tn.node_id
WHERE tn.topic_start_node_id = ?
ORDER BY tn.turn_idx DESC
LIMIT ?
```

**Guarantee:** If node X is in topic B, and you're assembling for topic A, node X will NOT appear in the query results.

---

## Token Budget and Truncation

Context has a token budget (default: 4000). When content exceeds budget:

### Topic-Local Truncation

1. System prompt is always included
2. Summary is always included (it's compressed)
3. Exchanges are dropped oldest-first

```
Budget: 4000 tokens

System prompt:     200 tokens  ✓
Topic summary:     150 tokens  ✓
Exchange 1 (old):  300 tokens  ✗ DROPPED
Exchange 2:        400 tokens  ✓
Exchange 3:        350 tokens  ✓
Exchange 4:        300 tokens  ✓
Current message:   100 tokens  ✓
─────────────────────────────
Total:            1500 tokens  (under budget)
```

### Ancestry Truncation

1. System prompt is always included
2. Messages dropped oldest-first from ancestry chain

---

## Database Tables Involved

### nodes
The conversation DAG. Every message is a node.

```
id       | parent_id | role      | content
─────────┼───────────┼───────────┼─────────────────
001      | NULL      | user      | "I have an IndexError..."
002      | 001       | assistant | "Can you show me..."
003      | 002       | user      | "It says list index..."
...
```

### topic_nodes
Maps nodes to topics. Enables "get all nodes in topic X".

```
topic_start_node_id | node_id | turn_idx | role
────────────────────┼─────────┼──────────┼──────
001                 | 001     | 1        | user
001                 | 002     | 2        | assistant
001                 | 003     | 3        | user
...
007                 | 007     | 1        | user      ← Different topic
007                 | 008     | 2        | assistant
```

### topic_working_set
Stores topic summaries and metadata.

```
topic_start_node_id | topic_name        | summary_md
────────────────────┼───────────────────┼──────────────────────
001                 | python-debugging  | "Discussed IndexError..."
007                 | coffee-brewing    | "Covered pour-over ratios..."
```

### topic_centroids
Stores medoid embeddings for ANN-based topic retrieval.

```
start_node_id | centroid_medoid_exchange_id | exchange_count
──────────────┼─────────────────────────────┼───────────────
001           | 003                         | 3
007           | 008                         | 2
```

---

## The Probe Decision Tree

```
probe_reactivation(user_input, user_embedding, active_topic, ...)
│
├─ Is cooldown active?
│  └─ YES → CONTINUE (wait 3 turns)
│
├─ Is input too short (< 4 words)?
│  └─ YES → CONTINUE (not enough signal)
│
├─ Any topics with centroids?
│  └─ NO → CONTINUE (nothing to reactivate to)
│
├─ Find best matching topic by embedding similarity
│  │
│  ├─ Best topic = active topic?
│  │  └─ YES → CONTINUE (already there)
│  │
│  ├─ Best topic too recent (< 4 turns dormant)?
│  │  └─ YES → CONTINUE (not dormant enough)
│  │
│  ├─ Similarity too low (< 0.3)?
│  │  └─ YES → CONTINUE (no good match)
│  │
│  ├─ Support too low (< 2 matching exchanges)?
│  │  └─ YES → CONTINUE (not enough evidence)
│  │
│  ├─ Multiple topics competitive (rank gap < 0.1)?
│  │  └─ YES → DISAMBIGUATE (let user choose)
│  │
│  └─ All checks pass?
│     └─ YES → REACTIVATE to best topic
```

---

## Configuration Reference

| Setting | Default | Effect |
|---------|---------|--------|
| `context_recovery_mode` | `hybrid` | Which strategy to use |
| `context_token_budget` | `4000` | Max tokens in context |
| `enable_topic_reactivation` | `true` | Enable/disable probe |
| `reactivation_cooldown` | `3` | Turns to wait after switch |
| `reactivation_dormancy_min` | `4` | Min turns before topic eligible |
| `reactivation_support_threshold` | `2` | Min matching exchanges |
| `min_anchors_for_topic_local` | `2` | Min topic nodes for topic-local |
| `min_tokens_for_topic_local` | `500` | Min topic content for topic-local |

---

## Debug Output Interpretation

With `/set debug memory`:

```
[REACTIVATION] REACTIVATE to "python-debugging"
  confidence: 0.85
  best_similarity: 0.72
  active_similarity: 0.31
  support_count: 3
  dormancy_turns: 6
  gates_passed: [cooldown, length, topics_exist, dormancy, similarity, support, rank_gap]
  gates_failed: []

[CONTEXT] TopicLocalStrategy assembled 7 messages (1847 tokens)
  topic: python-debugging (001)
  summary_tokens: 150
  exchange_tokens: 1200
  included_nodes: [001, 002, 003, 004, 005, 006]
  excluded_by_budget: []
```

---

## Edge Cases

### Thin Topic Fallback

If a topic has insufficient content:
- No summary (summary_md is empty)
- Few exchanges (< min_anchors_for_topic_local)
- Low token count (< min_tokens_for_topic_local)

**Behavior:** Falls back to ancestry strategy for that request. Topic context is supplemented with recent ancestry.

### Cross-Topic Import

When user explicitly references another topic:
- "Like we discussed in the Python topic..."
- The system may inject relevant context from the referenced topic

This is **not** automatic reactivation - it's explicit import.

### Disambiguation Timeout

If user doesn't respond to disambiguation:
- 2 invalid inputs → auto-CONTINUE
- Ctrl+C → CONTINUE

---

## Visual Summary

```
User message arrives
        │
        ▼
┌─────────────────────┐
│ Compute embedding   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────────┐
│ Reactivation probe  │────▶│ CONTINUE: Use ancestry  │
└─────────┬───────────┘     └─────────────────────────┘
          │
          │ REACTIVATE
          ▼
┌─────────────────────┐
│ Set active topic    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Topic-local assemble│──── Only nodes from this topic
└─────────┬───────────┘     Other topics excluded
          │
          ▼
┌─────────────────────┐
│ LLM receives clean  │
│ single-topic context│
└─────────────────────┘
```
