# Topic Boundary Analysis

## Overview

When Episodic detects a topic change, it now analyzes the conversation history to find where the topic *actually* changed, rather than just using the detection point. This provides more accurate topic boundaries.

## The Problem

Previously, when a topic change was detected at message N:
- The previous topic would end at message N-1 (the assistant's last response)
- The new topic would start at message N

However, the actual topic transition often occurred earlier:
- User asks about new topic at N-2
- Assistant responds about new topic at N-1  
- User continues on new topic at N (detection happens here)

This caused messages N-2 and N-1 to be incorrectly assigned to the old topic.

## The Solution

The topic boundary analyzer looks backwards from the detection point to find where the transition actually occurred by:

1. **LLM-based analysis** (default): Uses the configured topic detection model to analyze recent messages and identify the transition point
2. **Heuristic analysis** (fallback): Looks for transition phrases like "let me ask about a different topic"

## Configuration

The feature can be configured with these settings:

```python
# Enable/disable boundary analysis (default: True)
config.set("analyze_topic_boundaries", True)

# Use LLM for analysis vs heuristics only (default: True)
config.set("use_llm_boundary_analysis", True)
```

## How It Works

1. Topic change is detected at user message N
2. System retrieves the last ~20 messages for context
3. Boundary analyzer examines the conversation to find the actual transition
4. The previous topic's end boundary is set at the message BEFORE the transition
5. The new topic starts at the transition point

## Example

```
User: How do I cook pasta?                    # Topic: cooking
Assistant: Boil water, add pasta...            # Topic: cooking
User: What sauce should I use?                 # Topic: cooking
Assistant: Try a simple tomato sauce...        # Topic: cooking
User: Thanks! Can you help with Python?        # ACTUAL TRANSITION
Assistant: Sure! What do you need help with?   # New topic response
User: I need to sort a list                    # DETECTION HAPPENS HERE
```

Previously: 
- Cooking topic would end at "Sure! What do you need help with?"
- Python topic would start at "I need to sort a list"

Now:
- Cooking topic ends at "Try a simple tomato sauce..."
- Python topic starts at "Thanks! Can you help with Python?"

## Testing

Run the boundary analysis tests:
```bash
python scripts/test_boundary_analysis.py
```

Test the feature in conversation:
```bash
python -m episodic < scripts/test-boundary-fix.txt
```