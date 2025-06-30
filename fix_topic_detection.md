# Topic Detection Fix Recommendations

## Current Issues

1. **Configuration**: `min_messages_before_topic_change` is set to 2 (should be 8)
2. **Boundary Lag**: Topics are created 2-4 messages after the actual change
3. **Detection Timing**: The system detects changes too late due to the threshold

## Immediate Fixes

### 1. Configuration (Already Applied)
```bash
/set min_messages_before_topic_change 8
```

### 2. Adjust Hybrid Detection (If Enabled)
```bash
# Check if enabled
/set hybrid_topics

# If enabled, make it less sensitive:
/set hybrid_topic_weights {"semantic_drift": 0.3, "keyword_explicit": 0.4, "keyword_domain": 0.2, "message_gap": 0.05, "conversation_flow": 0.05}
/set hybrid_topic_threshold 0.7
```

### 3. Disable Hybrid Detection (Alternative)
```bash
/set use_hybrid_topic_detection false
```

## Long-term Code Fix Needed

The topic boundary assignment logic needs to be revised to:

1. When a topic change is detected at message N:
   - Look back to find where the topic actually changed (usually N-2 or N-3)
   - Set the previous topic end at the last message of the old topic
   - Set the new topic start at the first message of the new topic

2. Implement a "topic change confirmation" that:
   - Detects potential changes early
   - Confirms after 2-3 messages
   - Then retroactively sets correct boundaries

## Current Workaround

With `min_messages_before_topic_change=8`, future conversations will have better topic boundaries, though there will still be some lag in detection.