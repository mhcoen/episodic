# Topic Detection System Fix

## Issues Found

1. **All topics named "ongoing-discussion"**: When a topic change is detected, every new topic is created with the generic name "ongoing-discussion". This prevents multiple topics from being properly tracked.

2. **Broken parent chains**: Multiple conversation roots exist (orphan nodes), breaking the DAG structure and causing topic boundary issues.

3. **Topic boundary validation fails**: The `count_nodes_in_topic` function fails when the start node isn't in the ancestry chain due to broken parent relationships.

## Root Causes

1. **Generic topic naming**: Line 446 in `conversation.py` always creates new topics with name "ongoing-discussion"
2. **No unique topic identification**: Topics need unique names or IDs to be properly tracked
3. **Parent chain breaks**: When new conversations are started without proper parent linking

## Proposed Fixes

### Fix 1: Generate unique topic names for new topics
Instead of always using "ongoing-discussion", generate a unique placeholder name like:
- "topic-1", "topic-2", etc.
- "ongoing-discussion-{timestamp}"
- "topic-{short_uuid}"

### Fix 2: Add topic ID to database schema
Add a unique identifier for topics independent of their name:
```sql
ALTER TABLE topics ADD COLUMN topic_id TEXT UNIQUE;
```

### Fix 3: Handle broken parent chains gracefully
Update `count_nodes_in_topic` to handle cases where start_node isn't in ancestry:
- Return minimum count (2) when chain is broken
- Log warning about broken chain
- Consider alternative ancestry traversal

### Fix 4: Prevent orphan nodes
Ensure all new conversation threads properly link to existing nodes or create proper root nodes.

## Implementation Priority

1. **Immediate fix**: Change line 446 to generate unique topic names
2. **Short-term**: Update topic detection to handle broken chains gracefully  
3. **Long-term**: Add proper topic IDs and improve parent chain management