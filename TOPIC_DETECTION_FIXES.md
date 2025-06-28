# Topic Detection Fixes

## Version 2: Intent-Based Detection

Added a new intent-based topic detection system that complements the JSON schema approach:

### Features

1. **Intent Classification**: Messages are classified into one of four intents:
   - `JUST_COMMENT`: Brief acknowledgments that don't advance conversation
   - `DEVELOP_TOPIC`: Continuing the current topic
   - `INTRODUCE_TOPIC`: Starting a new conversation
   - `CHANGE_TOPIC`: Shifting to a different subject

2. **Two-Step Reasoning**: The model first classifies intent, then determines if it's a topic shift

3. **User-Message Focus**: V2 prompt only analyzes user messages, ignoring assistant responses

4. **Structured JSON Output**: Combines prompt engineering with JSON schema validation

### Configuration

Enable V2 detection:
```bash
/set topic_detection_v2 true
```

### Results

V2 detection correctly identifies:
- ✅ Topic development (Mars → Mars moons)
- ✅ Brief comments that shouldn't trigger changes
- ✅ Clear topic shifts (Mars → Cooking)
- ✅ First topic introduction

---

# Topic Detection Fixes

## Issues Identified

1. **Inconsistent Thresholds**: The code had undocumented behavior where the first 2 topics used half the configured threshold (4 messages instead of 8), causing topics to be created with only 2 user messages.

2. **Overly Sensitive Detection**: The original prompt was too simple and the model was detecting topic changes too aggressively, even for related subjects like "Mars facts" → "Mars rovers".

## Changes Made

### 1. Fixed Threshold Logic (topics.py and conversation.py)

Removed the special case for early topics:
```python
# OLD: Half threshold for first 2 topics
if total_topics <= 2:
    effective_min = max(4, min_messages_before_change // 2)
else:
    effective_min = min_messages_before_change

# NEW: Consistent threshold
effective_min = min_messages_before_change
```

### 2. Improved Topic Detection Prompt (prompts/topic_detection.md)

- Made the prompt clearer about what constitutes a topic change
- Added explicit examples of what should and shouldn't trigger a change
- Introduced the key test: "Could these topics naturally appear in the same conversation?"
- Maintained JSON output format for consistency

### 3. Results

The JSON-based detection with temperature=0 is now working correctly:
- ✅ Correctly identifies continuation of related topics (Mars → Mars rovers)
- ✅ Correctly identifies clear domain shifts (Mars → Cooking)
- ✅ Consistent threshold prevents premature topic creation

## Configuration

Users can adjust the threshold via:
```bash
/set min_messages_before_topic_change 8  # Default is 8
```

The first topic still uses a configurable threshold (default 3) via:
```bash
/set first_topic_threshold 3  # Default is 3
```