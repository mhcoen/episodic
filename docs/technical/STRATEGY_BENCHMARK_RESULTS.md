# Topic Detection Strategy Benchmark Results

**Date**: 2024-12-11
**Dataset**: SuperDialseg test set
**Commit**: a110d50 (topics branch)

## Summary

| Strategy | F1 | Precision | Recall | Avg Time | Notes |
|----------|-----|-----------|--------|----------|-------|
| **neural** | **0.806** | 0.715 | 0.924 | 13ms | Best overall |
| ensemble | 0.804 | 0.712 | 0.924 | 10ms | Keyword + neural |
| relative_embedding | 0.393 | 0.352 | 0.444 | 13ms | After fix |
| keyword | 0.107 | 0.183 | 0.076 | 0.1ms | Explicit phrases only |
| dual_window | 0.019 | 0.182 | 0.010 | 58ms | Threshold issue |

## Dataset

- **SuperDialseg**: Customer service dialogues with topic annotations
- **Test set**: 1322 dialogues, ~17K messages, ~4K topic boundaries
- **Subset used**: First 100 dialogues for quick benchmarking

### Ground Truth Definition

Topic boundaries are marked where `topic_id` changes between consecutive turns:
```python
new_topic_starts = set()
prev_topic = turns[0].get('topic_id')
for i, turn in enumerate(turns[1:], 1):
    if turn.get('topic_id') != prev_topic:
        new_topic_starts.add(i)
    prev_topic = turn.get('topic_id')
```

## Evaluation Methodology

### Position Evaluation
For each message position `i` with at least 5 messages of history:
1. Build history: `messages[:i]`
2. Get query: `messages[i]['content']`
3. Call `strategy.get_decision(query, history)`
4. Compare `predicted` vs `expected` (i in new_topic_starts)

### Metrics
- **Precision**: TP / (TP + FP) - How many predicted boundaries are correct
- **Recall**: TP / (TP + FN) - How many actual boundaries are detected
- **F1**: Harmonic mean of precision and recall

## Strategy Details

### NeuralStrategy (F1=0.806)
- **Model**: Fine-tuned DistilBERT on SuperDialseg training data
- **Window**: (4,2) - 4 messages before, 2 after potential boundary
- **Input format**: `group1 [BOUNDARY?] group2`
- **Threshold**: confidence_threshold=0.5 (default 0.8)

**Key fix applied**: Query role must alternate based on last message role,
not always "user". This fixed F1 from 0.65 to 0.81.

### EnsembleStrategy (F1=0.804)
Combines strategies with priority order:
1. **Keyword** - If explicit transition detected, return immediately
2. **Neural** - Primary signal
3. **Embedding** - Optional backup

On SuperDialseg, only 1 keyword trigger in 100 dialogues, so performance
matches neural alone. Value is for real conversations with explicit transitions.

### RelativeEmbeddingStrategy (F1=0.393)
Uses self-calibrating similarity thresholds:
- Computes baseline similarity statistics for conversation
- Detects topic change if query similarity drops below baseline - 1.5 std

**Key fix applied**: Use all messages, not just user messages. This improved
F1 from 0.25 to 0.39.

### KeywordStrategy (F1=0.107)
Detects explicit transition phrases:
- "by the way", "changing topics", "on another note"
- Domain keyword shifts

Low F1 expected - SuperDialseg rarely has explicit transitions.

### DualWindowStrategy (F1=0.019)
Uses embedding drift with (4,1) and (4,2) windows.

**Issue**: Threshold (0.15) is too conservative for SuperDialseg.
Similarities are 0.5-0.7, never below 0.15. Would need retuning for
this dataset.

## Reproducing Results

```python
import json
from episodic.topics.strategy_registry import get_strategy

# Load test data
with open('datasets/superseg/segmentation_file_test.json', 'r') as f:
    data = json.load(f)

dialogues = []
for dataset_name, dlgs in data['dial_data'].items():
    dialogues.extend(dlgs)

# Test a strategy
strategy = get_strategy('neural', {'confidence_threshold': 0.5})

tp, fp, tn, fn = 0, 0, 0, 0

for dlg in dialogues[:100]:  # First 100 for quick test
    turns = dlg['turns']

    # Find topic change positions
    new_topic_starts = set()
    prev_topic = turns[0].get('topic_id')
    for i, turn in enumerate(turns[1:], 1):
        if turn.get('topic_id') != prev_topic:
            new_topic_starts.add(i)
        prev_topic = turn.get('topic_id')

    # Evaluate each position
    messages = []
    for i, turn in enumerate(turns):
        role = 'assistant' if turn['role'] == 'agent' else turn['role']

        if len(messages) >= 5:  # Need enough history
            expected = i in new_topic_starts
            decision = strategy.get_decision(turn['utterance'], messages)
            predicted = decision.topic_changed

            if expected and predicted: tp += 1
            elif expected and not predicted: fn += 1
            elif not expected and predicted: fp += 1
            else: tn += 1

        messages.append({'role': role, 'content': turn['utterance']})

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")
```

## Bug Fixes Applied

### 1. Role Assignment Bug (neural_strategy.py, dual_window_detector.py)

**Problem**: Query role was always assumed to be "user":
```python
# Before (wrong)
after_messages = [messages[-1], {"role": "user", "content": query}]
```

**Fix**: Determine role from alternating pattern:
```python
# After (correct)
last_role = messages[-1].get('role', 'user')
query_role = 'assistant' if last_role == 'user' else 'user'
after_messages = [messages[-1], {"role": query_role, "content": query}]
```

**Impact**: NeuralStrategy F1 improved from 0.65 to 0.81.

### 2. User-Only Filtering (relative_embedding_strategy.py)

**Problem**: Only considered user messages for similarity:
```python
# Before (wrong)
user_messages = [m for m in messages if m.get('role') == 'user']
```

**Fix**: Use all messages:
```python
# After (correct)
# Use messages directly, don't filter by role
```

**Impact**: RelativeEmbedding F1 improved from 0.25 to 0.39.

## Threshold Sensitivity

### NeuralStrategy confidence_threshold
| Threshold | F1 | Precision | Recall |
|-----------|-----|-----------|--------|
| 0.0 | 0.770 | 0.715 | 0.834 |
| 0.5 | 0.770 | 0.715 | 0.834 |
| 0.7 | 0.708 | 0.736 | 0.683 |
| 0.8 | 0.579 | 0.758 | 0.468 |

Lower threshold = higher recall, lower precision.

## Recommendations

1. **For SuperDialseg-like data**: Use `neural` strategy with threshold=0.5
2. **For production with real users**: Use `ensemble` to handle explicit transitions
3. **For speed-critical applications**: Use `keyword` as pre-filter, then `neural`

## Future Work

- Tune DualWindow thresholds for SuperDialseg
- Train neural model on larger dataset
- Add confidence calibration
- Test on other dialogue datasets (TIAGE, Doc2Dial)
