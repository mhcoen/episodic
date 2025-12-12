# Topic Detection Architecture: Real-time vs Retroactive

**Date**: 2024-12-11
**Status**: Design analysis for integration

## Overview

Episodic has two distinct topic detection systems that currently operate independently:

1. **Real-time Detection** - Strategies that make instant decisions as messages arrive
2. **Retroactive Analysis** - Batch processing that analyzes complete conversations

This document analyzes both systems and proposes integration approaches.

---

## System 1: Real-time Detection (Strategies)

### Location
- `episodic/topics/strategies/` - Strategy implementations
- `episodic/topics/strategy.py` - Abstract base class
- `episodic/topics/strategy_registry.py` - Factory/registry

### Interface
```python
class TopicStrategy(ABC):
    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision
```

### Implementations

| Strategy | F1 (SuperDialseg) | Latency | Description |
|----------|-------------------|---------|-------------|
| NeuralStrategy | 0.806 | 13ms | Fine-tuned DistilBERT on (4,2) window |
| EnsembleStrategy | 0.804 | 10ms | Keyword + neural combination |
| RelativeEmbeddingStrategy | 0.393 | 13ms | Self-calibrating similarity |
| DualWindowStrategy (adaptive) | 0.271 | 58ms | Adaptive z-score thresholds |
| KeywordStrategy | 0.107 | 0.1ms | Explicit transition phrases |
| DualWindowStrategy (fixed) | 0.019 | 58ms | Fixed 0.15 threshold |

### Characteristics
- **Latency**: 10-60ms per decision (must be fast)
- **Context**: Limited window (4-10 messages)
- **Output**: Immediate `topic_changed` boolean + confidence score
- **Errors**: Can miss boundaries (FN) or hallucinate boundaries (FP)
- **Advantage**: Instant feedback during conversation

### Current Usage
Called from `episodic/topic_management.py` after each user message to decide whether to create a new topic.

---

## System 2: Retroactive Analysis

### Location
- `episodic/topics/reanalyze.py` - Main implementation
- `scripts/retroactive_topic_creation.py` - Drift threshold approach
- `scripts/retroactive_topic_creation_v2.py` - (4,2) window approach

### Algorithm: Hierarchical Agglomerative Clustering

```
1. Start: Each message is its own segment
2. Compute centroid embeddings for each segment
3. Find most similar adjacent pair
4. Merge them into one segment
5. Repeat until stopping criterion met
6. Use elbow detection to find optimal stopping point
```

Key constraint: **Contiguity** - only adjacent segments can merge, preserving temporal order.

### Stopping Criterion: Elbow Detection

```python
def find_elbow(merge_history):
    # Calculate drops in similarity between consecutive merges
    drops = []
    for i in range(1, len(merge_history)):
        drop = merge_history[i-1].similarity - merge_history[i].similarity
        drops.append((i, drop))

    # Biggest drop = natural boundary between topics
    max_drop_idx = max(drops, key=lambda x: x[1])[0]
    return merge_history[max_drop_idx - 1].segments
```

### Characteristics
- **Latency**: Seconds to minutes (entire conversation)
- **Context**: Complete conversation history
- **Output**: Optimal segment boundaries
- **Errors**: May over/under-segment; no real-time feedback
- **Advantage**: Global view enables better decisions

### Current Usage
Manual trigger via `/topics reanalyze [apply]` command.

---

## The Integration Problem

Currently these systems are disconnected:

```
Conversation Flow:
    Message 1 → Real-time: "same topic"     → Topic A
    Message 2 → Real-time: "same topic"     → Topic A
    Message 3 → Real-time: "NEW TOPIC!"     → Topic B created
    Message 4 → Real-time: "same topic"     → Topic B
    ...

Later:
    User: /topics reanalyze apply
    → Hierarchical clustering runs
    → Replaces ALL topics with new segmentation
    → Real-time decisions discarded
```

**Problems:**
1. No reconciliation between the two views
2. Real-time decisions may be wrong but aren't corrected
3. Retroactive analysis is manual, not automatic
4. No feedback loop to improve real-time from retroactive

---

## Proposed Integration Approaches

### Approach 1: Sequential (Current, Minimal Changes)

```
Message → Real-time decision → Store provisional topic
                                    ↓
                    User runs `/topics reanalyze apply`
                                    ↓
                           Replaces all topics
```

**Pros:** Simple, already implemented
**Cons:** Manual, loses real-time decisions entirely

### Approach 2: Confidence-based Hybrid

```
Message → Real-time decision (confidence=C)
           ↓
        if C < threshold:
           Mark topic boundary as "uncertain"
           ↓
        Periodically run hierarchical clustering
        on uncertain segments only
           ↓
        Update boundaries where clustering disagrees
```

**Implementation:**
```python
class TopicDecision:
    topic_changed: bool
    confidence: Confidence  # HIGH, MEDIUM, LOW, UNCERTAIN
    confidence_score: float  # 0.0 - 1.0

# In topic_management.py
def process_message(query, messages):
    decision = strategy.get_decision(query, messages)

    if decision.topic_changed:
        if decision.confidence_score < 0.6:
            create_topic(provisional=True)  # Mark for review
        else:
            create_topic(provisional=False)  # High confidence
```

**Pros:** Leverages confidence signals, targeted refinement
**Cons:** Requires confidence calibration, adds complexity

### Approach 3: Progressive Refinement

```
Message → Real-time creates provisional boundary
           ↓
        After N messages (e.g., 10), trigger local review
           ↓
        Run hierarchical clustering on recent segment
           ↓
        if clustering agrees with real-time:
            Mark boundary as "confirmed"
        else:
            Adjust boundary or merge segments
```

**Implementation:**
```python
def maybe_refine_recent_topics():
    recent_topics = get_topics_since(messages_ago=20)

    if any(t.status == 'provisional' for t in recent_topics):
        # Get messages spanning recent topics
        messages = get_messages_for_topics(recent_topics)

        # Run local clustering
        refined_segments = hierarchical_segment(messages, use_elbow=True)

        # Reconcile
        reconcile_topics(recent_topics, refined_segments)
```

**Pros:** Automatic refinement, preserves real-time responsiveness
**Cons:** Complex reconciliation logic, potential for flickering

### Approach 4: Feedback Loop (Learning)

```
Retroactive analysis identifies "correct" boundaries
           ↓
        Compare to real-time decisions at same positions
           ↓
        Log (real-time prediction, retroactive correction)
           ↓
        Use to tune real-time parameters:
        - Threshold adjustments
        - Confidence calibration
        - Feature weights
```

**Implementation:**
```python
def analyze_disagreements():
    """Find where real-time and retroactive disagree."""
    rt_boundaries = get_realtime_boundaries()
    retro_boundaries = get_retroactive_boundaries()

    false_positives = rt_boundaries - retro_boundaries
    false_negatives = retro_boundaries - rt_boundaries

    return {
        'fp': false_positives,  # Real-time was wrong to create boundary
        'fn': false_negatives,  # Real-time missed a boundary
        'signals_at_fp': extract_signals(false_positives),
        'signals_at_fn': extract_signals(false_negatives),
    }

def tune_threshold(disagreements):
    """Adjust threshold based on error analysis."""
    # If too many FP, raise threshold
    # If too many FN, lower threshold
    pass
```

**Pros:** System improves over time, data-driven tuning
**Cons:** Requires labeled data, complex to implement correctly

---

## Recommended Path Forward

### Phase 1: Add Confidence Tracking (Low effort)
- Ensure all strategies return meaningful confidence scores
- Store confidence with topic boundaries in database
- Display confidence in `/topics list`

### Phase 2: Automatic Background Refinement (Medium effort)
- Run retroactive analysis periodically (e.g., every 20 messages)
- Only on segments containing low-confidence boundaries
- Log but don't auto-apply corrections initially

### Phase 3: Reconciliation UI (Medium effort)
- Show disagreements: "Real-time said X, clustering suggests Y"
- Let user approve corrections
- Build labeled dataset from user decisions

### Phase 4: Parameter Tuning (Higher effort)
- Use labeled data to tune real-time thresholds
- Implement adaptive threshold adjustment
- Consider retraining neural model on local data

---

## Database Schema Considerations

Current `topics` table would need:

```sql
ALTER TABLE topics ADD COLUMN confidence_score REAL;
ALTER TABLE topics ADD COLUMN source TEXT;  -- 'realtime', 'retroactive', 'user'
ALTER TABLE topics ADD COLUMN provisional BOOLEAN DEFAULT FALSE;
ALTER TABLE topics ADD COLUMN reviewed_at TIMESTAMP;
```

---

## Key Architectural Question

**Should retroactive analysis correct real-time decisions automatically, or just provide an alternative view?**

Arguments for automatic correction:
- Better accuracy over time
- User doesn't need to manually review
- Consistent topic structure

Arguments against automatic correction:
- May confuse users if topics change
- Real-time decisions had context user saw
- Harder to debug/understand system behavior

**Recommended:** Start with manual review (Phase 3), evolve to automatic with high-confidence threshold.

---

## Related Files

- `docs/technical/STRATEGY_BENCHMARK_RESULTS.md` - Strategy comparison data
- `scripts/benchmark_strategies.py` - Benchmark harness
- `episodic/topics/strategy.py` - Strategy interface
- `episodic/topics/reanalyze.py` - Retroactive implementation
