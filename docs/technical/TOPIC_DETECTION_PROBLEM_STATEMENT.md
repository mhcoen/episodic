# Topic Detection in Conversations: Problem Statement

**Date**: 2024-12-11
**Purpose**: Seeking input on alternative approaches

---

## The Problem

We're building a conversational memory system that needs to detect when a conversation shifts from one topic to another. The goal is to segment conversations into coherent "threads" that can later be retrieved when the user returns to a topic.

**Example scenario:**
```
Messages 1-5:   Discussing favorite books
Messages 6-15:  Work-related discussion
Messages 16-20: "What was that author we talked about?"
```

When the user asks about the author at message 16, we want to retrieve context from messages 1-5 (the books discussion), not just the recent work discussion.

**The detection problem**: Given a new message and recent conversation history, determine if this message represents a topic change.

---

## Constraints

1. **Real-time**: Detection must be fast (<100ms) to run during conversation flow
2. **Streaming context**: Only see messages up to the current point (no future lookahead)
3. **Model-agnostic**: Should work across different embedding models without retuning
4. **No explicit markers**: Users rarely say "changing topics" - transitions are implicit
5. **Variable granularity**: Topics can span 4 messages or 40 messages
6. **Robust to noise**: Small phrasing changes shouldn't flip the decision

---

## What We've Tried

### Approach 1: Fixed Embedding Similarity Threshold

**Method**: Compute cosine similarity between new message embedding and recent context centroid. If similarity < 0.15, declare topic change.

**Result**: F1 = 0.019 on benchmark

**Why it failed**: Similarities on our test data range 0.5-0.7. The fixed threshold (0.15) never triggers. Threshold would need to be ~0.5, but that varies by embedding model and conversation style.

### Approach 2: Adaptive/Self-Calibrating Thresholds

**Method**: Track running mean and standard deviation of similarities. Detect topic change if similarity drops below `mean - z * std` (z-score based anomaly detection).

**Result**: F1 = 0.271 (14x improvement, still poor)

**Why it's limited**: On our test data, the similarity distribution for topic boundaries (mean=0.520) nearly matches non-boundaries (mean=0.498). The signal just isn't there in raw embedding similarity.

### Approach 3: Neural Classifier

**Method**: Fine-tune DistilBERT on labeled topic segmentation data. Input format: `[messages before] [BOUNDARY?] [messages after]`. Binary classification: boundary or not.

**Result**: F1 = 0.806 (best so far)

**Why it works**: The neural model learns subtle patterns beyond raw similarity - linguistic cues, discourse markers, question-answer patterns, etc.

**Downsides**:
- Requires training data
- Model-specific (trained on DistilBERT)
- 13ms latency (acceptable but not instant)

### Approach 4: Keyword Detection

**Method**: Look for explicit transition phrases ("by the way", "on another note", "changing gears") and domain keyword shifts.

**Result**: F1 = 0.107 on benchmark

**Why it's limited**: Works great when users are explicit, but most topic changes are implicit. Only 1 keyword trigger in 100 test dialogues.

### Approach 5: Ensemble Combinations

**Method**: Combine multiple weak classifiers hoping for complementary signal.

**Attempts**:
- Simple voting (any classifier says boundary)
- Weighted voting (by individual F1 scores)
- Corroboration (neural AND any other)
- Stacking (logistic regression on predictions + confidences)

**Result**: No combination beat neural alone (F1 = 0.806)

**Why it failed**:
- Neural already achieves 92.4% recall
- Weak classifiers make correlated errors with each other
- Adding weak signals just adds false positives
- Stacking learned to just use neural's confidence (coefficient 4.68 vs <1 for others)

### Approach 6: Retroactive Hierarchical Clustering

**Method**: After conversation completes, run hierarchical agglomerative clustering with contiguity constraint. Use elbow detection to find optimal segment count.

**Result**: Works well for batch analysis, but not real-time

**How it works**:
1. Start with each message as its own segment
2. Compute centroid embeddings
3. Merge most similar adjacent pair
4. Repeat, tracking similarity at each merge
5. Find "elbow" (biggest similarity drop) = natural boundary

**Limitation**: Requires seeing full conversation. Can't use for real-time detection.

### Approach 7: CUSUM Drift Detector

**Method**: Cumulative sum (CUSUM) change-point detection. Accumulates drift evidence over time: S_t = max(0, S_{t-1} + drift_t - baseline). Triggers boundary when S_t > threshold.

**Result**: F1 = 0.377-0.391 (depending on threshold)

**Why it's limited**: While theoretically elegant for detecting gradual shifts, the accumulated signal doesn't discriminate well on our test data. Better suited for clearly drifting time series than conversation semantics.

### Approach 8: Delta-Representation

**Method**: Model topic transitions as *changes* in embedding space, not states. Compute Δh_t = h_t - h_{t-k} and detect when delta magnitude exceeds threshold.

**Result**: F1 = 0.410 (with near-perfect 99.5% recall but only 25.8% precision)

**Why it failed**: The delta signal triggers on almost everything. The magnitude threshold isn't discriminating - nearly every message shows "change" from the previous window.

### Approach 9: Speech-Act Pattern Detection

**Method**: Use functional linguistic signals - explicit transition phrases ("by the way", "new question"), Q→A→Q patterns, discourse markers.

**Result**: F1 = 0.673 (second best after neural)

**Why it's interesting**:
- Zero computational cost (pure regex matching)
- High precision for explicit transitions
- But limited recall - most boundaries are implicit

**Ensemble attempt**: Combining with neural showed 96% overlap in catches. Speech-act provides only 6 unique true positives while adding 34 false positives. No F1 improvement over neural alone.

### Approach 10: Time-Aware Detection (Production Only)

**Method**: Incorporate message timestamps. Long gaps (>30 min) strongly suggest new topics. Combine time signal with drift using Bayesian-style weighting.

**Result**: Cannot benchmark (SuperDialseg has no timestamps)

**Implementation**: Available as `time_aware` strategy for production use where timestamps exist. Not applicable to benchmark dataset.

---

## Current Approach

**Production strategy**: Ensemble with priority
1. **Keyword check first** - If explicit transition phrase detected, immediately return "topic change" (high confidence, instant)
2. **Neural classifier** - For everything else, use fine-tuned DistilBERT with confidence threshold

**Rationale**: Keywords catch the easy cases with 100% precision. Neural handles subtle implicit transitions.

**Fallback**: Retroactive reanalysis available via `/topics reanalyze` command for batch correction.

---

## Benchmark Dataset

**SuperDialseg**: Customer service dialogues with human-annotated topic boundaries
- 1322 dialogues in test set
- ~17K messages, ~4K topic boundaries
- Ground truth: topic_id changes between consecutive turns

**Caveat**: This dataset may not represent all conversation types. Customer service has specific patterns (greeting → issue → resolution → closing).

---

## Current Results Summary

| Strategy | F1 | Precision | Recall | Latency |
|----------|-----|-----------|--------|---------|
| Neural (threshold=0.5) | **0.806** | 0.715 | 0.924 | 13ms |
| Ensemble (keyword+neural) | 0.804 | 0.712 | 0.924 | 10ms |
| Speech-Act | 0.673 | 0.630 | 0.722 | <1ms |
| Delta-Representation | 0.410 | 0.258 | 0.995 | 35ms |
| Relative Embedding | 0.393 | 0.352 | 0.444 | 13ms |
| CUSUM (best config) | 0.391 | 0.283 | 0.636 | 33ms |
| Dual Window (adaptive) | 0.271 | 0.299 | 0.247 | 58ms |
| Keyword | 0.107 | 0.183 | 0.076 | 0.1ms |
| Dual Window (fixed) | 0.019 | 0.182 | 0.010 | 58ms |

---

## Error Analysis

**Where neural fails** (88 errors out of 766 positions):
- When other classifiers are wrong, neural is only also wrong 7-14% of the time
- Neural makes unique errors - not correlated with weak classifiers

**Oracle ceiling**: 95.5% recall if we could perfectly combine all classifiers
- 189/198 boundaries caught by at least one strategy
- 9 boundaries missed by ALL strategies

**Unique catches by strategy**:
- Neural: 88 boundaries only it detects
- Relative embedding: 4
- Keyword: 1
- Dual window: 0

---

## Open Questions

1. **Is embedding similarity fundamentally limited?** The similar distributions for boundaries vs non-boundaries suggest the signal may not be there.

2. **What features would a better classifier use?** The neural model works but is opaque. What is it learning?

3. **Should we accept different strategies for different use cases?**
   - Real-time: Fast, lower accuracy
   - Batch: Slow, higher accuracy with full context

4. **How do we handle the granularity problem?** Is "discussing Python" one topic, or are "Python syntax" and "Python libraries" separate topics?

5. **Is there a hybrid approach** that uses real-time detection as a starting point, then refines with batch analysis?

---

## Code Pointers

- `episodic/topics/strategies/` - Strategy implementations
- `episodic/topics/strategy.py` - Abstract base class
- `episodic/topics/reanalyze.py` - Retroactive clustering
- `scripts/benchmark_strategies.py` - Evaluation harness
- `docs/technical/STRATEGY_BENCHMARK_RESULTS.md` - Detailed results
- `docs/technical/TOPIC_DETECTION_ARCHITECTURE.md` - Architecture overview

---

## What We're Looking For

Alternative approaches we haven't considered:
- Different features or representations
- Different problem formulations
- Relevant literature or existing solutions
- Ways to better leverage the batch/retroactive system
- Ideas for generating better training data
