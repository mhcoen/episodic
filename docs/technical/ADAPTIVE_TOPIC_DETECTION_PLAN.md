# Adaptive Topic Detection Implementation Plan

## Vision
Enable Episodic to dynamically detect when users return to previous topics and offer intelligent context management, ultimately supporting non-linear DAG-based conversations.

## Design Principles

1. **Graceful Degradation**: Every feature works even with 0% detection accuracy
2. **User Control**: Nothing automatic without high confidence and user consent  
3. **Progressive Enhancement**: Start simple, add complexity only when proven
4. **Transparent Operation**: Users understand what's happening and why
5. **Learning System**: Improves from user feedback and choices

## Architecture Overview

### Core Components

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Topic Detector     │────▶│  Topic Matcher   │────▶│  Action Handler  │
│  (Current system)   │     │  (New)           │     │  (New)           │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
         │                           │                         │
         ▼                           ▼                         ▼
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Topic Boundaries   │     │  Topic Embeddings│     │  User Interface  │
│  (DB)               │     │  (New DB table)  │     │  Suggestions     │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
```

### New Database Tables

```sql
-- Store topic embeddings for fast similarity matching
CREATE TABLE topic_embeddings (
    topic_id INTEGER PRIMARY KEY,
    topic_name TEXT,
    embedding BLOB,  -- Serialized numpy array
    centroid_method TEXT,  -- 'mean', 'weighted', etc.
    message_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES topics(id)
);

-- Track similarity detections and user responses
CREATE TABLE topic_similarity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_node_id TEXT,
    similar_topic_id INTEGER,
    confidence REAL,
    action_taken TEXT,  -- 'suggested', 'accepted', 'rejected', 'auto_applied'
    user_feedback TEXT,  -- 'correct', 'incorrect', null
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(message_node_id) REFERENCES nodes(id),
    FOREIGN KEY(similar_topic_id) REFERENCES topics(id)
);

-- Store user preferences
CREATE TABLE topic_preferences (
    user_id TEXT PRIMARY KEY DEFAULT 'default',
    auto_include_threshold REAL DEFAULT 0.85,
    show_suggestions BOOLEAN DEFAULT TRUE,
    enable_branching BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Build infrastructure without changing user experience

1. **Topic Embedding System**
   ```python
   class TopicEmbeddingManager:
       def create_topic_embedding(self, topic_id: int) -> np.ndarray:
           """Generate embedding for entire topic"""
           messages = get_topic_messages(topic_id)
           embeddings = [embed(msg.content) for msg in messages]
           return np.mean(embeddings, axis=0)  # Start with simple mean
       
       def update_embedding(self, topic_id: int, new_message: Message):
           """Incrementally update topic embedding"""
           # Efficient update without recomputing everything
   ```

2. **Similarity Matcher**
   ```python
   class TopicSimilarityMatcher:
       def __init__(self, threshold: float = 0.7):
           self.threshold = threshold
           self.embeddings = load_topic_embeddings()
       
       def find_similar_topics(self, message: str) -> List[SimilarTopic]:
           message_embedding = embed(message)
           similarities = []
           
           for topic_id, topic_embedding in self.embeddings.items():
               score = cosine_similarity(message_embedding, topic_embedding)
               if score > self.threshold:
                   similarities.append(SimilarTopic(topic_id, score))
           
           return sorted(similarities, key=lambda x: x.score, reverse=True)
   ```

3. **Logging Infrastructure**
   - Log all detections without taking action
   - Track confidence scores and thresholds
   - Build dataset for accuracy analysis

**Success Metrics**:
- Topic embeddings generated for all topics
- Similarity matching runs in <100ms
- No user-visible changes

### Phase 2: Passive Suggestions (Week 3-4)
**Goal**: Show non-intrusive suggestions when topic similarity detected

1. **UI Integration**
   ```python
   # In conversation.py
   def handle_message_with_similarity(self, message: str):
       # Normal response generation
       response = self.generate_response(message)
       
       # Check for similar topics (async/background)
       similar_topics = self.topic_matcher.find_similar_topics(message)
       
       if similar_topics and similar_topics[0].confidence > 0.7:
           response.add_metadata({
               'similar_topic': {
                   'name': similar_topics[0].name,
                   'confidence': similar_topics[0].confidence,
                   'action': 'view_context'
               }
           })
       
       return response
   ```

2. **User Feedback Collection**
   ```
   Assistant: [response text]
   
   💭 This seems related to your earlier "Italian Cooking" discussion (78% match)
      [View Context] [Include Context] [Not Related ✗]
   ```

3. **Commands**
   - `/topic-suggestions on/off` - Toggle suggestions
   - `/topic-stats` - Show accuracy statistics
   - `/calibrate-topics` - Adjust thresholds based on feedback

**Success Metrics**:
- 80% of suggestions marked as "correct" by users
- <5% of users disable suggestions
- Positive user feedback

### Phase 3: Smart Context Inclusion (Week 5-6)
**Goal**: Automatically include relevant context at high confidence

1. **Context Injection**
   ```python
   def build_context_with_similarity(self, current_messages, similar_topic=None):
       if similar_topic and similar_topic.confidence > 0.85:
           # Insert relevant messages from similar topic
           context = []
           context.append(SystemMessage(
               f"[Including relevant context from earlier '{similar_topic.name}' discussion]"
           ))
           context.extend(get_topic_highlights(similar_topic.id, max_messages=3))
           context.append(SystemMessage("[Current conversation continues]"))
           context.extend(current_messages[-5:])  # Recent messages
           return context
       else:
           return current_messages[-10:]  # Normal context window
   ```

2. **Gradual Rollout**
   - Start with opt-in users
   - Monitor response quality
   - Track context window efficiency

3. **User Controls**
   ```
   /context-mode [off|manual|auto]
   /context-threshold 0.85
   /exclude-topic <topic-name>
   ```

**Success Metrics**:
- Response relevance improves by 15%
- Context window usage more efficient
- No increase in confusion/complaints

### Phase 4: Topic Return Detection (Week 7-8)
**Goal**: Detect when users explicitly return to previous topics

1. **Return Patterns**
   ```python
   RETURN_PATTERNS = [
       r"going back to (?:our discussion about |the )?(.+)",
       r"returning to (?:the topic of )?(.+)",
       r"about that (.+) we discussed",
       r"remember when we talked about (.+)",
   ]
   
   def detect_topic_return(self, message: str) -> Optional[TopicReturn]:
       # Check explicit patterns first
       for pattern in RETURN_PATTERNS:
           if match := re.search(pattern, message.lower()):
               topic_hint = match.group(1)
               # Find matching topic
               return self.find_topic_by_hint(topic_hint)
       
       # Fall back to embedding similarity
       return self.check_embedding_similarity(message)
   ```

2. **Confirmation UI**
   ```
   User: "Going back to that pasta recipe we discussed..."
   
   🔄 Returning to "Italian Cooking" topic?
      [Continue from there] [Just include context] [New discussion]
   ```

3. **Soft Branching**
   - Add "via" references without changing DAG structure
   - Track topic transitions in new table
   - Prepare for true branching

**Success Metrics**:
- 90% accuracy on explicit returns
- Users choose "Continue" >50% of the time
- Reduced repetition in conversations

### Phase 5: Progressive DAG Branching (Week 9-10)
**Goal**: Enable true non-linear conversations for power users

1. **Branch Points**
   ```python
   class BranchPoint:
       def __init__(self, from_node_id: str, to_topic_id: int, confidence: float):
           self.from_node = from_node_id
           self.to_topic = to_topic_id
           self.confidence = confidence
           self.user_confirmed = False
       
       def create_branch(self):
           if self.confidence > 0.9 or self.user_confirmed:
               # Create new DAG branch
               new_parent = get_topic_last_node(self.to_topic)
               update_node_parent(self.from_node, new_parent)
   ```

2. **Visualization Support**
   - Show branch points in graph
   - Different colors for topic returns
   - Timeline vs. topic view

3. **Safety Features**
   ```
   /unbranch - Linearize current branch
   /branch-history - Show all branches
   /merge-topics - Combine related topics
   ```

**Success Metrics**:
- Power users create 2+ branches per conversation
- No increase in user confusion
- Positive feedback on non-linear flow

## Failure Modes and Mitigations

### 1. False Positive Topic Matches
**Problem**: System suggests unrelated topics
**Mitigation**: 
- High confidence thresholds
- User feedback lowers future confidence
- "Not Related" button trains system

### 2. Missed Topic Returns  
**Problem**: User returns to topic but system misses it
**Mitigation**:
- Explicit commands: `/return-to <topic>`
- Manual topic selection UI
- System still works normally

### 3. Context Overload
**Problem**: Too much context included
**Mitigation**:
- Limit context to 3-5 key messages
- Summarize long topics
- User can adjust context window

### 4. Confusing Branches
**Problem**: Non-linear flow confuses users
**Mitigation**:
- Branching off by default
- Clear visual indicators
- Always can linearize

## Technical Considerations

### Performance
- Cache topic embeddings
- Update embeddings incrementally
- Background processing for non-critical paths
- Similarity check <100ms target

### Accuracy Improvements
- Fine-tune embeddings on conversation data
- Learn from user feedback
- A/B test different thresholds
- Model per-user preferences

### Integration Points
- Reuse existing embedding infrastructure
- Extend current topic detection
- Build on sliding window work
- Compatible with compression system

## Success Criteria

### Phase 1-2 (Months 1-2)
- [ ] 75% suggestion accuracy
- [ ] <100ms performance impact
- [ ] Positive user feedback

### Phase 3-4 (Months 3-4)
- [ ] 15% improvement in response relevance
- [ ] 50% of topic returns detected
- [ ] No increase in user errors

### Phase 5 (Month 5)
- [ ] 10% of power users using branches
- [ ] Clean visualization of non-linear flows
- [ ] Feature used in 25% of long conversations

## Configuration

```python
# Default settings (episodic/config.py)
ADAPTIVE_TOPIC_DEFAULTS = {
    "enable_similarity_detection": True,
    "show_suggestions": True,
    "auto_context_threshold": 0.85,
    "suggestion_threshold": 0.70,
    "enable_branching": False,
    "max_context_messages": 5,
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_metric": "cosine"
}
```

## Next Steps

1. Review and refine plan with team
2. Create feature flag infrastructure
3. Build Phase 1 foundation
4. Set up metrics collection
5. Begin user testing with volunteers

---

This plan prioritizes gradual rollout, user control, and graceful failure while building toward the vision of truly non-linear, adaptive conversations.