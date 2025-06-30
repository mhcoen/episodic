# Hybrid Topic Detection System Design

## Executive Summary

This document outlines the design for a hybrid topic detection system that combines:
1. **Embedding-based semantic drift detection** (primary method)
2. **Keyword-based explicit transition detection** (secondary method)
3. **LLM-based detection** (fallback method)
4. **User control commands** (override mechanism)

The system aims to provide more accurate and reliable topic detection while reducing LLM API costs and improving response times.

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   User Message Input                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Topic Detection Pipeline                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. User Command Check (/topic, /new-topic)         │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │ 2. Embedding-based Drift Detection                  │   │
│  │    - Generate embeddings for recent messages        │   │
│  │    - Calculate semantic drift scores                │   │
│  │    - Apply peak detection algorithms                │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │ 3. Keyword Detection (if drift inconclusive)        │   │
│  │    - Check for explicit transition phrases          │   │
│  │    - Analyze domain-specific keywords               │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │ 4. LLM Fallback (if still inconclusive)            │   │
│  │    - Use existing LLM-based detection               │   │
│  │    - Apply with higher confidence threshold         │   │
│  └─────────────────────┬───────────────────────────────┘   │
└────────────────────────┼───────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Topic Change Decision & Actions                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Message Reception**
   - User message arrives
   - Add to message history buffer
   - Check for user commands

2. **Embedding Generation**
   - Generate embeddings for new message
   - Retrieve cached embeddings for recent messages
   - Store new embeddings in cache/database

3. **Drift Calculation**
   - Calculate pairwise drift between consecutive messages
   - Calculate cumulative drift from topic start
   - Apply windowed averaging for noise reduction

4. **Peak Detection**
   - Apply configured peak detection algorithm
   - Consider multiple signals (sudden drift, cumulative drift, keyword presence)
   - Generate confidence score

5. **Decision Making**
   - Combine all signals with weighted scoring
   - Apply threshold based on configuration
   - Execute topic change if threshold exceeded

6. **Post-Processing**
   - Close current topic at last assistant response
   - Create new topic with placeholder name
   - Queue topic name extraction

## Algorithm Details

### 1. Embedding-based Semantic Drift

#### Configuration Options
```python
{
    "embedding_provider": "sentence-transformers",  # or "openai", "huggingface"
    "embedding_model": "all-MiniLM-L6-v2",         # fast, good quality
    "distance_algorithm": "cosine",                 # or "euclidean", "manhattan"
    "drift_threshold": 0.35,                        # topic change threshold
    "window_size": 5,                               # messages to consider
    "peak_detection": {
        "strategy": "adaptive",                     # or "threshold", "statistical"
        "min_prominence": 0.2,                      # minimum peak prominence
        "lookback": 3                               # messages to look back
    }
}
```

#### Algorithm Steps

1. **Embedding Generation**
   ```python
   def generate_embedding(message: str) -> List[float]:
       # Check cache first
       if message_hash in embedding_cache:
           return embedding_cache[message_hash]
       
       # Generate new embedding
       embedding = embedding_provider.embed(message)
       
       # Store in cache and database
       embedding_cache[message_hash] = embedding
       store_embedding_in_db(message_hash, embedding)
       
       return embedding
   ```

2. **Drift Calculation**
   ```python
   def calculate_drift_score(messages: List[Message]) -> float:
       # Get embeddings for recent messages
       embeddings = [generate_embedding(msg.content) for msg in messages]
       
       # Calculate pairwise distances
       distances = []
       for i in range(1, len(embeddings)):
           dist = distance_function(embeddings[i-1], embeddings[i])
           distances.append(dist)
       
       # Apply windowed averaging
       avg_drift = np.mean(distances[-window_size:])
       
       # Calculate cumulative drift from topic start
       if current_topic_start_embedding:
           cumulative_drift = distance_function(
               current_topic_start_embedding,
               embeddings[-1]
           )
       
       # Combine signals
       combined_score = (0.7 * avg_drift + 0.3 * cumulative_drift)
       
       return combined_score
   ```

3. **Peak Detection**
   ```python
   def detect_drift_peak(drift_scores: List[float]) -> bool:
       if peak_strategy == "adaptive":
           # Adaptive threshold based on recent history
           baseline = np.mean(drift_scores[:-1])
           std_dev = np.std(drift_scores[:-1])
           threshold = baseline + (2 * std_dev)
           
           current_drift = drift_scores[-1]
           prominence = (current_drift - baseline) / baseline
           
           return (current_drift > threshold and 
                   prominence > min_prominence)
       
       elif peak_strategy == "threshold":
           return drift_scores[-1] > drift_threshold
   ```

### 2. Keyword-based Detection

#### Configuration
```python
{
    "keyword_detection": {
        "enabled": true,
        "weight": 0.3,  # contribution to final score
        "transition_phrases": [
            "let's talk about",
            "changing the subject",
            "on a different note",
            "switching gears",
            "moving on to",
            "different question"
        ],
        "domain_keywords": {
            "technology": ["programming", "software", "computer", "AI"],
            "science": ["physics", "chemistry", "biology", "research"],
            "cooking": ["recipe", "ingredients", "cooking", "food"],
            # ... more domains
        }
    }
}
```

#### Algorithm
```python
def detect_keyword_transition(current_msg: str, recent_msgs: List[str]) -> float:
    score = 0.0
    
    # Check for explicit transition phrases
    for phrase in transition_phrases:
        if phrase.lower() in current_msg.lower():
            score += 0.8
            break
    
    # Extract domains from recent messages
    recent_domains = extract_domains(recent_msgs)
    current_domains = extract_domains([current_msg])
    
    # Calculate domain shift
    domain_overlap = len(recent_domains & current_domains)
    domain_shift = 1.0 - (domain_overlap / max(len(recent_domains), 1))
    
    score += domain_shift * 0.5
    
    return min(score, 1.0)
```

### 3. Hybrid Scoring

```python
def calculate_hybrid_score(
    embedding_drift: float,
    keyword_score: float,
    message_count: int
) -> Tuple[float, str]:
    """
    Returns (score, method_used)
    """
    # Apply message count threshold
    if message_count < min_messages_before_topic_change:
        return (0.0, "threshold_not_met")
    
    # User command overrides everything
    if has_user_topic_command():
        return (1.0, "user_command")
    
    # High embedding drift is primary signal
    if embedding_drift > 0.6:
        return (embedding_drift, "embedding_high")
    
    # Combine signals for medium drift
    if embedding_drift > 0.3:
        combined = (0.7 * embedding_drift + 0.3 * keyword_score)
        if combined > 0.5:
            return (combined, "hybrid")
    
    # Strong keyword signal can trigger on its own
    if keyword_score > 0.8:
        return (keyword_score, "keyword_strong")
    
    # Fall back to LLM if inconclusive
    if 0.3 <= embedding_drift <= 0.5:
        llm_score = run_llm_detection()
        if llm_score > 0.7:  # Higher threshold for LLM
            return (llm_score, "llm_fallback")
    
    return (max(embedding_drift, keyword_score), "no_change")
```

## Database Schema Updates

### New Tables

```sql
-- Store embeddings for messages
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    message_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- Stored as binary numpy array
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(node_id) REFERENCES nodes(id),
    UNIQUE(message_hash, model_name)
);

-- Index for fast lookups
CREATE INDEX idx_embeddings_hash ON embeddings(message_hash);
CREATE INDEX idx_embeddings_node ON embeddings(node_id);

-- Store drift scores for analysis
CREATE TABLE IF NOT EXISTS drift_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_node_id TEXT NOT NULL,
    to_node_id TEXT NOT NULL,
    drift_score REAL NOT NULL,
    detection_method TEXT,  -- 'embedding', 'keyword', 'hybrid', 'llm'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(from_node_id) REFERENCES nodes(id),
    FOREIGN KEY(to_node_id) REFERENCES nodes(id)
);
```

## Integration Points

### 1. Configuration Integration

Update `episodic/config.py`:
```python
# Add to default configuration
{
    "topic_detection_method": "hybrid",  # or "llm_only", "embedding_only"
    "embedding_config": {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        "cache_embeddings": true
    },
    "drift_config": {
        "algorithm": "cosine",
        "threshold": 0.35,
        "window_size": 5
    },
    "keyword_config": {
        "enabled": true,
        "weight": 0.3
    }
}
```

### 2. Topic Manager Integration

Modify `episodic/topics.py`:
```python
class TopicManager:
    def __init__(self):
        self.prompt_manager = PromptManager()
        # Initialize hybrid detection components
        self.drift_detector = ConversationalDrift(
            embedding_provider=config.get("embedding_config.provider"),
            embedding_model=config.get("embedding_config.model"),
            distance_algorithm=config.get("drift_config.algorithm")
        )
        self.keyword_detector = KeywordDetector(
            config.get("keyword_config")
        )
    
    def detect_topic_change_hybrid(
        self,
        recent_messages: List[Dict[str, Any]],
        new_message: str,
        current_topic: Optional[Tuple[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """New hybrid detection method"""
        # Implementation here
```

### 3. CLI Integration

Add new commands:
```python
# In episodic/cli.py
@app.command()
def topic_detection_config(
    method: str = typer.Option(None, help="Detection method: hybrid, llm_only, embedding_only"),
    threshold: float = typer.Option(None, help="Drift threshold (0.0-1.0)"),
    provider: str = typer.Option(None, help="Embedding provider")
):
    """Configure topic detection settings"""
    # Implementation
```

## Example Scenarios

### Scenario 1: Gradual Topic Drift
```
User: Tell me about machine learning
Assistant: [ML explanation]
User: What about neural networks specifically?  <- Drift: 0.15 (same domain)
Assistant: [NN explanation]
User: How do transformers work?                <- Drift: 0.18 (still ML)
Assistant: [Transformer explanation]
User: Speaking of transformation, I'm renovating my kitchen  <- Drift: 0.72 (domain shift)
[TOPIC CHANGE DETECTED - Method: embedding_high]
```

### Scenario 2: Explicit Transition
```
User: Can you explain Python decorators?
Assistant: [Decorator explanation]
User: Let's switch gears - what's a good pasta recipe?  <- Keyword detected + drift: 0.65
[TOPIC CHANGE DETECTED - Method: hybrid]
```

### Scenario 3: Ambiguous Case
```
User: How do I debug Python code?
Assistant: [Debugging tips]
User: What about debugging life problems?  <- Drift: 0.42 (medium)
[LLM FALLBACK - Checking with model]
[TOPIC CHANGE DETECTED - Method: llm_fallback]
```

### Scenario 4: User Override
```
User: Tell me more about quantum physics
Assistant: [Physics explanation]
User: /new-topic Actually, let's talk about cooking
[TOPIC CHANGE DETECTED - Method: user_command]
```

## Performance Considerations

### Embedding Caching
- Store embeddings in database for reuse
- Use message hash as cache key
- Implement LRU cache in memory (limit: 1000 embeddings)
- Batch embedding generation when possible

### Optimization Strategies
1. **Lazy Loading**: Only load ML models when hybrid detection is enabled
2. **Background Processing**: Generate embeddings asynchronously
3. **Batch Operations**: Process multiple messages together
4. **Model Selection**: 
   - Fast: all-MiniLM-L6-v2 (22M params, 80MB)
   - Balanced: all-mpnet-base-v2 (110M params, 420MB)
   - High Quality: all-roberta-large-v1 (355M params, 1.4GB)

### Resource Usage
- Memory: ~100-500MB depending on model
- CPU: Embedding generation takes ~50-200ms per message
- Storage: ~2KB per embedding (384-768 dimensions)

## Migration Path

### Phase 1: Infrastructure (Week 1)
1. Add database tables for embeddings
2. Implement embedding generation and caching
3. Add drift calculation algorithms
4. Create configuration options

### Phase 2: Integration (Week 2)
1. Integrate with TopicManager
2. Add hybrid detection method
3. Implement fallback logic
4. Update CLI commands

### Phase 3: Testing & Tuning (Week 3)
1. Run on test conversations
2. Tune thresholds and weights
3. Compare with LLM-only approach
4. Optimize performance

### Phase 4: Rollout (Week 4)
1. Enable as opt-in feature
2. Collect metrics and feedback
3. Fine-tune based on usage
4. Make default if successful

## Monitoring & Metrics

### Key Metrics to Track
1. **Accuracy Metrics**
   - Topic changes detected vs. ground truth
   - False positive rate
   - False negative rate

2. **Performance Metrics**
   - Embedding generation time
   - Total detection time
   - Cache hit rate

3. **Usage Metrics**
   - Detection method distribution
   - LLM fallback frequency
   - User override frequency

### Debugging Tools
```bash
# Show drift scores for current conversation
episodic> /show-drift

# Analyze topic detection for specific range
episodic> /analyze-topics 1-20

# Export embeddings for visualization
episodic> /export-embeddings conversation.json
```

## Future Enhancements

1. **Multi-modal Embeddings**: Support for code, images, tables
2. **Personalized Thresholds**: Learn user-specific topic preferences
3. **Topic Prediction**: Suggest potential topics based on conversation flow
4. **Embedding Fine-tuning**: Train custom embeddings on conversation data
5. **Real-time Visualization**: Show drift scores in UI as conversation progresses

## Conclusion

This hybrid approach combines the best of multiple methods:
- **Accuracy**: Embeddings capture semantic meaning better than keywords
- **Speed**: No LLM calls for most detections  
- **Cost**: Reduced API usage by 80-90%
- **Control**: Users can override when needed
- **Reliability**: Multiple fallback mechanisms

The system is designed to be modular, configurable, and extensible, allowing for easy experimentation and improvement over time.