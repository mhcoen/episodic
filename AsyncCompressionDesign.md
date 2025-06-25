# Async Background Compression Design

## Overview
An intelligent background compression system that uses topic detection to automatically compress conversation segments at natural boundaries.

## Key Components

### 1. Topic-Triggered Compression
- Leverages existing `detect_and_extract_topic_from_response()` function
- When a topic change is detected, the previous topic segment is queued for compression
- Natural conversation boundaries create more coherent compressed summaries

### 2. Background Worker Architecture
```python
# Compression Queue
compression_queue = Queue()

# Background worker thread
def compression_worker():
    while True:
        job = compression_queue.get()
        if job is None:  # Shutdown signal
            break
        
        # Process compression job
        compress_topic_segment(
            start_node_id=job['start_node_id'],
            end_node_id=job['end_node_id'],
            topic_name=job['topic_name']
        )
```

### 3. Integration Points

#### In `store_topic()` function:
```python
def store_topic(topic_name, start_node_id, end_node_id, confidence):
    # Existing topic storage code...
    
    # Queue previous topic for compression if exists
    previous_topics = get_recent_topics(limit=1)
    if previous_topics and config.get('auto_compress_topics', True):
        prev_topic = previous_topics[0]
        compression_queue.put({
            'start_node_id': prev_topic['start_node_id'],
            'end_node_id': prev_topic['end_node_id'],
            'topic_name': prev_topic['topic_name']
        })
```

### 4. Compression Strategy

#### Topic-Aware Compression Prompt
```python
def compress_topic_segment(start_node_id, end_node_id, topic_name):
    nodes = get_nodes_between(start_node_id, end_node_id)
    
    prompt = f"""Compress this conversation about '{topic_name}' into a concise summary.
    Preserve key insights, decisions, and conclusions.
    
    Conversation:
    {format_nodes(nodes)}
    
    Summary:"""
    
    summary = query_llm(prompt, model=compression_model)
    
    # Create compressed node with topic metadata
    compressed_id = insert_node(
        content=f"[Compressed: {topic_name}]\n{summary}",
        parent_id=end_node_id,
        role="system"
    )
    
    store_compression(...)
```

### 5. Configuration Options

```python
# episodic/config.py additions
DEFAULT_CONFIG = {
    'auto_compress_topics': True,
    'compression_min_nodes': 5,  # Min nodes before compression
    'compression_max_age_hours': 24,  # Compress topics older than X hours
    'compression_model': 'ollama/llama3',  # Fast model for background work
    'compression_worker_threads': 1
}
```

### 6. User Controls

- `/set auto_compress_topics true/false` - Enable/disable auto compression
- `/compress-queue` - Show pending compression jobs
- `/compress-stats` - Enhanced to show auto vs manual compressions

## Implementation Phases

1. **Phase 1**: Basic queue and worker thread
2. **Phase 2**: Integration with topic detection
3. **Phase 3**: Smart compression strategies (topic-aware prompts)
4. **Phase 4**: Advanced features (age-based triggers, parallel workers)

## Benefits

1. **Natural boundaries** - Topics provide semantic compression points
2. **Non-blocking** - Conversation continues while compression happens
3. **Context-aware** - Topic names improve summary quality
4. **Automatic** - No manual intervention needed
5. **Resource efficient** - Uses cheap/fast models in background

## Considerations

1. **Thread safety** - Database operations must be thread-safe
2. **Queue persistence** - Consider saving queue state for restarts
3. **Error handling** - Failed compressions shouldn't crash worker
4. **User notification** - Optional notifications when compression completes