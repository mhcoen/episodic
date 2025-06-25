# Conversational Drift Detection and DAG Restructuring Design

## Overview

Design discussion for implementing semantic drift detection and automatic DAG restructuring in the episodic conversational system. The goal is to enable natural conversation flow with a fixed-length attention window by intelligently navigating the conversation DAG based on topic relevance rather than just recency.

## Core Concept

Instead of using a simple sliding window of the last N messages, the system will:
- Dynamically select relevant conversation history based on semantic similarity
- Automatically restructure the DAG to reflect topical relationships
- Maintain natural conversation flow across different conversation patterns

## Key Design Elements

### 1. Semantic Depth (semdepth)
- Configurable parameter indicating how many ancestor nodes to include when creating embeddings
- Example: `semdepth=3` would embed current node + 2 most recent ancestors
- Provides local conversational context for semantic analysis

### 2. Node Embeddings
- Each node embedding includes the node content plus its semdepth ancestors
- Question: Should embeddings include both query and response, or just query?
- Consideration: Node might be moved or re-parented before response is generated

### 3. Modular Components
- **Embedding Library**: Customizable (OpenAI, sentence-transformers, etc.)
- **Distance Functions**: Various similarity measures (cosine, euclidean, hybrid search, re-ranking)
- **Branch Summarization**: Pluggable strategies for different conversation types

## Implementation Phases

### Phase 1: Measurement and Observation
- Real-time drift calculation and display during CLI interaction
- Print semantic drift scores above each query/response pair
- Format: `[Semantic drift: 0.85 from previous context]`
- Goal: Understand drift patterns across different conversation types

### Phase 2: Branch Summarization
- Implement modular branch summarization strategies
- Compare drift against both local context (semdepth) and branch summary
- Display both local and global drift scores

### Phase 3: Automatic Restructuring
- Implement automated DAG restructuring based on drift thresholds
- Walk up DAG comparing drift when high drift detected
- Create new parent relationships or weak links as appropriate

## Conversation Patterns to Handle

### 1. Long Focused Sessions
- Extended discussions on single topics
- Low drift within session
- Recent context most important

### 2. One-Shot Questions
- Unrelated queries that won't be revisited
- High drift from existing conversation
- Should become children of root node
- Minimal historical context needed

### 3. Recurring Topics
- Periodic return to previous discussion threads
- Example: Game strategy discussions resumed across sessions
- Rich historical context from previous branch summaries valuable
- Low drift when compared to relevant branch summaries

## Branch Summarization Strategies

### Strategy 1: Bottom-up Hierarchical
- Start at leaf nodes, summarize small groups (2-3 nodes)
- Work upward, summarizing the summaries
- Root of branch gets final summary
- Preserves conversational flow structure

### Strategy 2: Progressive/Rolling
- Maintain running summary traversing root to current node
- Each new node updates rather than replaces summary
- Captures conversation evolution

### Strategy 3: Key Moment Extraction
- Identify important nodes (topic shifts, decisions, conclusions)
- Summarize only key moments
- Preserves critical information while compressing

### Strategy 4: LLM Single-Pass
- Feed entire branch to LLM with summarization prompt
- Simple but potentially expensive for long branches

## Summarization Implementation Options

### Local/Free Options
- **Transformers Pipeline**: `pipeline("summarization")` from Hugging Face
  - Any Hugging Face summarization model can be plugged in
  - Good for custom branch summarization without API costs
  - Suitable for MVP implementation

### Cloud/Paid Options  
- **OpenAI GPT-3.5/4**: High-quality summarization via API
  - More expensive but potentially better quality
  - Good for production systems where quality is critical
  - Not required for initial MVP

## Distance Functions (from RAG techniques)

### Basic Measures
- Cosine similarity (most common)
- Euclidean distance  
- Dot product similarity

### Advanced Approaches
- **Hybrid search**: Semantic + keyword/lexical matching
- **Re-ranking**: Separate model for conversational relevance scoring
- **Contextual compression**: Filter retrieved context to most relevant parts
- **Parent document retrieval**: Retrieve larger context chunks
- **Hypothetical questions**: Match against generated questions the context might answer

## Real-time Drift Display

Display multiple drift measurements:
1. **Local drift**: Between current query and semdepth context window
2. **Branch drift**: Between current query and branch summary
3. **Global drift**: Comparison against other branch summaries

Example output:
```
> What's the best strategy for early game in chess?
[Local drift: 0.85 | Branch drift: 0.92 | Similar to: "Game Strategy Discussion"]
🤖 response about chess strategy...

> Should I focus on center control or piece development first?
[Local drift: 0.12 | Branch drift: 0.15 | Continuing: "Chess Strategy"]
🤖 response continues chess discussion...
```

## Automatic Restructuring Logic

### High Drift Detection
1. Calculate drift scores for new query
2. If drift exceeds threshold, walk up DAG from current position
3. Compare semantic similarity at each ancestor level
4. Find best semantic match in conversation history
5. Either move node or create weak link to better parent

### One-Shot Detection
- Very high drift from all existing nodes
- Automatically parent to root node
- Minimal context window construction

### Topic Resumption Detection
- High local drift but low drift from specific branch summary
- Re-parent to that branch or create strong link
- Include relevant branch summary in context window

## Technical Considerations

### Performance
- Embedding computation on-demand vs pre-computed
- Caching strategies for branch summaries
- Incremental updates vs full recalculation

### Modularity
- `BranchSummarizer` interface for pluggable strategies
- `EmbeddingProvider` interface for different models
- `DistanceFunction` interface for similarity measures
- `DriftAnalyzer` orchestrates the components

### Configuration
- Adjustable semdepth parameter
- Drift thresholds for restructuring triggers
- Summarization strategy selection
- Embedding model selection

## Open Questions

1. **Summary Scope**: Entire branches vs moving topic windows?
2. **Weighting**: How to weight nodes within semdepth window?
3. **Weak Links**: How to represent and utilize weak connections?
4. **Performance**: Real-time requirements vs computation cost?
5. **Validation**: How to measure if restructuring improves conversation quality?

## Success Metrics

- More natural conversation flow across topic boundaries
- Relevant historical context available when resuming topics
- Reduced irrelevant context in attention window
- Intuitive drift scores that match human perception of topic changes