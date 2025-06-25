# Conversational Drift Detection Implementation Plan

This document outlines the recommended order for implementing the conversational drift detection system described in `ConversationalDriftDesign.md`.

## Phase 1: Foundation (Start Here)

### 1. Implement basic DistanceFunction.calculate()
- **Goal**: Get cosine similarity working first (most common for text embeddings)
- **Location**: `episodic/mlconfig.py` - DistanceFunction class
- **Why first**: Simple, well-understood algorithm that will work for most cases
- **Success criteria**: Can calculate similarity between two embedding vectors

### 2. Implement basic EmbeddingProvider.embed()
- **Goal**: Get sentence-transformers working (local, no API needed)
- **Location**: `episodic/mlconfig.py` - EmbeddingProvider class
- **Why early**: Local setup, no API costs, easy to experiment
- **Success criteria**: Can generate embeddings for text strings
- **Suggested model**: Start with "all-MiniLM-L6-v2" (small, fast)

### 3. Create a simple test harness
- **Goal**: Load existing conversations and compute embeddings for testing
- **Location**: New file like `test_drift_manually.py` or add to existing CLI
- **Why important**: Need real data to validate the system works
- **Success criteria**: Can load conversations from DB and generate embeddings

## Phase 2: Core Drift Detection

### 4. Implement semdepth context building
- **Goal**: Function to collect N ancestor nodes and concatenate their content
- **Location**: New drift detection module or in `mlconfig.py`
- **Details**: Walk up DAG from current node, collect semdepth nodes, combine text
- **Success criteria**: Given a node ID and semdepth=3, returns combined text of node + 2 ancestors

### 5. Add real-time drift display to CLI
- **Goal**: Show drift scores above each query/response in talk mode
- **Location**: Modify `episodic/cli.py` - main conversation loop
- **Format**: `[Semantic drift: 0.85 from previous context]`
- **Why critical**: Immediate feedback loop for understanding drift patterns
- **Success criteria**: See drift scores in real-time during conversations

### 6. Test on your actual conversations
- **Goal**: Use the system on real conversations to see drift patterns
- **Approach**: Have conversations on different topics, watch drift scores
- **Data collection**: Note when drift scores feel right vs wrong
- **Success criteria**: Intuitive understanding of what different drift values mean

## Phase 3: Understanding Patterns

### 7. Add multiple distance functions
- **Goal**: Compare cosine vs euclidean vs dot product similarity
- **Location**: Extend `DistanceFunction.calculate()` with more algorithms
- **Why useful**: Different algorithms may capture different aspects of semantic drift
- **Success criteria**: Can switch between distance algorithms and see differences

### 8. Experiment with semdepth values
- **Goal**: Test different context window sizes (1, 3, 5 ancestor nodes)
- **Approach**: Try same conversations with different semdepth settings
- **Data**: Which semdepth gives most intuitive drift detection?
- **Success criteria**: Understand optimal semdepth for different conversation types

### 9. Collect data on your conversation patterns
- **Goal**: Document which settings work best for different conversation types
- **Data points**: 
  - One-shot questions vs long discussions
  - Topic resumption patterns
  - Natural topic drift within conversations
- **Success criteria**: Can predict good drift settings for different use cases

## Phase 4: Branch Summarization

### 10. Implement LocalLLMSummarizer
- **Goal**: Use existing Ollama/LiteLLM integration for branch summarization
- **Location**: `BranchSummarizer.summarize_branch()` method
- **Integration**: Leverage existing `episodic.llm` module
- **Success criteria**: Can generate summaries of conversation branches

### 11. Add branch summary drift comparison
- **Goal**: Show both local drift (semdepth) and global drift (vs branch summary)
- **Display**: `[Local drift: 0.85 | Branch drift: 0.12 | Similar to: "Chess Strategy"]`
- **Why important**: Detect topic resumption vs new topics
- **Success criteria**: Can distinguish between local topic drift and returning to previous topics

### 12. Test summarization quality
- **Goal**: Validate that branch summaries capture conversation essence
- **Method**: Read summaries vs original conversations
- **Tuning**: Adjust summarization prompts for better quality
- **Success criteria**: Summaries are useful for semantic comparison

## Phase 5: Automation (Much Later)

### 13. Implement automatic restructuring logic
- **Goal**: Automatically move nodes in DAG based on semantic similarity
- **Approach**: 
  - Detect high drift (threshold-based)
  - Walk up DAG comparing similarity
  - Move nodes or create weak links
- **Why last**: Need deep understanding of drift patterns first
- **Success criteria**: Automatic restructuring improves conversation flow

## Implementation Strategy

### Quick Wins First
- Start with steps 1-6 as a focused sprint
- Goal: See drift scores in real-time within 1-2 days
- This provides immediate feedback and motivation

### Iterative Refinement
- Each step builds understanding for the next
- Use real conversation data throughout
- Adjust approach based on what you learn

### Avoid Premature Optimization
- Don't automate until you understand the patterns
- Focus on measurement and observation first
- Automation is the final step, not the first

### Data-Driven Development
- Use your actual conversations for testing
- Document what works and what doesn't
- Let the data guide algorithm choices

## Success Metrics

### Phase 1-2 Success
- Can see drift scores in real-time
- Scores correlate with intuitive sense of topic changes
- System works reliably with existing conversations

### Phase 3 Success  
- Understand optimal settings for different conversation types
- Can predict when topics are shifting vs continuing
- Have baseline metrics for automation decisions

### Phase 4 Success
- Branch summaries are coherent and useful
- Can detect topic resumption vs new topic initiation
- System helps with context management

### Phase 5 Success
- Automatic restructuring feels natural
- Conversation flow improves noticeably
- System reduces cognitive load of context management

## Notes

- Keep implementations simple initially
- Focus on learning over optimization
- Document insights as you go
- Be prepared to adjust the plan based on discoveries

## Quick Start Commands

Once basic implementation is ready:

```python
# Test basic functionality
from episodic.mlconfig import get_local_config
config = get_local_config()

# Test embedding
embedding = config.embedding_provider.embed("Hello world")

# Test distance  
dist = config.distance_function.calculate(embedding1, embedding2)
```

Remember: The goal is understanding, not perfection. Start simple and iterate!