# Episodic Project TODO

## Semantic Drift Performance Optimization

### High Priority
- [ ] **Reduce drift calculation slowdown** - Switch back to faster embedding model (`all-MiniLM-L6-v2` vs current `paraphrase-mpnet-base-v2`) to improve CLI responsiveness
- [ ] **Implement LLM-based topic detection** - Replace embedding-based drift with LLM prompt-based detection for better accuracy on conversational text
- [ ] **Add async drift processing** - Calculate drift in background to eliminate user-facing delays

### Medium Priority  
- [ ] **Improve drift accuracy** - Current embedding approach shows high drift (0.6-0.9) for semantically similar short sentences
- [ ] **Better caching strategy** - Optimize embedding cache to reduce redundant calculations

### Low Priority
- [ ] **Tokenizer warning cleanup** - Already addressed with `TOKENIZERS_PARALLELISM=false`

## Silent LLM Topic Extraction Implementation

### Phase 1: Basic Infrastructure
- [ ] **Add topic extraction function** - Create `extract_topic_ollama()` function with prompt design
- [ ] **Database schema** - Add topics table to store topic names and conversation ranges
- [ ] **Integration point** - Hook topic extraction into existing LLM change detection

### Phase 2: `/topics` Command
- [ ] **Implement `/topics` command** - Show recent topics with conversation ranges and confidence
- [ ] **Topic storage/retrieval** - Store extracted topics with node ranges in database
- [ ] **Testing and refinement** - Test topic extraction quality and adjust prompts

### Phase 3: Optional Features  
- [ ] **Optional topic display** - Add `/set topics true` to show topic evolution in CLI
- [ ] **Topic navigation** - Allow jumping to topic ranges in conversation history
- [ ] **Topic refinement** - Improve extraction prompts based on real usage

## Notes
- Current drift detection compares consecutive user messages only (not assistant responses)
- LLM-based detection would use confidence levels: change-high, change-medium, change-low
- Silent topic extraction uses Ollama (free) to avoid interrupting conversation flow
- Performance vs accuracy tradeoff between fast/inaccurate vs slow/better models