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

## Notes
- Current drift detection compares consecutive user messages only (not assistant responses)
- LLM-based detection would use confidence levels: change-high, change-medium, change-low
- Performance vs accuracy tradeoff between fast/inaccurate vs slow/better models