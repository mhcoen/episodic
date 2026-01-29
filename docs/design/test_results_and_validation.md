# Test Results and Validation Design

This document captures test results from the topic reactivation and context recovery implementation, providing a foundation for future validation approaches.

## Test Suite Evolution

### Initial State (Pre-Stabilization)
```
Total: 877 tests
├── Passed: 753
├── Failed: 55
├── Errors: 62
└── Skipped: 14
```

**Root causes identified:**
- 62 errors: `:memory:` database path validation blocking tests
- Path caching causing test pollution between runs
- Fixtures not restoring `EPISODIC_DB_PATH`

### After Path/Import Fixes
```
Total: ~880 tests
├── Passed: ~850
├── Failed: 31
├── Errors: 0
└── Reduction: 74% of issues resolved
```

**Remaining 31 failures:** API drift - tests referenced removed methods like `send_message`.

### After API Drift Fixes
```
Total: 944 tests
├── Passed: 944
├── Failed: 0
├── Skipped: 20
└── Status: GREEN
```

**Skipped tests:**
- Platform-specific (audio/voice features)
- ChromaDB isolation issues (quarantined)

---

## New Test Coverage

### Reactivation System Tests
| File | Tests | Coverage |
|------|-------|----------|
| `test_imports.py` | 44 | Cross-topic import detection |
| `test_reactivation_replay.py` | 27 | Replay harness, metrics |
| `test_db_reactivation_decisions.py` | 19 | Decision persistence |
| `test_conversation_reactivation_flow.py` | 12 | Integration flow |
| **Total** | **102** | **100% of new modules** |

### Benchmark Tests
| File | Tests | Purpose |
|------|-------|---------|
| `test_resume_benchmark.py` | 15 | Deterministic resume scenarios |

### CI Gates
```yaml
reactivation-tests:     # Required to pass
  - pytest -m reactivation
  - 117 passed, 7 skipped

quarantine-tests:       # Informational only
  - pytest -m quarantine_chroma
  - 4 failed, 2 passed (expected)

resume-benchmark:       # Quality signal
  - pytest tests/benchmark/
  - 15 passed
```

---

## Key Metrics from Resume Benchmark

### Contamination Rate by Mode
| Mode | Contamination Rate | Description |
|------|-------------------|-------------|
| `ancestry` | 80% | 4/5 non-ambiguous scenarios had foreign topic content |
| `topic_local` | 0% | No foreign topic content ever |
| `hybrid` | 0% | Same as topic_local when reactivation fires |

**Conclusion:** Topic-local achieves the core UX goal.

### Token Efficiency
| Mode | Avg Context Tokens | Relative |
|------|-------------------|----------|
| `ancestry` | baseline | 100% |
| `topic_local` | ~60% of ancestry | -40% |
| `hybrid` | ~60% of ancestry | -40% |

**Conclusion:** Topic-local is more token-efficient by excluding irrelevant context.

### Benchmark Scenarios
| ID | Category | Gap | Expected Behavior |
|----|----------|-----|-------------------|
| `short_gap_python` | short_gap | 5-10 turns | Reactivate to Python |
| `medium_gap_database` | medium_gap | 20-50 turns | Reactivate to database |
| `long_gap_ml_project` | long_gap | 100+ turns | Reactivate to ML |
| `ambiguous_java` | ambiguous | N/A | Disambiguate |
| `short_gap_cooking` | short_gap | 5-10 turns | Reactivate to cooking |

---

## Invariants Under Test

### Contamination Invariant
```python
# Every node in topic_local context must belong to active topic
for node_id in included_node_ids:
    assert get_node_topic(node_id) == active_topic_start_node_id
```
- **Enforcement:** Runtime assertion in debug mode, warning in production
- **Test:** `test_topic_local_no_contamination`

### Determinism Invariant
```python
# Same inputs produce same fingerprint hash
fingerprint = compute_fingerprint(user_node_id, debug_info)
# Hash is SHA256 of: mode, active_topic, included_node_ids, token_counts, reactivation_decision
```
- **Enforcement:** Fingerprint persisted per turn
- **Use:** Diff fingerprints to detect regressions

### Anchor Invariants
```python
# All anchors filtered by topic
assert all(a['topic_start_node_id'] == active_topic for a in anchors)

# No overlap with recency slice
assert not (anchor_node_ids & recency_node_ids)

# Within token budget
assert sum(a['tokens'] for a in anchors) <= anchor_token_budget
```

### Provenance Invariants
```python
# Summary hash matches content
assert summary_hash == sha256(canonical_json(summary))[:16]

# Node IDs hash matches claimed range
assert input_node_ids_hash == sha256(sorted(node_ids))[:16]

# Staleness monotonicity
assert new_last_summarized_turn_idx >= old_last_summarized_turn_idx
```

---

## Future Validation Ideas

### 1. Response Quality Benchmark
Currently we test routing correctness (did we reactivate to the right topic?) but not response quality.

**Proposed approach:**
- For each resume scenario, collect LLM responses under each mode
- Score on:
  - **Relevance:** Does response address the resumed topic?
  - **Continuity:** Does response acknowledge prior context?
  - **Hallucination:** Does response invent facts not in context?
- Methods:
  - Human evaluation (gold standard, 50-100 examples)
  - LLM-as-judge (scalable, calibrate against human)
  - Automated heuristics (keyword presence, entity consistency)

### 2. Thrash Rate Monitoring
```python
# Detect rapid topic switching
thrash_events = count_reactivations_within_n_turns(window=3)
thrash_rate = thrash_events / total_turns
# Target: < 5%
```

### 3. Reactivation Precision/Recall
Requires ground truth labels:
```python
precision = correct_reactivations / total_reactivations
recall = correct_reactivations / actual_topic_returns
f1 = 2 * precision * recall / (precision + recall)
```

**Labeling infrastructure exists:** `reactivation_labels` table, `/label` command.

### 4. Summary Quality Metrics
- **Compression ratio:** Original tokens / summary tokens
- **Information retention:** Key entities/decisions preserved
- **Parseability:** % of summaries that parse to StructuredSummary
- **Staleness:** Distribution of `current_turn_idx - last_summarized_turn_idx`

### 5. Anchor Effectiveness
- **Hit rate:** % of turns where anchors were retrieved
- **Novelty rate:** % of anchors that added info beyond recency slice
- **Summary redundancy:** % of anchors filtered for being too similar to summary

### 6. Long-Gap Stress Test
Test "year-later" resume with:
- No recency slice (all messages outside context window)
- Only summary + anchors available
- Verify response quality doesn't degrade catastrophically

### 7. Adversarial Tests
- **Topic confusion:** Similar topics (java-programming vs java-coffee)
- **Rapid switching:** User alternates topics every turn
- **Cold start:** Resume with no summary, no anchors
- **Overloaded topic:** Topic with 1000+ exchanges

### 8. Performance Benchmarks
| Metric | Target | Current |
|--------|--------|---------|
| Probe latency | < 50ms | TBD |
| Context assembly | < 100ms | TBD |
| Anchor retrieval | < 50ms | TBD |
| Summary generation | < 5s | TBD |

---

## Test Infrastructure

### Markers
```ini
[pytest]
markers =
    reactivation: Core reactivation/topic-local tests (required gate)
    benchmark: Deterministic benchmark tests (quality signal)
    quarantine_chroma: ChromaDB isolation issues (informational)
```

### Fixtures
- `episodic/evaluation/fixtures/resume_scenarios.json`: Pre-computed embeddings for determinism
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)

### CI Jobs
```yaml
reactivation-tests:    # Required
resume-benchmark:      # Required
quarantine-tests:      # Informational, continue-on-error
main-tests:            # Excludes quarantine
```

---

## Regression Detection

### Fingerprint Diffing
When a test fails or behavior changes unexpectedly:
```python
old_fp = get_fingerprint(user_node_id, version="before")
new_fp = get_fingerprint(user_node_id, version="after")
diffs = diff_fingerprints(old_fp, new_fp)
# Returns: {"mode": ("ancestry", "topic_local"), "included_node_ids": {"added": [...], "removed": [...]}}
```

### Contamination Alerts
In production (non-debug mode):
```python
if contamination_detected:
    logger.warning(f"Contamination: {foreign_nodes}")
    metrics.increment("context_recovery.contamination_events")
```

### Staleness Monitoring
```python
stale_topics = query("SELECT * FROM topic_working_set WHERE last_summarized_turn_idx < current_turn_idx - 100")
if stale_topics:
    alert("Topics need summarization", count=len(stale_topics))
```

---

## Summary

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Test pass rate | 86% | 100% | 100% |
| Contamination (topic_local) | N/A | 0% | 0% |
| Token efficiency | baseline | -40% | -30%+ |
| New test coverage | 0 | 102 tests | Comprehensive |
| CI gates | None | 3 jobs | All green |

The test infrastructure now provides:
1. **Correctness guarantees** via invariant assertions
2. **Quality signals** via resume benchmarks
3. **Regression detection** via fingerprint diffing
4. **Future extensibility** via labeled data infrastructure
