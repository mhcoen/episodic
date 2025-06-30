# Episodic Codebase Cleanup Plan

## Overview
Systematic cleanup to prepare codebase for adaptive topic detection features. Each PR is designed to be small, focused, and independently mergeable.

## PR Sequence

### PR #1: File Organization (Low Risk, High Impact)
**Branch**: `cleanup/organize-files`
**Size**: ~20 files moved, 0 logic changes

**Changes**:
```bash
# Create directory structure
mkdir -p scripts/{analysis,testing,benchmarks}
mkdir -p episodic/{migrations,topics,commands/utils}
mkdir -p tests/{unit,integration,scripts}

# Move analysis scripts
git mv check_index_scores.py scripts/analysis/
git mv compare_window_sizes.py scripts/analysis/
git mv test_index_order.py scripts/analysis/
git mv verify_windows.py scripts/analysis/
git mv analyze_topics.py scripts/analysis/
git mv show_topic_contents.py scripts/analysis/
git mv test_dynamic_threshold.py scripts/analysis/

# Move test scripts
git mv run_realistic_test.py scripts/testing/
git mv test_*.py tests/scripts/  # Various test files

# Move migrations
git mv migrate_index_table.py episodic/migrations/
```

**Update imports**:
- Fix any broken imports in moved files
- Update .gitignore if needed

**Testing**: 
- Verify all moved scripts still run
- Check no imports broken in main code

**Review checklist**:
- [ ] No logic changes
- [ ] All files accounted for
- [ ] Scripts still executable

---

### PR #2: Configuration Consolidation
**Branch**: `cleanup/unified-config`
**Size**: ~5 files changed, unified configuration

**Changes**:

1. Create `episodic/config_defaults.py`:
```python
# All topic detection settings in one place
TOPIC_DETECTION_DEFAULTS = {
    # Core settings
    "automatic_topic_detection": True,
    "detection_method": "hybrid",  # "hybrid", "sliding_window", "simple"
    
    # Sliding window settings
    "window_size": 3,
    "drift_threshold": 0.75,
    "use_or_logic": True,  # drift OR keywords
    
    # Hybrid detection settings  
    "hybrid_topic_threshold": 0.55,
    "min_messages_before_topic_change": 8,
    "first_topic_threshold": 3,
    
    # Keyword detection
    "enable_keyword_detection": True,
    "keyword_confidence_threshold": 0.5,
    
    # Model settings
    "topic_detection_model": "ollama/llama3",
    "embedding_model": "paraphrase-mpnet-base-v2",
    "embedding_provider": "sentence-transformers",
    
    # UI settings
    "show_topics": False,
    "topic_colors": True,
    
    # Background processing (future)
    "background_topic_detection": False,
    "topic_detection_delay": 5,
}

# Organize other settings
UI_DEFAULTS = {...}
MODEL_DEFAULTS = {...}
COMPRESSION_DEFAULTS = {...}
```

2. Update `config.py` to use defaults:
```python
from .config_defaults import TOPIC_DETECTION_DEFAULTS

def get_default_config():
    config = {}
    config.update(TOPIC_DETECTION_DEFAULTS)
    config.update(UI_DEFAULTS)
    # ...
    return config
```

3. Remove hardcoded values:
- Search for magic numbers like `0.55`, `0.75`, `3`
- Replace with config lookups
- Document each setting's purpose

**Testing**:
- Verify all settings still work
- Check `/set` command recognizes all settings
- Ensure defaults are sensible

---

### PR #3: Topic Detection Module Restructure
**Branch**: `cleanup/topic-module`
**Size**: ~10 files, moving code around

**Create structure**:
```
episodic/topics/
├── __init__.py           # Public API
├── detector.py           # Main TopicDetector class
├── boundaries.py         # Boundary detection logic
├── embeddings.py         # Topic embedding management
├── windows.py            # Sliding window implementation
├── keywords.py           # Keyword detection
├── hybrid.py             # Hybrid detector (current)
└── utils.py             # Shared utilities
```

**Move code**:
1. Extract from `topics.py`:
   - TopicDetector → `detector.py`
   - Boundary logic → `boundaries.py`
   - Window logic → `windows.py`

2. Extract from `topics_hybrid.py`:
   - HybridTopicDetector → `hybrid.py`
   - TransitionKeywordDetector → `keywords.py`

3. Create clean public API in `__init__.py`:
```python
from .detector import TopicDetector
from .windows import SlidingWindowDetector
from .hybrid import HybridTopicDetector

__all__ = ['TopicDetector', 'SlidingWindowDetector', 'HybridTopicDetector']
```

**Testing**:
- All existing tests pass
- Topic detection still works
- No functionality changes

---

### PR #4: Database Schema Cleanup
**Branch**: `cleanup/database-schema`
**Size**: ~5 files, schema migrations

**Changes**:

1. Create migration script:
```python
# episodic/migrations/004_cleanup_topic_tables.py
def upgrade():
    # Rename manual_index_scores to be more generic
    conn.execute("ALTER TABLE manual_index_scores RENAME TO topic_detection_scores")
    
    # Add detection_method column
    conn.execute("ALTER TABLE topic_detection_scores ADD COLUMN detection_method TEXT DEFAULT 'manual'")
    
    # Add indexes for performance
    conn.execute("CREATE INDEX idx_topic_scores_node ON topic_detection_scores(user_node_short_id)")
    conn.execute("CREATE INDEX idx_topics_boundaries ON topics(start_node_id, end_node_id)")
    
    # Add missing foreign keys
    # ...
```

2. Update all references:
- `store_manual_index_score` → `store_topic_detection_score`
- `get_manual_index_scores` → `get_topic_detection_scores`

3. Document schema:
```sql
-- Add to episodic/schema.sql
-- Complete schema documentation
```

**Testing**:
- Migration runs cleanly
- Existing data preserved
- All queries still work

---

### PR #5: Command Consolidation
**Branch**: `cleanup/commands`
**Size**: ~8 files, command refactoring

**Changes**:

1. Consolidate topic commands:
```python
# episodic/commands/topics.py
def topics_command(
    action: str = typer.Argument("list", help="list|rename|reindex|stats"),
    **kwargs
):
    """Unified topic management command"""
    if action == "list":
        list_topics(**kwargs)
    elif action == "rename":
        rename_topics(**kwargs)
    elif action == "reindex":
        reindex_topics(**kwargs)
    elif action == "stats":
        show_topic_stats(**kwargs)
```

2. Deprecate redundant commands:
- Mark old commands as deprecated
- Point to new unified commands
- Plan removal in future version

3. Improve command discovery:
```python
# episodic/commands/__init__.py
COMMAND_REGISTRY = {
    'topics': topics_command,
    'index': index_command,
    'settings': settings_command,
    # ...
}
```

**Testing**:
- All commands still accessible
- Help text is clear
- Deprecation warnings work

---

### PR #6: Test Infrastructure
**Branch**: `cleanup/test-infrastructure`
**Size**: ~10 files, test organization

**Changes**:

1. Organize tests:
```
tests/
├── unit/
│   ├── test_topics.py
│   ├── test_embeddings.py
│   └── test_commands.py
├── integration/
│   ├── test_topic_detection.py
│   └── test_conversation_flow.py
├── fixtures/
│   ├── conversations.py
│   └── test_data.py
└── conftest.py
```

2. Create test utilities:
```python
# tests/fixtures/conversations.py
def create_test_conversation(topics: List[str]) -> List[Node]:
    """Generate test conversation with known topic boundaries"""
    
def assert_topics_detected(detected: List[Topic], expected: List[Topic]):
    """Helper for topic detection tests"""
```

3. Add missing tests:
- Sliding window detection
- OR logic for keywords
- Configuration handling

**Testing**:
- All tests pass
- Coverage improves
- Tests run faster

---

### PR #7: Dead Code Removal
**Branch**: `cleanup/remove-dead-code`
**Size**: ~15 files, removing code

**Identify and remove**:
1. Unused imports
2. Commented code blocks
3. Deprecated functions
4. Orphaned test files

**Document decisions**:
```python
# DEPRECATED.md
## Removed in v0.X.0

### Single-message topic detection
- Removed: `detect_topic_change_simple()`
- Reason: Sliding window proven more effective
- Migration: Use `SlidingWindowDetector`

### TF-IDF drift calculation
- Removed: Legacy drift calculator
- Reason: Embeddings provide better results
- Migration: Already using embeddings
```

**Testing**:
- No functionality broken
- All imports resolve
- No warnings in IDE

---

### PR #8: Documentation Update
**Branch**: `cleanup/documentation`
**Size**: Documentation only

**Update**:
1. `CLAUDE.md` - Current architecture
2. `README.md` - Remove outdated info
3. `ARCHITECTURE.md` - New file with system design
4. API documentation for topic module
5. Configuration reference

**Add decision log**:
```markdown
# DECISIONS.md

## Topic Detection Architecture (2024-11)
- Chose sliding window over single-message
- Reason: Better accuracy (83% vs 45%)
- Trade-off: Slightly more complex

## Database Schema (2024-11)
- Kept topics table simple
- Store detection scores separately
- Reason: Flexibility for experiments
```

---

## Execution Strategy

### Week 1: Foundation (PR #1-2)
- Monday: PR #1 (File organization)
- Tuesday-Wednesday: PR #2 (Configuration)
- Thursday-Friday: Review and merge

### Week 2: Core Refactoring (PR #3-4)
- Monday-Tuesday: PR #3 (Topic module)
- Wednesday-Thursday: PR #4 (Database)
- Friday: Integration testing

### Week 3: Polish (PR #5-7)
- Monday-Tuesday: PR #5 (Commands)
- Wednesday: PR #6 (Tests)
- Thursday: PR #7 (Dead code)
- Friday: PR #8 (Documentation)

## Success Metrics

### Code Quality
- [ ] Cyclomatic complexity reduced by 30%
- [ ] Test coverage increased to 80%
- [ ] No circular imports
- [ ] All TODOs addressed or documented

### Developer Experience
- [ ] New developer can understand structure in 30 min
- [ ] Clear where to add new features
- [ ] Consistent patterns throughout

### Performance
- [ ] No performance regressions
- [ ] Topic detection <100ms
- [ ] Tests run in <30s

## Risks and Mitigations

### Risk: Breaking existing functionality
**Mitigation**: Each PR includes tests, small incremental changes

### Risk: Merge conflicts during cleanup
**Mitigation**: Fast review cycle, coordinate with team

### Risk: User-visible changes
**Mitigation**: Maintain backward compatibility, deprecation warnings

## Post-Cleanup Benefits

1. **Ready for adaptive detection**: Clear where to add new features
2. **Easier maintenance**: Consistent structure and patterns
3. **Better testing**: Organized test suite with utilities
4. **Improved performance**: Removed redundant calculations
5. **Clearer documentation**: Accurate representation of system

## Next Steps After Cleanup

With clean codebase, can proceed to:
1. Implement Phase 1 of adaptive topic detection
2. Add background processing infrastructure
3. Experiment with better embeddings
4. Build topic-based features

---

This cleanup plan will take approximately 3 weeks with focused effort, but each PR provides immediate value and can be done independently if needed.