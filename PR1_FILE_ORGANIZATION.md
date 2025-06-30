# PR #1: File Organization

## Summary
Organized test and analysis scripts into proper directories with no logic changes.

## Changes Made

### Created Directory Structure
```
scripts/
├── analysis/      # Analysis and debugging scripts
├── testing/       # Test execution scripts
└── benchmarks/    # (Reserved for future benchmark scripts)

tests/
├── unit/          # Unit tests
├── integration/   # Integration tests
├── scripts/       # Test scripts (formerly in root)
└── fixtures/      # (Reserved for test fixtures)

episodic/
└── migrations/    # Database migration scripts
```

### Files Moved

#### To `scripts/analysis/`:
- analyze_topic_boundaries.py
- analyze_topics.py
- check_conversation.py
- check_index_scores.py
- check_topic_names.py
- compare_window_sizes.py
- show_all_nodes.py
- show_topic_contents.py
- test_dynamic_threshold.py
- verify_windows.py

#### To `scripts/testing/`:
- fix_topic_config.py
- run_realistic_test.py
- test_benchmark_after_commands.py

#### To `tests/scripts/`:
- test_check_weights.py
- test_cli_topics.py
- test_command_debug.py
- test_debug_embeddings.py
- test_debug_hybrid.py
- test_debug_keywords.py
- test_debug_signals.py
- test_doc_context.py
- test_edge_cases.py
- test_final_verification.py
- test_hybrid_topics.py
- test_integration.py
- test_interactive.py
- test_manual_indexing.py
- test_simple_hybrid.py
- test_topic_boundaries.py
- test_topic_flow.py
- test_topic_messages.py
- test_topic_tracking.py
- test_topics_command.py
- test_topics_debug.py

#### To `tests/unit/`:
- test_drift_manual.py (from episodic/)

#### To `episodic/migrations/`:
- migrate_index_table.py

### Import Fixes
All moved scripts had their imports updated to work from their new locations:
- Changed `sys.path.insert(0, '.')` to `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))`
- Added `import os` where needed

## Testing
Run a few scripts to verify they still work:
```bash
# Test an analysis script
python scripts/analysis/check_topic_names.py

# Test a test script
python tests/scripts/test_manual_indexing.py

# Test the migration script
python episodic/migrations/migrate_index_table.py
```

## Next Steps
With files properly organized, we can now proceed to PR #2: Configuration Consolidation.