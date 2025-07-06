# Cruft Removal Plan for Episodic

## Analysis Summary

After analyzing the codebase for deprecated functionality and unused code, here's what can be safely removed:

## 1. Deprecated Commands (Ready to Remove)

According to DEPRECATED.md, these commands were deprecated in PR #5 and scheduled for removal in v0.5.0:

### Topic Commands
- `/rename-topics` → Use `/topics rename`
- `/compress-current-topic` → Use `/topics compress`
- `/index` → Use `/topics index`
- `/topic-scores` → Use `/topics scores`

### Compression Commands
- `/compression-stats` → Use `/compression stats`
- `/compression-queue` → Use `/compression queue`
- `/api-stats` → Use `/compression api-stats`
- `/reset-api-stats` → Use `/compression reset-api`

**Action**: Remove handlers and registrations for these deprecated commands.

## 2. Backward Compatibility Wrappers

These files exist only for backward compatibility and can be removed once all imports are updated:

- `episodic/topics.py` - Wrapper importing from new topics module structure
- `episodic/topics_hybrid.py` - Wrapper importing from topics.hybrid and topics.keywords

**Action**: Update all imports to use new module paths, then remove these files.

## 3. Legacy Configuration Parameters

Found in config_defaults.py:
- `streaming` (line 61) - Marked as legacy, replaced by `stream_responses`

**Action**: Remove from config_defaults.py and config files.

## 4. Unused ML Stub Implementations

The ML directory contains stub implementations that were never completed:

### Files with NotImplementedError or TODO-only implementations:
- `episodic/mlconfig.py` - Contains incomplete EmbeddingProvider and BranchSummarizer classes
- `episodic/ml/embeddings/providers.py` - All backends have only TODO comments
- `episodic/ml/summarization/strategies.py` - All backends raise NotImplementedError

**Status**: ConversationalDrift from ml/drift.py IS being used, so ML module cannot be entirely removed.

**Action**: Remove only the unused stub files while keeping the working drift detection.

## 5. Empty/Stub Functions

From DEPRECATED.md:
- `episodic/db.py::close_connection()` - No-op function
- `episodic/cli.py::setup_readline()` - Empty function

**Action**: Remove these functions and update any calling code.

## 6. Legacy Command Handling

In `episodic/cli_registry.py`:
- `handle_legacy_command()` function (lines 88-107) - Only needed for unconverted commands

**Action**: Can be removed once all commands use the new registry style.

## 7. Comment Artifacts in Config Files

JSON files contain comment fields like:
- `"_comment_core"`, `"_comment_topic"`, etc.

**Action**: These are actually useful for documentation, so KEEP them.

## 8. Unused Variables (per DEPRECATED.md)

- `episodic/cli.py` line 547: `hex_color` assigned but never used
- `episodic/commands/model.py` line 123: `selected_provider` assigned but never used

**Action**: Remove these unused variables.

## Removal Priority

### Phase 1 (Immediate - Low Risk):
1. Remove deprecated command handlers and registrations
2. Remove legacy `streaming` config parameter
3. Remove unused variables
4. Remove empty stub functions (close_connection, setup_readline)

### Phase 2 (After Testing):
1. Remove backward compatibility wrappers (topics.py, topics_hybrid.py) after updating imports
2. Remove ML stub implementations (but keep working drift.py)

### Phase 3 (Future):
1. Remove handle_legacy_command once all commands are converted
2. Consider consolidating the command handling to use only the registry

## Next Steps

1. Create a branch for cruft removal
2. Remove Phase 1 items first
3. Run full test suite
4. Update imports for Phase 2
5. Remove Phase 2 items
6. Update documentation

## Notes

- The defunct/ directory has already been removed ✓
- The ML ConversationalDrift functionality is actively used, so only remove stub implementations
- Config comment fields are useful for documentation - keep them
- Be careful with command removal - ensure no scripts depend on old names