# Deprecated Code in Episodic

This document tracks deprecated code that can be removed in future versions.

## Deprecated Commands (as of PR #5)

The following commands have been replaced with unified commands and are marked as deprecated:

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

**Migration Timeline**: These deprecated commands will be removed in v0.5.0

## Backward Compatibility Files

### episodic/topics.py
- **Status**: Backward compatibility wrapper
- **Purpose**: Imports from new topics module structure
- **Can be removed**: When all code is updated to import from `episodic.topics.*`

### episodic/topics_hybrid.py
- **Status**: Backward compatibility wrapper
- **Purpose**: Imports from `episodic.topics.hybrid` and `episodic.topics.keywords`
- **Can be removed**: When all code is updated to use new imports

## Unimplemented ML Features

The following files contain stub implementations that should either be completed or removed:

### episodic/mlconfig.py
- `EmbeddingProvider` class with TODO comments
- `BranchSummarizer` class that raises NotImplementedError
- Various configuration classes with incomplete implementations

### episodic/ml/embeddings/providers.py
- `OpenAIBackend`: All methods contain TODO comments
- `HuggingFaceBackend`: All methods contain TODO comments

### episodic/ml/summarization/strategies.py
- `LocalLLMBackend`: Raises NotImplementedError
- `OpenAIBackend`: Raises NotImplementedError
- `HuggingFaceBackend`: Raises NotImplementedError
- `HierarchicalBackend`: Raises NotImplementedError

**Decision**: These ML features were experimental and not integrated into the main application. They can be removed unless there's a plan to implement them.

## Empty/Stub Functions

### episodic/db.py
- `close_connection()`: Now a no-op since connections are managed by context manager
- **Can be removed**: After updating all calling code

### episodic/cli.py
- `setup_readline()`: Empty function, no longer needed with prompt_toolkit
- **Can be removed**: Safe to remove

### episodic/server.py
- `broadcast_graph_update()`: Empty placeholder for future WebSocket implementation
- **Keep for now**: Placeholder for planned feature

## Test Files in Wrong Locations

### defunct/ directory
Contains old test files that should be removed:
- test_core.py
- test_db.py
- test_integration.py
- test_interactive_features.py
- test_native_visualization.py
- test_server.py

**Action**: Delete entire defunct/ directory

### scripts/ directory
Contains various test and analysis scripts that should be in tests/:
- test_*.py files
- Various analysis scripts

**Action**: These have been moved to proper test directories in PR #6

## Unused Variables

### episodic/cli.py
- Line 547: `hex_color` is assigned but never used
- **Fix**: Remove the variable or use it

### episodic/commands/model.py
- Line 123: `selected_provider` is assigned but never used
- **Fix**: Remove the variable

## Recommendations for Next Cleanup

1. **Remove deprecated commands** after migration period (target: v0.5.0)
2. **Delete ML stub implementations** if not planning to implement
3. **Remove defunct/ directory** entirely
4. **Clean up backward compatibility wrappers** after code migration
5. **Remove unused local variables** identified by linters

## Version History

- **v0.4.0** (current): Deprecated commands marked, backward compatibility maintained
- **v0.5.0** (planned): Remove deprecated commands and unused ML code
- **v0.6.0** (future): Remove backward compatibility wrappers