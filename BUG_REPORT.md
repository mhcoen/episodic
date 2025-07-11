# Episodic CLI Bug Report - Post-Refactoring Issues

**Date**: July 10, 2025  
**Test Coverage**: ~60 commands tested  
**Pass Rate**: 38% (23 passed, 37 failed)

## Executive Summary

The recent refactoring has introduced significant breaking changes across the CLI. Major issues include:
- 9 commands are completely missing/unrecognized 
- 7 commands have import errors
- 6 commands have incorrect function signatures
- 1 command has missing object attributes
- Core functionality like `/model`, `/compression`, `/rag`, and `/muse` are broken

## Critical Issues (High Priority)

### 1. Model Command Completely Broken
- **Commands affected**: `/model`, `/model list`, `/model chat`, `/model detection`, `/model compression`, `/model synthesis`
- **Error**: `'module' object is not callable`
- **Root cause**: Likely importing a module instead of a function
- **Impact**: Users cannot change models - core functionality broken

### 2. RAG System Non-Functional
- **Commands affected**: `/rag`, `/rag on`, `/rag off`, `/index`, `/i`, `/docs`, `/docs list`
- **Errors**: 
  - `cannot import name 'rag' from 'episodic.commands.rag'`
  - `cannot import name 'index' from 'episodic.commands.rag'`
  - `cannot import name 'docs' from 'episodic.commands.rag'`
- **Root cause**: Functions were likely renamed or moved during refactoring
- **Impact**: Entire RAG/knowledge base system unusable

### 3. Compression System Broken
- **Commands affected**: `/compression`, `/compression stats`, `/compression queue`, `/compression api-stats`, `/compression reset-api`
- **Error**: `cannot import name 'compression' from 'episodic.commands.unified_compression'`
- **Root cause**: Incorrect import statement or missing function
- **Impact**: Cannot manage compression features

### 4. Web Search Configuration Broken
- **Commands affected**: `/websearch on`, `/websearch off`, `/websearch config`, `/websearch stats`, `/websearch cache clear`
- **Error**: `websearch() got an unexpected keyword argument 'action'`
- **Root cause**: Function signature changed but caller not updated
- **Note**: Basic web search (`/websearch query` and `/ws query`) still work

### 5. Muse Mode Missing
- **Commands affected**: `/muse`, `/muse on`, `/muse off`
- **Error**: `No module named 'episodic.commands.muse'`
- **Root cause**: Module not created or incorrectly referenced
- **Impact**: Perplexity-like search mode unavailable

## Medium Priority Issues

### 6. Missing Basic Commands
These commands are shown in help or expected but don't exist:
- `/h` (help shortcut)
- `/about`
- `/welcome` 
- `/config` (different from `/config-docs`)
- `/history`, `/history 5`, `/history all`
- `/tree`
- `/graph`
- `/summary`

### 7. Topic Compression Issue
- **Command**: `/topics compress`
- **Error**: `'TopicManager' object has no attribute 'get_current_topic'`
- **Root cause**: Method name changed or removed

### 8. Topic Scores Database Error
- **Command**: `/topics scores`
- **Error**: `Error binding parameter 1: type 'OptionInfo' is not supported`
- **Root cause**: Typer decorator issue when calling function

## Working Commands

The following commands are confirmed working:
- Help: `/help`, `/config-docs`
- Settings: `/set debug on/off`, `/set text_wrap on`, `/set show_costs on`, `/verify`, `/reset`
- Model parameters: `/mset`, `/mset chat`, `/mset chat.temperature 0.7`, `/mset detection.temperature 0`
- Topics (partial): `/topics`, `/topics list`, `/topics rename`, `/topics stats`, `/topics index 5`
- Search: `/search test`, `/s test`
- Web search (basic): `/websearch test query`, `/ws test query`
- Others: `/cost`, `--init`

## Recommendations

1. **Immediate Actions**:
   - Fix `/model` command - this is critical for basic functionality
   - Restore RAG system imports
   - Fix compression command imports
   - Add missing muse module

2. **Import/Module Issues** (most common problem):
   - Review all command imports in `cli_command_router.py`
   - Ensure functions exist with expected names in their modules
   - Check for module vs function import confusion

3. **Function Signature Issues**:
   - Update websearch command handler to match new function signature
   - Fix Typer decorator issues in topics scores

4. **Missing Commands**:
   - Either implement missing commands or remove from help/documentation
   - Add command aliases (like `/h` for `/help`)

5. **Testing**:
   - Add automated CLI command tests to prevent regression
   - Test all commands after any refactoring

## Technical Details

### Import Error Pattern
Most import errors follow this pattern:
```python
# In cli_command_router.py
from episodic.commands.module import function_name  # function_name doesn't exist
```

### Typer Decorator Pattern
Several issues stem from calling Typer-decorated functions directly:
```python
# Function with Typer decorators
def command(arg: str = typer.Argument(...)):
    ...

# Cannot call directly, need wrapper or different approach
command("value")  # Fails with OptionInfo/ArgumentInfo errors
```

### Action Parameter Pattern
websearch() function doesn't accept 'action' parameter but caller tries to pass it:
```python
# Caller expects
websearch(action="on")

# But function signature is likely
websearch(query: str, ...)
```

## File Locations for Fixes

- Command routing: `episodic/cli_command_router.py`
- Model commands: `episodic/commands/model.py` or similar
- RAG commands: `episodic/commands/rag.py`
- Compression: `episodic/commands/unified_compression.py`
- Web search: `episodic/commands/web_search.py`
- Topics: `episodic/commands/unified_topics.py`
- Muse mode: Need to create `episodic/commands/muse.py`

## Test Script

The test script used is saved as `test_all_commands.py` and can be re-run after fixes to verify resolution.