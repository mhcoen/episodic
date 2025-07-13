# Configuration System Refactoring

## Problem
The system had two separate configuration systems storing model information:
1. `episodic.config` storing in `config["model"]` (SQLite database)
2. `llm_config.py` storing in `llm_config["default_model"]` (JSON file at ~/.episodic/llm_config.json)

This dual-write pattern caused synchronization issues where the startup display showed a different model than the `/model` command.

## Solution
Made `episodic.config` the single source of truth for model selection:

### Changes Made

1. **Refactored `llm_config.py`**:
   - Removed all state management functions (`get_default_model`, `set_default_model`, `load_config`, `save_config`)
   - Removed JSON file handling (`~/.episodic/llm_config.json` is no longer used)
   - Converted to a stateless module that only provides provider information
   - Added `validate_model_selection()` to check if a model exists and has API key
   - Added `find_provider_for_model()` to determine which provider offers a model
   - `get_current_provider()` now derives the provider from the selected model

2. **Updated `commands/model.py`**:
   - Removed the dual-write pattern (no more `set_default_model()` call)
   - Now only writes to `config.set("model", name)`
   - Uses `validate_model_selection()` before setting the model
   - Removed import of `get_default_model` and `set_default_model`

3. **Updated `cli_display.py`**:
   - Removed dependency on `get_default_model()`
   - Now uses `config.get("model", "gpt-3.5-turbo")` directly
   - Consistent with what `/model` command shows

4. **Updated `config_defaults.py`**:
   - Set default model to `"gpt-3.5-turbo"` in MODEL_SELECTION_DEFAULTS
   - Updated documentation to reflect the default

## Benefits
1. **Single source of truth**: No more synchronization bugs
2. **Cleaner architecture**: Clear separation between user preferences (config) and provider capabilities (llm_config)
3. **Simpler code**: Removed redundant state management code
4. **Backward compatible**: Most code already used `config["model"]`

## Migration Notes
- The old `~/.episodic/llm_config.json` file is now ignored and can be safely deleted
- All model selection is now stored in the SQLite database via `episodic.config`
- Provider information is hardcoded in `llm_config.py` as static data

## Testing
The refactored system correctly:
- Shows the same model in startup display and `/model` command
- Validates model existence and API key availability
- Determines the correct provider for each model
- Maintains all existing functionality