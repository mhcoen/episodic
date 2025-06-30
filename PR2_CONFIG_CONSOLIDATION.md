# PR #2: Configuration Consolidation

## Summary
Centralized all default configuration values and removed hardcoded defaults throughout the codebase.

## Changes Made

### Created Central Configuration Defaults
- **`episodic/config_defaults.py`**: New file containing:
  - All default configuration values in one place
  - Organized by category (Core, Topic Detection, Streaming, Model Parameters)
  - Documentation for each configuration parameter
  - Dynamic threshold behavior constants

### Updated Configuration System
- **`episodic/config.py`**:
  - Import and use DEFAULT_CONFIG from config_defaults
  - Simplified _load() method to use centralized defaults
  - Added get_doc() method to retrieve parameter documentation
  - Added list_all() method to list all configs with documentation
  - Simplified reset logic for model parameters

### Removed Hardcoded Defaults
Updated all files to remove hardcoded default values in config.get() calls:
- **`episodic/conversation.py`**: Removed defaults for debug, automatic_topic_detection, text_wrap, show_drift, min_messages_before_topic_change, use_hybrid_topic_detection
- **`episodic/topics.py`**: Now uses TOPIC_THRESHOLD_BEHAVIOR constants
- **`episodic/compression.py`**: Removed debug default
- **`episodic/cli.py`**: Removed debug and auto_compress_topics defaults

### Added Configuration Documentation Command
- **`episodic/commands/settings.py`**: Added config_docs() function
- **`episodic/commands/__init__.py`**: Export config_docs
- **`episodic/cli.py`**: Added /config-docs command handler
- **`episodic/commands/utility.py`**: Updated help to include new commands

## Benefits
1. **Single Source of Truth**: All defaults in one file
2. **Better Documentation**: Each parameter has associated documentation
3. **Easier Maintenance**: No more searching for hardcoded values
4. **Consistency**: All config.get() calls now use the same defaults
5. **Discoverability**: New /config-docs command shows all available settings

## Testing
```bash
# Test configuration loading
python -m episodic
> /set
> /config-docs
> /model-params

# Test that defaults work correctly
> /set automatic_topic_detection false
> /set automatic_topic_detection true

# Verify no regression in functionality
> /init
> Hello, how are you?
> /topics
```

## Next Steps
With configuration consolidated, we can now proceed to PR #3: Test Organization.