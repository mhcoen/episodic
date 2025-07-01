# PR #5: Command Consolidation

## Summary
Consolidated related commands into unified interfaces and created a command registry for better organization and discoverability. The new system maintains full backward compatibility while providing a cleaner, more intuitive command structure.

## Changes Made

### Unified Commands
Created unified command interfaces that group related functionality:

1. **`unified_topics.py`**: Consolidated topic commands
   - `/topics` - List all topics (default)
   - `/topics rename` - Rename ongoing topics
   - `/topics compress` - Compress current topic
   - `/topics index <n>` - Manual topic detection
   - `/topics scores` - Show detection scores
   - `/topics stats` - Topic statistics

2. **`unified_compression.py`**: Consolidated compression commands
   - `/compression` - Show stats (default)
   - `/compression queue` - Show pending compressions
   - `/compression compress` - Compress topic/branch
   - `/compression api-stats` - API usage statistics
   - `/compression reset-api` - Reset API stats

3. **`unified_settings.py`**: Consolidated settings commands
   - `/settings` - Show all settings (default)
   - `/settings set` - Set configuration parameter
   - `/settings verify` - Verify configuration
   - `/settings cost` - Show session costs
   - `/settings params` - Model parameters
   - `/settings docs` - Configuration documentation

### Command Registry System
- **`registry.py`**: Centralized command registration
  - Tracks commands by category
  - Handles aliases and deprecation
  - Provides command discovery
  - Supports deprecation warnings

- **`cli_registry.py`**: Enhanced command handling
  - Uses registry for command lookup
  - Shows deprecation warnings
  - Organized help by categories

### Backward Compatibility
- All old commands still work
- Deprecated commands show warnings suggesting replacements
- Examples:
  ```
  /rename-topics → Use /topics rename
  /compress-current-topic → Use /topics compress
  /api-stats → Use /compression api-stats
  ```

### Improved Help System
- Commands organized by category:
  - 🧭 Navigation
  - 💬 Conversation
  - 📑 Topics
  - ⚙️ Configuration
  - 📦 Compression
  - 🛠️ Utility
- Deprecated commands shown separately
- Clear migration path for users

## Benefits
1. **Better Organization**: Related commands grouped together
2. **Easier Discovery**: Unified commands with subactions
3. **Smooth Migration**: Deprecation warnings guide users
4. **Cleaner CLI**: Fewer top-level commands
5. **Future-Proof**: Easy to add new subcommands

## Testing
```bash
# Test new unified commands
python -m episodic
> /topics              # List topics
> /topics rename       # Rename ongoing topics
> /topics stats        # Show statistics

> /compression         # Show compression stats
> /compression queue   # Show pending compressions

> /settings           # Show all settings
> /settings docs      # Show documentation

# Test deprecation warnings
> /rename-topics      # Should show warning
> /api-stats         # Should show warning

# Test help system
> /help              # Should show categorized help
```

## Migration Guide for Users

### Topic Commands
- `/topics` → `/topics` (unchanged)
- `/rename-topics` → `/topics rename`
- `/compress-current-topic` → `/topics compress`
- `/index <n>` → `/topics index <n>`
- `/topic-scores` → `/topics scores`

### Compression Commands
- `/compression-stats` → `/compression stats`
- `/compression-queue` → `/compression queue`
- `/api-stats` → `/compression api-stats`
- `/reset-api-stats` → `/compression reset-api`

### Settings Commands
- `/set` → `/set` or `/settings set`
- `/verify` → `/verify` or `/settings verify`
- `/cost` → `/cost` or `/settings cost`
- `/model-params` → `/model-params` or `/settings params`
- `/config-docs` → `/config-docs` or `/settings docs`

## Next Steps
With commands consolidated, we can proceed to PR #6: Test Infrastructure.