# Compression and Topic Naming Fixes

## Issues Fixed

### 1. Compression Schema Error
**Problem**: The `compressions` table was missing a `content` column, causing compression jobs to fail with:
```
table compressions has no column named content
```

**Solution**: Added a database migration `migrate_compression_content()` that:
- Checks if the `content` column exists
- Adds it if missing using `ALTER TABLE`
- Runs automatically during `initialize_db()`

### 2. Topic Naming Issue
**Problem**: Topics were showing as "ongoing-XXXXXX" instead of meaningful names because:
- Topics are renamed when the NEXT topic is detected
- The LAST topic in a session remained with its placeholder name

**Solution**: Added `finalize_current_topic()` method that:
- Extracts a proper name for the current topic if it has a placeholder name
- Is called automatically when:
  - A script execution completes
  - User types `/exit` or `/quit`
  - User presses Ctrl+D

## Usage

The fixes are automatic, but you can also:

1. **Manually rename all placeholder topics**:
   ```
   /rename-topics
   ```

2. **Check if topics need renaming**:
   ```
   /topics
   ```
   Look for topics named "ongoing-XXXXXX"

## Technical Details

- Migration runs automatically on startup if needed
- Topic finalization uses the same extraction logic as regular topic changes
- Both fixes are backward compatible