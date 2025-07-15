# Obsolete Tests Analysis

This document identifies tests that are obsolete after the recent refactoring and should be removed or updated.

## Test Files with Obsolete Tests

### 1. `/tests/test_db.py`
**Obsolete Tests:**
- `tearDown()` method (line 37-38): Uses `close_connection()` which was removed
  - The function was a no-op and has been deleted from the codebase
  - Should remove this line from tearDown

**Action Required:**
- Remove line 38: `close_connection()` 
- Remove import of `close_connection` from line 14

### 2. `/tests/test_server.py`
**Obsolete Tests:**
- `tearDown()` method (line 54-55): Uses `close_connection()` which was removed
- Import on line 19 includes `close_connection`

**Action Required:**
- Remove line 55: `close_connection()`
- Remove `close_connection` from imports on line 19

### 3. `/tests/test_cli.py`
**Obsolete Tests (all marked with @unittest.skip):**
- `test_version_command` (lines 116-121): Version command was removed
- `test_providers_command` (lines 123-131): Providers command was removed  
- `test_initialize_prompt` (lines 186-191): `_initialize_prompt` was removed
- `test_initialize_model` (lines 193-199): `_initialize_model` was removed
- `test_display_session_summary` (lines 261-265): `display_session_summary` was removed

**Action Required:**
- These tests are already skipped, can be deleted entirely
- Remove the skipped test methods

### 4. `/tests/integration/cli/test_all_commands.py`
**Potentially Obsolete Commands Being Tested:**
- Line 191: `("/rename-topics", "Rename topics (deprecated)")`
- Line 192: `("/compress-current-topic", "Compress current topic (deprecated)")`
- Line 180: `("/api-stats", "Show API statistics")`
- Line 181: `("/reset-api-stats", "Reset API statistics")`
- Lines 155-161: All `/websearch` commands (websearch was simplified to just `/web`)

**Action Required:**
- These deprecated commands should still be tested to ensure backward compatibility warnings work
- However, the `/websearch` commands should be updated to `/web` commands
- Update test descriptions to indicate they test deprecated functionality

## Test Files That Are Still Valid

### 1. `/tests/test_caching.py`
- All tests appear valid - caching functionality still exists
- No obsolete tests found

### 2. `/tests/unit/commands/test_unified_commands.py`
- Tests the new unified command structure
- Includes backward compatibility tests for deprecated commands
- All tests appear valid and necessary

### 3. `/tests/unit/commands/test_topics_command.py`
- Simple direct test of topics command
- Still valid

## Summary of Required Actions

1. **Remove `close_connection()` calls and imports from:**
   - `/tests/test_db.py` (line 14 import, line 38 call)
   - `/tests/test_server.py` (line 19 import, line 55 call)

2. **Delete skipped test methods from `/tests/test_cli.py`:**
   - `test_version_command`
   - `test_providers_command`
   - `test_initialize_prompt`
   - `test_initialize_model`
   - `test_display_session_summary`

3. **Update `/tests/integration/cli/test_all_commands.py`:**
   - Change `/websearch` commands to `/web` commands
   - Keep deprecated command tests but update descriptions

## Additional Test Files Reviewed

### 5. `/tests/integration/test_web_search_integration.py`
- Tests the web search functionality
- All tests appear valid - web search still exists
- No obsolete tests found

### 6. `/tests/integration/cli/test_cli_commands.py`
- May contain references to `/websearch` commands that should be updated to `/web`
- Should be reviewed for consistency with new command structure

## Notes on Deprecated Commands

The following deprecated commands are still functional with warnings:
- `/rename-topics` → `/topics rename`
- `/compress-current-topic` → `/topics compress`
- `/api-stats` → `/compression api-stats`
- `/reset-api-stats` → `/compression reset-api`

These should continue to be tested to ensure backward compatibility.

## Command Changes to Note

### Web Search Commands
The web search commands were simplified:
- Old: `/websearch`, `/websearch on/off`, `/websearch config`, etc.
- New: `/web`, `/web provider <name>`, `/web list`, `/web reset`
- The `/muse` command handles synthesis mode separately

## Database Schema Changes

No tests were found for the old `manual_index_scores` table name (now `topic_detection_scores`), so no schema-related test updates are needed.

## Removed Features Without Tests

The following removed features had no tests found:
- Old command structures before unification
- `manual_index_scores` table (renamed to `topic_detection_scores`)
- Empty exception blocks that were fixed
- Duplicate `debug_print()` implementations