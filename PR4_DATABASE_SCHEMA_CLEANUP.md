# PR #4: Database Schema Cleanup

## Summary
Cleaned up database schema by renaming tables, adding indexes for performance, and creating a migration system.

## Changes Made

### Migration System
Created a proper database migration framework:
- **`episodic/migrations/__init__.py`**: Migration runner and base classes
- **`episodic/migrations/m004_schema_cleanup.py`**: First migration for schema cleanup
- **`episodic/run_migrations.py`**: CLI tool to run migrations manually

### Schema Changes (Migration 004)
1. **Renamed table**: `manual_index_scores` → `topic_detection_scores`
2. **Added column**: `detection_method` to track how scores were generated
3. **Added indexes** for better performance:
   - `idx_topic_scores_node` on user_node_short_id
   - `idx_topic_scores_boundary` on is_boundary
   - `idx_topics_boundaries` on topic boundaries
   - `idx_topics_name` on topic names
   - `idx_nodes_parent` on parent relationships
   - `idx_nodes_short_id` on short IDs
   - `idx_compressions_node` on compression nodes

### Backward Compatibility
- **`episodic/db_wrappers.py`**: Compatibility layer supporting both old and new table names
- Functions automatically detect which schema is in use
- All code updated to use new function names:
  - `store_manual_index_score` → `store_topic_detection_score`
  - `get_manual_index_scores` → `get_topic_detection_scores`
  - `clear_manual_index_scores` → `clear_topic_detection_scores`

### Documentation
- **`episodic/schema.sql`**: Complete schema documentation with examples
- Includes table descriptions, column purposes, and common queries

### Updated Files
- `episodic/topics/windows.py`: Use new wrapper functions
- `episodic/commands/index_topics.py`: Use new wrapper functions
- `episodic/db.py`: Auto-run migrations on initialization

## Benefits
1. **Better Performance**: Indexes speed up common queries
2. **Clearer Naming**: Table names better reflect their purpose
3. **Migration System**: Controlled schema evolution
4. **Backward Compatibility**: No breaking changes
5. **Better Documentation**: Clear schema reference

## Testing
```bash
# Run migrations manually
python episodic/run_migrations.py

# Or migrations run automatically on init
python -m episodic
> /init

# Verify schema
sqlite3 ~/.episodic/episodic.db ".schema topic_detection_scores"

# Test functionality still works
> /index 3
> /topics
```

## Migration Details
The migration:
- Safely renames tables preserving all data
- Adds the new `detection_method` column with default value
- Creates indexes only if they don't exist
- Can be rolled back if needed

## Next Steps
With the database schema cleaned up, we can proceed to PR #5: Command Consolidation.