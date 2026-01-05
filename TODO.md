# Episodic Technical Debt

## Background Topic Compression (disabled)

**Status**: Disabled
**Location**: `topic_management.py` ~line 460, `compression.py`, `db_compression.py`

The compression persistence layer was never completed:
- `compression.py:199` calls `store_compression_v2()` which doesn't exist
- `db_compression.py` defines `store_compression()` which has an incompatible signature and is never called
- The bare `except:` at `compression.py:222` swallowed the `NameError`, so this has been silently failing

To re-enable:
1. Design a coherent storage API (decide on signature, schema, metrics)
2. Implement it in `db_compression.py`
3. Update the call in `compression.py` to use the new API
4. Uncomment the call in `topic_management.py`
5. Add integration tests

Related files: `compression.py`, `db_compression.py`, `topic_management.py`
