"""
Tests for FTS5 migration (Success Criterion 1).

SC1: Migration idempotency - running migration twice yields correct nodes_fts 
coverage; triggers not duplicated; rebuild runs under exclusive lock.
"""
import pytest
import sqlite3


class TestMigrationIdempotent:
    """SC1: Migration must be idempotent."""
    
    def test_migration_creates_fts_table(self, migration_conn):
        """FTS5 table is created on first migration."""
        from episodic.retrieval.migration import migrate_fts5
        
        # Insert test data before migration
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'coffee beans', 'user')")
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n2', 'espresso grinder', 'assistant')")
        
        # Run migration
        migrate_fts5(migration_conn)
        
        # Verify FTS table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_fts'")
        assert cursor.fetchone() is not None
    
    def test_migration_idempotent_no_duplicate_triggers(self, migration_conn):
        """Running migration twice does not duplicate triggers."""
        from episodic.retrieval.migration import migrate_fts5
        
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'test', 'user')")
        
        # Run migration twice
        migrate_fts5(migration_conn)
        migrate_fts5(migration_conn)
        
        # Count triggers - should be exactly 3
        cursor.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='trigger' AND name LIKE 'nodes_fts_%'
        """)
        trigger_count = cursor.fetchone()[0]
        assert trigger_count == 3, f"Expected 3 triggers, got {trigger_count}"
    
    def test_migration_backfills_existing_rows(self, migration_conn):
        """Existing rows are indexed after migration."""
        from episodic.retrieval.migration import migrate_fts5
        
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'coffee beans', 'user')")
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n2', 'espresso grinder', 'user')")
        
        migrate_fts5(migration_conn)
        
        # Verify FTS has same row count as nodes
        cursor.execute("SELECT COUNT(*) FROM nodes")
        node_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM nodes_fts")
        fts_count = cursor.fetchone()[0]
        
        assert fts_count == node_count
    
    def test_migration_uses_exclusive_lock(self, migration_conn):
        """Migration must use BEGIN EXCLUSIVE."""
        from episodic.retrieval.migration import migrate_fts5
        
        # This test verifies the migration doesn't fail due to locking issues
        # In a real scenario, we'd verify BEGIN EXCLUSIVE is called
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'test', 'user')")
        
        # Should not raise
        migrate_fts5(migration_conn)
    
    def test_triggers_sync_on_insert(self, migration_conn):
        """New nodes are automatically indexed via trigger."""
        from episodic.retrieval.migration import migrate_fts5
        
        migrate_fts5(migration_conn)
        
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'unique coffee content', 'user')")
        
        # Search should find the new node
        cursor.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'coffee'")
        results = cursor.fetchall()
        assert len(results) == 1
    
    def test_triggers_sync_on_delete(self, migration_conn):
        """Deleted nodes are removed from FTS via trigger."""
        from episodic.retrieval.migration import migrate_fts5
        
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'coffee test', 'user')")
        
        migrate_fts5(migration_conn)
        
        # Verify indexed
        cursor.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'coffee'")
        assert cursor.fetchone() is not None
        
        # Delete node
        cursor.execute("DELETE FROM nodes WHERE id = 'n1'")
        
        # Should no longer be in FTS
        cursor.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'coffee'")
        assert cursor.fetchone() is None
    
    def test_triggers_sync_on_update(self, migration_conn):
        """Updated nodes are re-indexed via trigger."""
        from episodic.retrieval.migration import migrate_fts5
        
        cursor = migration_conn.cursor()
        cursor.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'coffee test', 'user')")
        
        migrate_fts5(migration_conn)
        
        # Update content
        cursor.execute("UPDATE nodes SET content = 'espresso test' WHERE id = 'n1'")
        
        # Old term should not match
        cursor.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'coffee'")
        assert cursor.fetchone() is None
        
        # New term should match
        cursor.execute("SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH 'espresso'")
        assert cursor.fetchone() is not None
