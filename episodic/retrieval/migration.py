"""
FTS5 migration for lexical search.

Implements v1.1 spec section 4.
"""
import sqlite3
import logging
import re
import uuid

logger = logging.getLogger(__name__)


def migrate_fts5(conn: sqlite3.Connection) -> None:
    """
    Idempotent FTS5 migration with exclusive lock.
    
    Requirements per spec 4.3:
    - conn MUST be created with isolation_level=None (autocommit)
    - Uses BEGIN EXCLUSIVE for single-writer
    - DROP TRIGGER IF EXISTS before CREATE for idempotency
    - Backfills with 'rebuild' command
    
    Args:
        conn: Migration connection (isolation_level=None required)
    """
    cursor = conn.cursor()
    
    cursor.execute("BEGIN EXCLUSIVE")
    
    try:
        # Create FTS5 table if not exists
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                content,
                content='nodes',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )
        """)
        
        # Drop existing triggers for idempotency
        cursor.execute("DROP TRIGGER IF EXISTS nodes_fts_ai")
        cursor.execute("DROP TRIGGER IF EXISTS nodes_fts_ad")
        cursor.execute("DROP TRIGGER IF EXISTS nodes_fts_au")
        
        # Create insert trigger
        cursor.execute("""
            CREATE TRIGGER nodes_fts_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        
        # Create delete trigger
        cursor.execute("""
            CREATE TRIGGER nodes_fts_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content) 
                VALUES('delete', old.rowid, old.content);
            END
        """)
        
        # Create update trigger
        cursor.execute("""
            CREATE TRIGGER nodes_fts_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, content) 
                VALUES('delete', old.rowid, old.content);
                INSERT INTO nodes_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        
        # Backfill existing rows
        cursor.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        
        cursor.execute("COMMIT")
        logger.info("FTS5 migration completed successfully")
        
    except Exception as e:
        cursor.execute("ROLLBACK")
        logger.error(f"FTS5 migration failed: {e}")
        raise


def generate_temp_table_name() -> str:
    """Generate safe temp table name matching ^[a-zA-Z0-9_]+$."""
    suffix = uuid.uuid4().hex[:8]
    name = f"seg_filter_{suffix}"
    assert re.match(r'^[a-zA-Z0-9_]+$', name)
    return name
