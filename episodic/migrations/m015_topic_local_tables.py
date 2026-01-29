"""
Migration: Add topic_nodes and topic_working_set tables for topic-local context assembly.

topic_nodes: Fast topic membership mapping for "last N exchanges in topic X" queries.
topic_working_set: Persistent topic state for year-later resume without full transcript.

These tables enable the hybrid/topic_local context recovery modes where resuming
topic A excludes topic B from the prompt entirely.
"""

import sqlite3
import logging
from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)

MIGRATION_ID = "m015_topic_local_tables"
MIGRATION_VERSION = 15


def migrate_up(conn: sqlite3.Connection = None):
    """Apply migration: create topic_nodes and topic_working_set tables."""
    
    def _apply(c: sqlite3.Connection):
        cursor = c.cursor()
        
        # Check if tables already exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_nodes'")
        if cursor.fetchone():
            logger.info("topic_nodes table already exists, skipping creation")
        else:
            # Create topic_nodes table
            # Purpose: fast topic membership mapping so topic-local prompt assembly can
            # fetch "last N exchanges in topic X" without scanning global ancestry.
            cursor.execute('''
                CREATE TABLE topic_nodes (
                    topic_start_node_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    turn_idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    PRIMARY KEY(topic_start_node_id, node_id),
                    FOREIGN KEY(topic_start_node_id) REFERENCES topics(start_node_id),
                    FOREIGN KEY(node_id) REFERENCES nodes(id)
                )
            ''')
            
            # Index for efficient "last N in topic X" queries ordered by turn
            cursor.execute('''
                CREATE INDEX idx_topic_nodes_turn 
                ON topic_nodes(topic_start_node_id, turn_idx)
            ''')
            
            # Index for reverse lookup: which topic(s) contain this node?
            cursor.execute('''
                CREATE INDEX idx_topic_nodes_node 
                ON topic_nodes(node_id)
            ''')
            
            logger.info("Created topic_nodes table")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_working_set'")
        if cursor.fetchone():
            logger.info("topic_working_set table already exists, skipping creation")
        else:
            # Create topic_working_set table
            # Purpose: persistent "restorable thread state" for a topic, enabling
            # resume of topic A a year later without needing the full transcript.
            cursor.execute('''
                CREATE TABLE topic_working_set (
                    topic_start_node_id TEXT PRIMARY KEY,
                    topic_name TEXT,
                    summary_md TEXT NOT NULL DEFAULT '',
                    decisions_json TEXT NOT NULL DEFAULT '[]',
                    open_loops_json TEXT NOT NULL DEFAULT '[]',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    last_summarized_turn_idx INTEGER,
                    last_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    summary_version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(topic_start_node_id) REFERENCES topics(start_node_id)
                )
            ''')
            
            # Index for finding recently updated topics (maintenance queries)
            cursor.execute('''
                CREATE INDEX idx_topic_ws_updated_at 
                ON topic_working_set(last_updated_at)
            ''')
            
            logger.info("Created topic_working_set table")
        
        c.commit()
        logger.info(f"Migration {MIGRATION_ID} applied successfully")
    
    if conn is not None:
        _apply(conn)
    else:
        with get_connection() as c:
            _apply(c)


def migrate_down(conn: sqlite3.Connection = None):
    """Rollback migration: drop topic_nodes and topic_working_set tables."""
    
    def _rollback(c: sqlite3.Connection):
        cursor = c.cursor()
        
        cursor.execute("DROP INDEX IF EXISTS idx_topic_nodes_turn")
        cursor.execute("DROP INDEX IF EXISTS idx_topic_nodes_node")
        cursor.execute("DROP TABLE IF EXISTS topic_nodes")
        logger.info("Dropped topic_nodes table")
        
        cursor.execute("DROP INDEX IF EXISTS idx_topic_ws_updated_at")
        cursor.execute("DROP TABLE IF EXISTS topic_working_set")
        logger.info("Dropped topic_working_set table")
        
        c.commit()
        logger.info(f"Migration {MIGRATION_ID} rolled back successfully")
    
    if conn is not None:
        _rollback(conn)
    else:
        with get_connection() as c:
            _rollback(c)


def backfill_topic_nodes(conn: sqlite3.Connection = None):
    """
    Backfill topic_nodes from existing topics table.
    
    For each topic, find all nodes between start_node_id and end_node_id
    (using rowid as turn_idx) and insert into topic_nodes.
    """
    
    def _backfill(c: sqlite3.Connection):
        cursor = c.cursor()
        
        # Get all topics
        cursor.execute("""
            SELECT start_node_id, end_node_id, name 
            FROM topics 
            ORDER BY created_at
        """)
        topics = cursor.fetchall()
        
        total_inserted = 0
        for start_node_id, end_node_id, topic_name in topics:
            # Get turn_idx for start node
            cursor.execute("SELECT rowid FROM nodes WHERE id = ?", (start_node_id,))
            row = cursor.fetchone()
            if not row:
                continue
            start_idx = row[0]
            
            # Get turn_idx for end node (or max if topic is open)
            if end_node_id:
                cursor.execute("SELECT rowid FROM nodes WHERE id = ?", (end_node_id,))
                row = cursor.fetchone()
                end_idx = row[0] if row else None
            else:
                cursor.execute("SELECT MAX(rowid) FROM nodes")
                end_idx = cursor.fetchone()[0]
            
            if end_idx is None:
                continue
            
            # Get all nodes in range and insert into topic_nodes
            cursor.execute("""
                SELECT id, rowid, role FROM nodes
                WHERE rowid >= ? AND rowid <= ?
                AND role IN ('user', 'assistant')
            """, (start_idx, end_idx))
            
            nodes_in_topic = cursor.fetchall()
            for node_id, turn_idx, role in nodes_in_topic:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO topic_nodes 
                        (topic_start_node_id, node_id, turn_idx, role)
                        VALUES (?, ?, ?, ?)
                    """, (start_node_id, node_id, turn_idx, role))
                    if cursor.rowcount > 0:
                        total_inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert node {node_id} into topic_nodes: {e}")
            
            # Also ensure topic_working_set has an entry (with empty summary initially)
            cursor.execute("""
                INSERT OR IGNORE INTO topic_working_set 
                (topic_start_node_id, topic_name)
                VALUES (?, ?)
            """, (start_node_id, topic_name))
        
        c.commit()
        logger.info(f"Backfilled {total_inserted} nodes into topic_nodes for {len(topics)} topics")
        return total_inserted
    
    if conn is not None:
        return _backfill(conn)
    else:
        with get_connection() as c:
            return _backfill(c)


if __name__ == "__main__":
    import sys
    
    action = sys.argv[1] if len(sys.argv) > 1 else "up"
    
    if action == "up":
        migrate_up()
        print(f"Applied migration: {MIGRATION_ID}")
    elif action == "down":
        migrate_down()
        print(f"Rolled back migration: {MIGRATION_ID}")
    elif action == "backfill":
        migrate_up()  # Ensure tables exist
        count = backfill_topic_nodes()
        print(f"Backfilled {count} nodes into topic_nodes")
    else:
        print(f"Usage: python {sys.argv[0]} [up|down|backfill]")
