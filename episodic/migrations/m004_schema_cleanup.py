"""
Migration 004: Database Schema Cleanup

This migration:
1. Renames manual_index_scores to topic_detection_scores
2. Adds detection_method column to track how the score was generated
3. Creates indexes for better query performance
4. Consolidates topic detection scoring tables
"""

import sqlite3
from episodic.migrations import Migration


class SchemaCleanupMigration(Migration):
    """Clean up and consolidate database schema."""
    
    def __init__(self):
        super().__init__(
            version=4,
            description="Schema cleanup - rename tables, add indexes, consolidate scoring"
        )
    
    def up(self, conn: sqlite3.Connection):
        """Apply the migration."""
        cursor = conn.cursor()
        
        # First check if manual_index_scores exists and topic_detection_scores doesn't
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='manual_index_scores'
        """)
        manual_exists = cursor.fetchone() is not None
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='topic_detection_scores'
        """)
        detection_exists = cursor.fetchone() is not None
        
        if manual_exists and not detection_exists:
            # SQLite doesn't support ALTER TABLE RENAME directly, so we need to recreate
            cursor.execute("""
                CREATE TABLE topic_detection_scores_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_node_short_id TEXT NOT NULL UNIQUE,
                    window_size INTEGER NOT NULL,
                    detection_method TEXT DEFAULT 'sliding_window',
                    
                    -- Window information
                    window_a_start_short_id TEXT,
                    window_a_end_short_id TEXT,
                    window_a_size INTEGER NOT NULL,
                    window_b_start_short_id TEXT,
                    window_b_end_short_id TEXT,
                    window_b_size INTEGER NOT NULL,
                    
                    -- Scores
                    drift_score REAL NOT NULL,
                    keyword_score REAL NOT NULL,
                    combined_score REAL NOT NULL,
                    
                    -- Detection result
                    is_boundary BOOLEAN NOT NULL,
                    transition_phrase TEXT,
                    
                    -- Metadata
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    threshold_used REAL
                )
            """)
            
            # Copy data from old table
            cursor.execute("""
                INSERT INTO topic_detection_scores_new 
                SELECT 
                    id, user_node_short_id, window_size, 'sliding_window',
                    window_a_start_short_id, window_a_end_short_id, window_a_size,
                    window_b_start_short_id, window_b_end_short_id, window_b_size,
                    drift_score, keyword_score, combined_score,
                    is_boundary, transition_phrase,
                    detection_timestamp, threshold_used
                FROM manual_index_scores
            """)
            
            # Drop old table
            cursor.execute("DROP TABLE manual_index_scores")
            
            # Rename new table
            cursor.execute("ALTER TABLE topic_detection_scores_new RENAME TO topic_detection_scores")
            
            # Commit the rename before creating indexes
            conn.commit()
        
        # Create indexes for better performance
        # Re-check tables after potential rename
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        indexes = [
            ("idx_topic_scores_node", "topic_detection_scores", "user_node_id"),
            ("idx_topic_scores_changed", "topic_detection_scores", "topic_changed"),
            ("idx_topics_boundaries", "topics", "start_node_id, end_node_id"),
            ("idx_topics_name", "topics", "name"),
            ("idx_nodes_parent", "nodes", "parent_id"),
            ("idx_nodes_short_id", "nodes", "short_id"),
            ("idx_compressions_node", "compressions", "compressed_node_id"),
        ]
        
        for index_name, table_name, columns in indexes:
            # Skip if table doesn't exist
            if table_name not in existing_tables:
                continue
                
            # Check if index already exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name=?
            """, (index_name,))
            
            if not cursor.fetchone():
                cursor.execute(f"CREATE INDEX {index_name} ON {table_name}({columns})")
        
        # Add detection_method to the hybrid scores table if it doesn't exist
        cursor.execute("PRAGMA table_info(topic_detection_scores)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'detection_method' not in columns and 'topic_detection_scores' in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            # Note: SQLite doesn't support adding columns with NOT NULL without default
            # The column is already in the new table structure above
            pass
        
        conn.commit()
    
    def down(self, conn: sqlite3.Connection):
        """Rollback the migration."""
        cursor = conn.cursor()
        
        # Check if we need to rename back
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='topic_detection_scores'
        """)
        if cursor.fetchone():
            # Recreate the original table structure
            cursor.execute("""
                CREATE TABLE manual_index_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_node_short_id TEXT NOT NULL UNIQUE,
                    window_size INTEGER NOT NULL,
                    
                    window_a_start_short_id TEXT,
                    window_a_end_short_id TEXT,
                    window_a_size INTEGER NOT NULL,
                    window_b_start_short_id TEXT,
                    window_b_end_short_id TEXT,
                    window_b_size INTEGER NOT NULL,
                    
                    drift_score REAL NOT NULL,
                    keyword_score REAL NOT NULL,
                    combined_score REAL NOT NULL,
                    
                    is_boundary BOOLEAN NOT NULL,
                    transition_phrase TEXT,
                    
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    threshold_used REAL
                )
            """)
            
            # Copy data back
            cursor.execute("""
                INSERT INTO manual_index_scores 
                SELECT 
                    id, user_node_short_id, window_size,
                    window_a_start_short_id, window_a_end_short_id, window_a_size,
                    window_b_start_short_id, window_b_end_short_id, window_b_size,
                    drift_score, keyword_score, combined_score,
                    is_boundary, transition_phrase,
                    detection_timestamp, threshold_used
                FROM topic_detection_scores
            """)
            
            # Drop the new table
            cursor.execute("DROP TABLE topic_detection_scores")
        
        # Drop indexes
        indexes = [
            "idx_topic_scores_node",
            "idx_topic_scores_boundary", 
            "idx_topics_boundaries",
            "idx_topics_name",
            "idx_nodes_parent",
            "idx_nodes_short_id",
            "idx_compressions_node"
        ]
        
        for index_name in indexes:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
        
        conn.commit()


# Export the migration instance
migration = SchemaCleanupMigration()