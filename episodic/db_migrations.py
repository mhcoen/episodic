"""
Database migration operations for Episodic.

This module handles database initialization and migrations.
"""

import os
import sqlite3
import logging

from .db_connection import get_connection, get_db_path, database_exists
from .db_ids import generate_short_id
from .db_nodes import insert_node

# Set up logging
logger = logging.getLogger(__name__)


def initialize_db(erase=False, create_root_node=True, migrate=True):
    """
    Initialize the database with required tables.
    
    Args:
        erase: If True, delete existing database before creating
        create_root_node: If True, create an initial root node
        migrate: If True, run migrations after initialization
    """
    db_path = get_db_path()
    
    # If erase is True, delete the existing database
    if erase and os.path.exists(db_path):
        # First close any connection pool
        from .db_connection import close_pool
        close_pool()
        
        # Then remove the database file
        os.remove(db_path)
        logger.info(f"Deleted existing database at {db_path}")
    
    with get_connection() as conn:
        c = conn.cursor()
        
        # Create the main nodes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                short_id TEXT UNIQUE,
                role TEXT,
                provider TEXT,
                model TEXT,
                FOREIGN KEY (parent_id) REFERENCES nodes(id)
            )
        ''')
        
        # Create the state table for storing the current head
        c.execute('''
            CREATE TABLE IF NOT EXISTS state (
                name TEXT PRIMARY KEY,
                head_id TEXT,
                FOREIGN KEY (head_id) REFERENCES nodes(id)
            )
        ''')
        
        # Create the compressions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS compressions (
                compressed_node_id TEXT PRIMARY KEY,
                original_branch_head TEXT NOT NULL,
                compressed_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (compressed_node_id) REFERENCES nodes(id),
                FOREIGN KEY (original_branch_head) REFERENCES nodes(id)
            )
        ''')
        
        # Create the compressions_v2 table
        c.execute('''
            CREATE TABLE IF NOT EXISTS compressions_v2 (
                compressed_node_id TEXT PRIMARY KEY,
                original_branch_head TEXT NOT NULL,
                compressed_content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (compressed_node_id) REFERENCES nodes(id),
                FOREIGN KEY (original_branch_head) REFERENCES nodes(id)
            )
        ''')
        
        # Create the compression_nodes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS compression_nodes (
                compression_id TEXT NOT NULL,
                original_node_id TEXT NOT NULL,
                PRIMARY KEY (compression_id, original_node_id),
                FOREIGN KEY (compression_id) REFERENCES compressions_v2(compressed_node_id),
                FOREIGN KEY (original_node_id) REFERENCES nodes(id)
            )
        ''')
        
        # Create the topics table with nullable end_node_id
        c.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                start_node_id TEXT NOT NULL,
                end_node_id TEXT,
                confidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (start_node_id) REFERENCES nodes(id),
                FOREIGN KEY (end_node_id) REFERENCES nodes(id)
            )
        ''')
        
        # Create the manual_index_scores table
        c.execute('''
            CREATE TABLE IF NOT EXISTS manual_index_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_node_short_id TEXT NOT NULL,
                window_size INTEGER NOT NULL,
                window_a_start_short_id TEXT NOT NULL,
                window_a_end_short_id TEXT NOT NULL,
                window_a_size INTEGER NOT NULL,
                window_b_start_short_id TEXT NOT NULL,
                window_b_end_short_id TEXT NOT NULL,
                window_b_size INTEGER NOT NULL,
                drift_score REAL NOT NULL,
                keyword_score REAL DEFAULT 0.0,
                combined_score REAL DEFAULT 0.0,
                is_boundary BOOLEAN DEFAULT FALSE,
                transition_phrase TEXT,
                threshold_used REAL DEFAULT 0.9,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create the topic_detection_scores table
        c.execute('''
            CREATE TABLE IF NOT EXISTS topic_detection_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_node_short_id TEXT NOT NULL,
                detection_method TEXT NOT NULL,
                current_topic TEXT,
                messages_in_topic INTEGER DEFAULT 0,
                drift_score REAL DEFAULT 0.0,
                keyword_score REAL DEFAULT 0.0,
                combined_score REAL DEFAULT 0.0,
                effective_threshold REAL DEFAULT 0.9,
                topic_changed BOOLEAN DEFAULT FALSE,
                detection_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create the configuration table
        c.execute('''
            CREATE TABLE IF NOT EXISTS configuration (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create the conversations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                root_node_id TEXT,
                current_head_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                FOREIGN KEY (root_node_id) REFERENCES nodes(id),
                FOREIGN KEY (current_head_id) REFERENCES nodes(id)
            )
        ''')
        
        # Create indices for better performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_nodes_short_id ON nodes(short_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_topics_start ON topics(start_node_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_topics_end ON topics(end_node_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_manual_scores_node ON manual_index_scores(user_node_short_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_detection_scores_node ON topic_detection_scores(user_node_short_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_detection_scores_created ON topic_detection_scores(created_at)')
        
        # Initialize the head pointer if it doesn't exist
        c.execute("INSERT OR IGNORE INTO state (name, head_id) VALUES ('head', NULL)")
        
        # Create root node if requested and database is new
        if create_root_node and not database_exists():
            root_id, root_short_id = insert_node("", None, role="system")
            logger.info(f"Created root node with ID {root_id} (short: {root_short_id})")
        
        conn.commit()
        
    # Run migrations if requested
    if migrate:
        migrate_to_short_ids()
        migrate_to_provider_model()
        migrate_topics_nullable_end()
        migrate_to_roles()
        migrate_to_meta_query_flag()
        migrate_to_topic_centroids()
        migrate_to_topic_local_tables()
        migrate_to_context_assembly_debug()
        migrate_to_reactivation_decisions()
        migrate_to_node_source_type()

    logger.info("Database initialized successfully")


def migrate_to_short_ids():
    """Add short_id column if it doesn't exist and populate it."""
    with get_connection() as conn:
        c = conn.cursor()
        
        # Check if short_id column exists
        c.execute("PRAGMA table_info(nodes)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'short_id' not in columns:
            # Add the column
            c.execute("ALTER TABLE nodes ADD COLUMN short_id TEXT")
            
            # Generate short IDs for existing nodes
            c.execute("SELECT id FROM nodes WHERE short_id IS NULL")
            nodes_to_update = c.fetchall()
            
            for (node_id,) in nodes_to_update:
                short_id = generate_short_id()
                c.execute("UPDATE nodes SET short_id = ? WHERE id = ?", (short_id, node_id))
            
            # Create unique index
            try:
                c.execute("CREATE UNIQUE INDEX idx_nodes_short_id_unique ON nodes(short_id)")
            except sqlite3.OperationalError:
                # Index might already exist
                pass
                
            conn.commit()
            logger.info(f"Migrated {len(nodes_to_update)} nodes to have short IDs")


def migrate_to_provider_model():
    """Add provider and model columns if they don't exist."""
    with get_connection() as conn:
        c = conn.cursor()
        
        # Check existing columns
        c.execute("PRAGMA table_info(nodes)")
        columns = [column[1] for column in c.fetchall()]
        
        # Add provider column if missing
        if 'provider' not in columns:
            c.execute("ALTER TABLE nodes ADD COLUMN provider TEXT")
            logger.info("Added provider column to nodes table")
        
        # Add model column if missing
        if 'model' not in columns:
            c.execute("ALTER TABLE nodes ADD COLUMN model TEXT")
            logger.info("Added model column to nodes table")
            
        conn.commit()


def migrate_topics_nullable_end():
    """Make end_node_id nullable in topics table."""
    with get_connection() as conn:
        c = conn.cursor()
        
        # Check if topics table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topics'")
        if not c.fetchone():
            # No topics table, nothing to migrate
            return
            
        # Get current schema
        c.execute("PRAGMA table_info(topics)")
        columns = c.fetchall()
        
        # Check if end_node_id is already nullable
        for col in columns:
            if col[1] == 'end_node_id' and col[3] == 0:  # col[3] is "notnull"
                # Already nullable
                return
                
        # Need to recreate table with nullable end_node_id
        # SQLite doesn't support ALTER COLUMN, so we need to recreate
        
        # Create new table with correct schema
        c.execute('''
            CREATE TABLE IF NOT EXISTS topics_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                start_node_id TEXT NOT NULL,
                end_node_id TEXT,
                confidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (start_node_id) REFERENCES nodes(id),
                FOREIGN KEY (end_node_id) REFERENCES nodes(id)
            )
        ''')
        
        # Copy data
        c.execute("""
            INSERT INTO topics_new (id, name, start_node_id, end_node_id, confidence, created_at)
            SELECT id, name, start_node_id, end_node_id, 
                   CASE WHEN confidence IS NULL THEN NULL ELSE confidence END,
                   created_at
            FROM topics
        """)
        
        # Drop old table and rename new one
        c.execute("DROP TABLE topics")
        c.execute("ALTER TABLE topics_new RENAME TO topics")
        
        # Recreate indices
        c.execute('CREATE INDEX IF NOT EXISTS idx_topics_start ON topics(start_node_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_topics_end ON topics(end_node_id)')
        
        conn.commit()
        logger.info("Migrated topics table to have nullable end_node_id")


def migrate_to_roles():
    """Add role column to nodes table if it doesn't exist."""
    with get_connection() as conn:
        c = conn.cursor()
        
        # Check if role column exists
        c.execute("PRAGMA table_info(nodes)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'role' not in columns:
            # Add the role column
            c.execute("ALTER TABLE nodes ADD COLUMN role TEXT")
            
            # Set default roles for existing nodes
            # This is a simple heuristic: nodes with content are 'user', 
            # nodes with only response content are 'assistant'
            c.execute("""
                UPDATE nodes 
                SET role = CASE 
                    WHEN content IS NOT NULL AND content != '' THEN 'user'
                    ELSE 'assistant'
                END
                WHERE role IS NULL
            """)
            
            conn.commit()
            logger.info("Added role column to nodes table and set default values")


def migrate_to_meta_query_flag():
    """Add is_meta_query column to nodes table for filtering out meta-queries from retrieval."""
    with get_connection() as conn:
        c = conn.cursor()

        # Check if is_meta_query column exists
        c.execute("PRAGMA table_info(nodes)")
        columns = [column[1] for column in c.fetchall()]

        if 'is_meta_query' not in columns:
            # Add the is_meta_query column with default FALSE
            c.execute("ALTER TABLE nodes ADD COLUMN is_meta_query BOOLEAN DEFAULT FALSE")
            conn.commit()
            logger.info("Added is_meta_query column to nodes table")


def migrate_to_topic_centroids():
    """Add topic_centroids table for implicit topic reactivation.

    This table stores centroid/medoid information for each topic, enabling
    efficient ANN queries to find which dormant topic a new message relates to.
    """
    with get_connection() as conn:
        c = conn.cursor()

        # Check if topic_centroids table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_centroids'")
        if c.fetchone():
            # Table already exists
            return

        # Create the topic_centroids table
        c.execute('''
            CREATE TABLE topic_centroids (
                start_node_id TEXT PRIMARY KEY,
                centroid_medoid_exchange_id TEXT,
                exchange_count INTEGER DEFAULT 0,
                last_active_turn_idx INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (start_node_id) REFERENCES nodes(id),
                FOREIGN KEY (centroid_medoid_exchange_id) REFERENCES nodes(id)
            )
        ''')

        # Create index for efficient turn_idx queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_topic_centroids_turn ON topic_centroids(last_active_turn_idx)')

        conn.commit()
        logger.info("Created topic_centroids table for implicit topic reactivation")


def migrate_to_topic_local_tables():
    """Add topic_nodes and topic_working_set tables for topic-local context assembly.

    topic_nodes: Fast topic membership mapping for "last N exchanges in topic X" queries.
    topic_working_set: Persistent topic state for year-later resume without full transcript.
    """
    from .migrations.m015_topic_local_tables import migrate_up
    migrate_up()


def migrate_to_context_assembly_debug():
    """Add context_assembly_debug table for instrumentation.

    Stores debug info from each context assembly operation keyed by user_node_id.
    """
    from .migrations.m016_context_assembly_debug import migrate, is_applied
    with get_connection() as conn:
        if not is_applied(conn):
            migrate(conn)
            logger.info("Created context_assembly_debug table for instrumentation")


def migrate_to_reactivation_decisions():
    """Add reactivation_decisions and reactivation_labels tables for probe calibration.

    reactivation_decisions: Stores detailed feature info for every probe decision.
    reactivation_labels: Stores human-labeled ground truth for calibration.
    """
    from .migrations.m017_reactivation_decisions import migrate as migrate_decisions
    from .migrations.m018_reactivation_labels import migrate as migrate_labels

    with get_connection() as conn:
        cursor = conn.cursor()

        # Check if reactivation_decisions table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='reactivation_decisions'
        """)
        if not cursor.fetchone():
            migrate_decisions(conn)
            logger.info("Created reactivation_decisions table for probe calibration")

        # Check if reactivation_labels table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='reactivation_labels'
        """)
        if not cursor.fetchone():
            migrate_labels(conn)
            logger.info("Created reactivation_labels table for ground truth")


def migrate_to_node_source_type():
    """Add source_type column to nodes table for muse security provenance tracking."""
    with get_connection() as conn:
        c = conn.cursor()

        c.execute("PRAGMA table_info(nodes)")
        columns = [column[1] for column in c.fetchall()]

        if 'source_type' not in columns:
            c.execute("ALTER TABLE nodes ADD COLUMN source_type TEXT DEFAULT 'chat'")
            conn.commit()
            logger.info("Added source_type column to nodes table")