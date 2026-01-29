"""
Migration: Add topic_centroids table for implicit reactivation.

Stores per-topic centroid embeddings for fast similarity lookup.
"""

MIGRATION_ID = "add_topic_centroids"
MIGRATION_VERSION = 1

UP_SQL = """
CREATE TABLE IF NOT EXISTS topic_centroids (
    topic_id INTEGER PRIMARY KEY,
    centroid_mean BLOB NOT NULL,              -- Incremental mean embedding (float32 array)
    centroid_medoid_exchange_id TEXT,         -- Node ID of most central exchange
    exchange_count INTEGER NOT NULL DEFAULT 0,
    last_activity_ts TIMESTAMP,               -- Always updated on new exchange
    last_active_exchange_id TEXT,             -- For dormancy calculation
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES topics(id),
    FOREIGN KEY(centroid_medoid_exchange_id) REFERENCES nodes(id),
    FOREIGN KEY(last_active_exchange_id) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_topic_centroids_last_activity 
    ON topic_centroids(last_activity_ts);
"""

DOWN_SQL = """
DROP INDEX IF EXISTS idx_topic_centroids_last_activity;
DROP TABLE IF EXISTS topic_centroids;
"""


def migrate_up(conn):
    """Apply migration."""
    conn.executescript(UP_SQL)
    conn.commit()


def migrate_down(conn):
    """Rollback migration."""
    conn.executescript(DOWN_SQL)
    conn.commit()


if __name__ == "__main__":
    import sys
    from episodic.db_connection import get_connection
    
    action = sys.argv[1] if len(sys.argv) > 1 else "up"
    
    with get_connection() as conn:
        if action == "up":
            migrate_up(conn)
            print(f"Applied migration: {MIGRATION_ID}")
        elif action == "down":
            migrate_down(conn)
            print(f"Rolled back migration: {MIGRATION_ID}")
        else:
            print(f"Usage: python {sys.argv[0]} [up|down]")
