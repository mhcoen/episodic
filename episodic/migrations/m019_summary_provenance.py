"""
Migration: Add comprehensive provenance fields to topic_working_set.

These fields enable:
- Schema versioning for summary format evolution
- Full auditability of summarization (model, prompt, inputs)
- Hash verification for tamper detection
- Structured JSON storage alongside markdown

Run: python -m episodic.migrations.m019_summary_provenance [up|down]
"""

import logging
import sqlite3

from episodic.db_connection import get_connection

logger = logging.getLogger(__name__)

MIGRATION_ID = "m019_summary_provenance"
MIGRATION_VERSION = 19


def migrate_up(conn: sqlite3.Connection = None):
    """Apply migration: add provenance columns to topic_working_set."""

    def _apply(c: sqlite3.Connection):
        cursor = c.cursor()

        # Check if table exists first
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='topic_working_set'"
        )
        if not cursor.fetchone():
            logger.warning(
                "topic_working_set table does not exist, run m015_topic_local_tables first"
            )
            return

        columns = [
            ("schema_version", "INTEGER DEFAULT 1"),
            ("summarizer_model_id", "TEXT"),
            ("prompt_hash", "TEXT"),
            ("input_start_turn_idx", "INTEGER"),
            ("input_end_turn_idx", "INTEGER"),
            ("input_node_ids_hash", "TEXT"),  # Hash of ordered list of node_ids
            ("summary_hash", "TEXT"),  # Hash of canonical JSON
            ("summary_json", "TEXT"),  # Full structured JSON
            ("canonicalizer_version", "INTEGER DEFAULT 1"),
            ("last_summarized_at", "TIMESTAMP"),
        ]

        for col_name, col_type in columns:
            try:
                cursor.execute(
                    f"ALTER TABLE topic_working_set ADD COLUMN {col_name} {col_type}"
                )
                logger.debug(f"Added column {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    logger.debug(f"Column {col_name} already exists")
                else:
                    raise

        c.commit()
        logger.info(f"Migration {MIGRATION_ID} applied successfully")

    if conn is not None:
        _apply(conn)
    else:
        with get_connection() as c:
            _apply(c)


def migrate_down(conn: sqlite3.Connection = None):
    """
    Rollback migration: SQLite doesn't support DROP COLUMN directly.

    For a full rollback, would need to:
    1. Create new table without the columns
    2. Copy data
    3. Drop old table
    4. Rename new table

    For safety, we just log a warning here.
    """

    def _rollback(c: sqlite3.Connection):
        logger.warning(
            f"Migration {MIGRATION_ID} rollback: SQLite doesn't support DROP COLUMN. "
            "Columns will remain but can be ignored."
        )

    if conn is not None:
        _rollback(conn)
    else:
        with get_connection() as c:
            _rollback(c)


def check_migration_status(conn: sqlite3.Connection = None) -> dict:
    """Check if migration has been applied."""

    def _check(c: sqlite3.Connection) -> dict:
        cursor = c.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='topic_working_set'"
        )
        if not cursor.fetchone():
            return {"applied": False, "reason": "table_not_exists"}

        # Check for the new columns
        cursor.execute("PRAGMA table_info(topic_working_set)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "schema_version",
            "summarizer_model_id",
            "prompt_hash",
            "input_start_turn_idx",
            "input_end_turn_idx",
            "input_node_ids_hash",
            "summary_hash",
            "summary_json",
            "canonicalizer_version",
            "last_summarized_at",
        }

        missing = expected_columns - columns

        if missing:
            return {
                "applied": False,
                "reason": "missing_columns",
                "missing": list(missing),
            }

        return {"applied": True}

    if conn is not None:
        return _check(conn)

    with get_connection() as c:
        return _check(c)


if __name__ == "__main__":
    import sys

    action = sys.argv[1] if len(sys.argv) > 1 else "up"

    if action == "up":
        migrate_up()
        print(f"Applied migration: {MIGRATION_ID}")
    elif action == "down":
        migrate_down()
        print(f"Rolled back migration: {MIGRATION_ID}")
    elif action == "status":
        status = check_migration_status()
        print(f"Migration {MIGRATION_ID} status: {status}")
    else:
        print(f"Usage: python -m episodic.migrations.{MIGRATION_ID} [up|down|status]")
