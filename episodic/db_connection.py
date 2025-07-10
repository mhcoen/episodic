"""
Database connection management for Episodic.

This module handles database connection setup and lifecycle.
"""

import sqlite3
import os
import threading
import contextlib
import logging

from .configuration import DATABASE_FILENAME

# Set up logging
logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", DATABASE_FILENAME))
# Alias for backward compatibility with test scripts
DB_PATH = DEFAULT_DB_PATH

# Thread-local storage for database connections
_local = threading.local()


def get_db_path():
    """Get the database path from the environment variable or use the default."""
    return os.environ.get("EPISODIC_DB_PATH", DEFAULT_DB_PATH)


@contextlib.contextmanager
def get_connection():
    """
    Get a connection to the database.

    This function returns a context manager that ensures the connection
    is properly closed when the context exits.

    Returns:
        A SQLite database connection.
    """
    # Always create a new connection to avoid issues with thread-local storage
    # in multi-threaded environments like WebSocket tests
    connection = sqlite3.connect(get_db_path())

    try:
        # Yield the connection to the caller
        yield connection
    except Exception:
        # If an exception occurs, rollback and close the connection, then re-raise
        connection.rollback()
        connection.close()
        raise
    finally:
        # Commit any pending changes and close the connection
        try:
            connection.commit()
        except Exception:
            # If commit fails, try to rollback
            try:
                connection.rollback()
            except Exception:
                # Rollback failed, connection is likely already closed
                logger.debug("Failed to rollback transaction during cleanup")
        finally:
            connection.close()


def database_exists():
    """Check if the database file exists and has tables."""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return False

    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
            return c.fetchone() is not None
    except sqlite3.Error:
        return False