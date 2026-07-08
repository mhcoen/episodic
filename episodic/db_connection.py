"""
Database connection management for Episodic.

This module handles database connection setup and lifecycle.
"""

import sqlite3
import os
import threading
import contextlib
import logging
import queue
import time
from typing import Optional

from .configuration import DATABASE_FILENAME

# Set up logging
logger = logging.getLogger(__name__)

# Default database path - use user's home directory
DEFAULT_DB_PATH = os.path.expanduser(os.path.join("~/.episodic", DATABASE_FILENAME))
# Alias for backward compatibility with test scripts
DB_PATH = DEFAULT_DB_PATH

# Resolved database path (read once on first access)
_resolved_db_path = None

# Thread-local storage for database connections
_local = threading.local()

# Connection pool configuration
POOL_SIZE = 5  # Maximum number of connections in the pool
POOL_TIMEOUT = 30  # Timeout in seconds to wait for a connection
CONNECTION_MAX_AGE = 300  # Maximum age of a connection in seconds (5 minutes)

# Global connection pool
_connection_pool = None
_pool_lock = threading.Lock()


def get_db_path():
    """Get the database path (resolved once on first access)."""
    global _resolved_db_path
    if _resolved_db_path is None:
        # Path is resolved once when the connection pool is first created.
        # Changing EPISODIC_DB_PATH after that point has no effect.
        db_path = os.environ.get("EPISODIC_DB_PATH") or DEFAULT_DB_PATH

        # Validate the path to ensure it's not in the project directory
        from .db_safeguards import validate_db_path
        db_path = validate_db_path(db_path)

        # Ensure the directory exists
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

        _resolved_db_path = db_path

    return _resolved_db_path


class ConnectionPool:
    """Bounded connection pool for SQLite connections.

    Connections are created with check_same_thread=False so a pooled
    connection can be validly used from a background thread (access is
    serialized by checkout — one caller holds a connection at a time — and
    WAL mode handles the write locking). `_total` tracks the number of live
    connections (idle in the queue + checked out) and is the single source of
    truth for the pool_size limit, so idle connections are never double-counted
    (the old qsize()+len(connection_info) test saturated the pool and caused
    spurious 30s timeouts). Discarded connections are always closed.
    """

    def __init__(self, db_path: str, pool_size: int = POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.connection_info = {}  # id(conn) -> created_time, for live conns
        self._total = 0            # live connections (idle + checked out)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection (caller has reserved a slot)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            logger.debug("Could not set WAL mode (database might be locked)")
        self.connection_info[id(conn)] = time.time()
        return conn

    def _is_usable(self, conn: sqlite3.Connection) -> bool:
        created = self.connection_info.get(id(conn), 0)
        if time.time() - created > CONNECTION_MAX_AGE:
            return False
        try:
            conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _discard(self, conn: sqlite3.Connection) -> None:
        """Close and untrack a connection. Caller adjusts _total."""
        self.connection_info.pop(id(conn), None)
        try:
            conn.close()
        except Exception:
            pass

    def get_connection(self, timeout: float = POOL_TIMEOUT) -> Optional[sqlite3.Connection]:
        """Get a connection from the pool, creating or waiting as needed."""
        deadline = time.time() + timeout
        while True:
            # 1. Reuse an idle connection if one is usable.
            try:
                conn = self.pool.get(block=False)
            except queue.Empty:
                conn = None
            if conn is not None:
                if self._is_usable(conn):
                    return conn
                with self.lock:
                    self._total -= 1
                self._discard(conn)
                continue

            # 2. Create a new connection if under the limit.
            with self.lock:
                if self._total < self.pool_size:
                    self._total += 1
                    reserved = True
                else:
                    reserved = False
            if reserved:
                try:
                    return self._create_connection()
                except Exception:
                    with self.lock:
                        self._total -= 1
                    raise

            # 3. At capacity — wait for a returned connection.
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Could not get database connection within {timeout} seconds")
            try:
                conn = self.pool.get(block=True, timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"Could not get database connection within {timeout} seconds")
            if self._is_usable(conn):
                return conn
            with self.lock:
                self._total -= 1
            self._discard(conn)
            # loop and retry

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        if conn is None:
            return
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            self.pool.put(conn, block=False)
        except queue.Full:
            # No room to keep it idle — close and untrack.
            with self.lock:
                self._total -= 1
            self._discard(conn)

    def close_all(self):
        """Close all idle connections and reset accounting."""
        while True:
            try:
                conn = self.pool.get(block=False)
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                pass
        self.connection_info.clear()
        with self.lock:
            self._total = 0


def _get_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = ConnectionPool(get_db_path())
                
    return _connection_pool


@contextlib.contextmanager
def get_connection():
    """
    Get a connection to the database.

    This function returns a context manager that ensures the connection
    is properly returned to the pool when the context exits.

    Returns:
        A SQLite database connection.
    """
    # Check if pooling is disabled (e.g., for tests)
    if os.environ.get("EPISODIC_DISABLE_POOL", "").lower() == "true":
        # Fall back to creating a new connection each time
        connection = sqlite3.connect(get_db_path())
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            # Commit on the success path so a commit failure (disk full,
            # SQLITE_BUSY) propagates instead of being silently swallowed,
            # which would let callers keep return values for rolled-back writes.
            connection.commit()
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()
        return

    # Use connection pool
    pool = _get_pool()
    connection = None

    try:
        connection = pool.get_connection()
        yield connection
        # Commit on the success path; a failing commit must reach the caller.
        connection.commit()
    except Exception:
        # Body error OR commit failure: rollback and re-raise.
        if connection:
            try:
                connection.rollback()
            except Exception:
                pass
        raise
    finally:
        # Return connection to pool regardless of outcome.
        if connection:
            pool.return_connection(connection)


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


def close_pool():
    """Close all connections in the pool. Call this on application shutdown."""
    global _connection_pool, _resolved_db_path

    if _connection_pool:
        _connection_pool.close_all()
        _connection_pool = None

    # Reset the resolved path so it will be re-read from env on next access
    _resolved_db_path = None
