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
    """Get the database path from the environment variable or use the default."""
    db_path = os.environ.get("EPISODIC_DB_PATH", DEFAULT_DB_PATH)
    
    # Validate the path to ensure it's not in the project directory
    from .db_safeguards import validate_db_path
    db_path = validate_db_path(db_path)
    
    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")
    
    return db_path


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


class ConnectionPool:
    """Simple connection pool for SQLite connections."""
    
    def __init__(self, db_path: str, pool_size: int = POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.connection_info = {}  # Track connection creation time
        
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        # Enable query optimization
        conn.execute("PRAGMA optimize")
        return conn
        
    def get_connection(self, timeout: float = POOL_TIMEOUT) -> Optional[sqlite3.Connection]:
        """Get a connection from the pool."""
        try:
            # Try to get an existing connection
            conn = self.pool.get(block=False)
            
            # Check if connection is still valid and not too old
            created_time = self.connection_info.get(id(conn), 0)
            if time.time() - created_time > CONNECTION_MAX_AGE:
                # Connection is too old, close it and create a new one
                try:
                    conn.close()
                except:
                    pass
                conn = self._create_connection()
                self.connection_info[id(conn)] = time.time()
                
            # Test the connection
            try:
                conn.execute("SELECT 1")
            except:
                # Connection is dead, create a new one
                conn = self._create_connection()
                self.connection_info[id(conn)] = time.time()
                
            return conn
            
        except queue.Empty:
            # No connections available, create a new one if under limit
            with self.lock:
                if self.pool.qsize() + len(self.connection_info) < self.pool_size:
                    conn = self._create_connection()
                    self.connection_info[id(conn)] = time.time()
                    return conn
                    
            # Wait for a connection to become available
            try:
                conn = self.pool.get(block=True, timeout=timeout)
                # Validate connection as above
                try:
                    conn.execute("SELECT 1")
                except:
                    conn = self._create_connection()
                    self.connection_info[id(conn)] = time.time()
                return conn
            except queue.Empty:
                raise TimeoutError(f"Could not get database connection within {timeout} seconds")
                
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        if conn is None:
            return
            
        try:
            # Reset the connection state
            conn.rollback()
            # Return to pool if there's space
            self.pool.put(conn, block=False)
        except queue.Full:
            # Pool is full, close the connection
            try:
                conn.close()
            except:
                pass
            # Remove from tracking
            self.connection_info.pop(id(conn), None)
            
    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get(block=False)
                conn.close()
            except:
                pass
        self.connection_info.clear()


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
        try:
            yield connection
        except Exception:
            connection.rollback()
            connection.close()
            raise
        finally:
            try:
                connection.commit()
            except Exception:
                try:
                    connection.rollback()
                except Exception:
                    logger.debug("Failed to rollback transaction during cleanup")
            finally:
                connection.close()
        return
    
    # Use connection pool
    pool = _get_pool()
    connection = None
    
    try:
        connection = pool.get_connection()
        yield connection
    except Exception:
        # If an exception occurs, rollback and re-raise
        if connection:
            try:
                connection.rollback()
            except:
                pass
        raise
    finally:
        # Return connection to pool
        if connection:
            try:
                connection.commit()
            except Exception:
                try:
                    connection.rollback()
                except Exception:
                    logger.debug("Failed to rollback transaction during cleanup")
            finally:
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
    global _connection_pool
    
    if _connection_pool:
        _connection_pool.close_all()
        _connection_pool = None